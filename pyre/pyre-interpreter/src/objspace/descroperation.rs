//! pypy/objspace/descroperation.py — binary/unary operation dispatch.
//!
//! The ObjSpace mediates all operations on Python objects. This module
//! contains the dispatch layer that routes `+`, `-`, `*`, `//`, `%`,
//! `**`, `<<`, `>>`, `&`, `|`, `^`, comparisons, and unary `+`/`-`/`~`
//! through type-specific fast paths and then the dunder protocol.
#![allow(non_camel_case_types, non_snake_case)]
#![allow(unsafe_op_in_unsafe_fn)]

use majit_rlib::rbigint::{RBigInt as BigInt, RBigIntGcRoot};
use num_traits::ToPrimitive;

use pyre_object::unicodeobject::is_str;
use pyre_object::*;
use rustpython_wtf8::Wtf8Buf;

use crate::baseobjspace::{
    getattr, getitem, is_true, issubtype_w, lookup, lookup_in_type, lookup_in_type_where,
    lookup_where_with_method_cache, p_abstract_issubclass_w,
};
pub use crate::{PyError, PyErrorKind, PyResult};

/// Every zero-divisor `ZeroDivisionError` carries this one message, whatever
/// the operator and whatever the operand types — int, long, float or complex,
/// `/`, `//`, `%` or `divmod`. The per-operator and per-type wordings
/// ("integer division or modulo by zero", "float modulo", …) were unified in
/// 3.12. `0 ** -1` is the one ZeroDivisionError that keeps its own message
/// ("zero to a negative power"), because it is not a division.
const ZERO_DIVISION_MSG: &str = "division by zero";

// ── BigInt helpers ──────────────────────────────────────────────────

/// Box a BigInt result, demoting to W_IntObject if it fits in i64.

pub(crate) fn bigint_result(value: BigInt) -> PyObjectRef {
    if jit_bigint_to_i64_fits(&value) != 0 {
        w_int_new(jit_bigint_to_i64_value(&value))
    } else {
        w_long_new(value)
    }
}

/// CPython `long_invmod` / PyPy three-argument `pow` inverse step.
///
/// Keep the Bézout coefficients as rbigints: the loop is the ordinary
/// extended Euclidean algorithm and never crosses into an opaque host bigint.
fn bigint_mod_inverse(base: &BigInt, modulus: &BigInt) -> Result<BigInt, PyError> {
    let mut old_r = base
        .r#mod(modulus)
        .map_err(|_| PyError::value_error("pow() 3rd argument cannot be 0"))?;
    let mut r = modulus.translated_alias();
    let mut old_s = BigInt::one();
    let mut s = BigInt::zero();
    while !r.is_zero() {
        let quotient = old_r
            .floordiv(&r)
            .map_err(|_| PyError::value_error("base is not invertible for the given modulus"))?;
        let next_r = old_r.sub(&quotient.mul(&r));
        old_r = r;
        r = next_r;
        let next_s = old_s.sub(&quotient.mul(&s));
        old_s = s;
        s = next_s;
    }
    if !old_r.int_eq(1) {
        return Err(PyError::value_error(
            "base is not invertible for the given modulus",
        ));
    }
    old_s
        .r#mod(modulus)
        .map_err(|_| PyError::value_error("pow() 3rd argument cannot be 0"))
}

#[majit_macros::elidable]
fn bigint_add(a: BigInt, b: BigInt) -> BigInt {
    a + b
}

#[majit_macros::elidable]
fn bigint_sub(a: BigInt, b: BigInt) -> BigInt {
    a - b
}

#[majit_macros::elidable]
fn bigint_mul(a: BigInt, b: BigInt) -> BigInt {
    a * b
}

/// Host spelling of the already-zero-checked quotient half of
/// `rbigint.divmod`. The MIR front retargets this exact seam to
/// `jit_bigint_div_floor`, restoring RPython's one-GCREF CALL_PURE result
/// instead of exposing Rust's `Result<(RBigInt, RBigInt), RBigIntError>`.
#[majit_macros::jit_elidable]
fn bigint_floordiv_nonzero(a: &BigInt, b: &BigInt) -> BigInt {
    a.divmod(b).expect("divisor was checked nonzero").0
}

/// Remainder companion of [`bigint_floordiv_nonzero`].
#[majit_macros::jit_elidable]
fn bigint_modulo_nonzero(a: &BigInt, b: &BigInt) -> BigInt {
    a.divmod(b).expect("divisor was checked nonzero").1
}

/// Machine-int-divisor form of [`bigint_floordiv_nonzero`]
/// (`longobject.py:418 _int_floordiv` → `rbigint.int_floordiv`). The dedicated
/// leg divides by a single digit instead of first materializing an rbigint for
/// the `W_IntObject`.
#[majit_macros::jit_elidable]
fn bigint_int_floordiv_nonzero(a: &BigInt, b: i64) -> BigInt {
    a.int_floordiv(b).expect("divisor was checked nonzero")
}

/// Machine-int-divisor form of [`bigint_modulo_nonzero`]
/// (`longobject.py:435 _int_mod` → `rbigint.int_mod_int_result`). The remainder
/// of a long by a machine int always fits a machine int, so the descriptor
/// returns `space.newint` and this leg allocates no result bigint.
#[majit_macros::jit_elidable]
fn bigint_int_modulo_int_result_nonzero(a: &BigInt, b: i64) -> i64 {
    a.int_mod_int_result(b)
        .expect("divisor was checked nonzero")
}

#[majit_macros::elidable]
fn bigint_and(a: BigInt, b: BigInt) -> BigInt {
    a & b
}

#[majit_macros::elidable]
fn bigint_or(a: BigInt, b: BigInt) -> BigInt {
    a | b
}

#[majit_macros::elidable]
fn bigint_xor(a: BigInt, b: BigInt) -> BigInt {
    a ^ b
}

#[majit_macros::elidable]
fn bigint_lshift(a: &BigInt, shift: i64) -> Result<BigInt, majit_rlib::rbigint::RBigIntError> {
    // longobject.py passes the result of rbigint.toint(), an RPython Signed.
    // Keep that word signed and pointer-width independent all the way to
    // rbigint.lshift; narrowing through usize truncates valid counts on wasm32.
    a.lshift(shift)
}

#[majit_macros::elidable]
fn bigint_rshift(a: &BigInt, shift: i64) -> BigInt {
    // Same Signed count shape as bigint_lshift.  The caller has rejected a
    // negative value, so rshift cannot fail here.
    a.rshift(shift, false).expect("nonnegative shift")
}

/// Exact `rbigint._divrem` early-return predicate (`rbigint.py:2406-2411`).
///
/// Upstream deliberately tests only the digit count and most-significant
/// digit here. When it succeeds, the remainder is the input object `a`
/// itself, so pointer-ABI residuals must not allocate a shallow copy.
#[inline]
fn divrem_returns_input_as_remainder(a: &BigInt, b: &BigInt) -> bool {
    let size_a = a.numdigits();
    let size_b = b.numdigits();
    size_a < size_b
        || (size_a == size_b && a.digit((size_a - 1).abs()) < b.digit((size_b - 1).abs()))
}

/// Host form of `rbigint.pow(a, b, None)` used by `long_pow`.
///
/// The MIR front erases this Rust `Result` carrier back to RPython's implicit
/// exception edge and retargets the payload call to `jit_bigint_pow_nomod`.
/// Keeping the conversion here lets the ordinary interpreter retain its
/// native `PyResult` contract without exposing `RBigIntError` to translated
/// callers.
#[majit_macros::dont_look_inside]
fn bigint_pow_nomod(a: &BigInt, b: &BigInt) -> Result<BigInt, PyError> {
    // `rbigint.pow` performs several collecting digit-array allocations while
    // both operands remain live. RPython's GC transform roots those incoming
    // handles for the whole call.
    let a = RBigIntGcRoot::new(a.translated_alias());
    let b = RBigIntGcRoot::new(b.translated_alias());
    a.pow(&b, None)
        .map_err(|_| PyError::memory_error("exponent too large"))
}

/// Machine-int-exponent form of [`bigint_pow_nomod`]. `descr_pow` keeps a
/// `W_IntObject` exponent unwrapped and calls `rbigint.int_pow`
/// (`longobject.py:230`); only a long exponent reaches `rbigint.pow`.
#[majit_macros::dont_look_inside]
fn bigint_int_pow_nomod(a: &BigInt, b: i64) -> Result<BigInt, PyError> {
    // `int_pow` also keeps the base live across its intermediate allocations.
    let a = RBigIntGcRoot::new(a.translated_alias());
    a.int_pow(b, None)
        .map_err(|_| PyError::memory_error("exponent too large"))
}

/// Source-level spelling of `rbigint.lshift`'s implicit MemoryError edge.
///
/// `long_lshift` has already validated the count exactly like
/// longobject.py:372-381.  The MIR front retargets this Result carrier to the
/// GCREF-returning `jit_bigint_lshift_count` residual, just as it does for
/// `bigint_pow_nomod`.
#[majit_macros::dont_look_inside]
fn bigint_lshift_count(a: &BigInt, shift: i64) -> Result<BigInt, PyError> {
    a.lshift(shift).map_err(|error| match error {
        majit_rlib::rbigint::RBigIntError::Memory => PyError::memory_error(""),
        _ => PyError::value_error("negative shift count"),
    })
}

/// Machine-int-specialized counterpart of [`bigint_lshift_count`].
///
/// intobject.py:861-868 calls
/// `rbigint.lshift_int_int_bigint_result(a, b)` directly on two Signed words.
/// Rust exposes its implicit MemoryError as `Result`; the MIR front retargets
/// this exact carrier to `jit_bigint_lshift_int_int_result`, preserving the
/// specialized no-input-rbigint-allocation shape.
#[majit_macros::dont_look_inside]
fn bigint_lshift_int_int_result(iself: i64, shift: i64) -> Result<BigInt, PyError> {
    BigInt::lshift_int_int_bigint_result(iself, shift).map_err(|error| match error {
        majit_rlib::rbigint::RBigIntError::Memory => PyError::memory_error(""),
        _ => PyError::value_error("negative shift count"),
    })
}

#[majit_macros::elidable]
fn bigint_to_f64(a: BigInt) -> f64 {
    jit_bigint_to_f64_or_inf(&a)
}

/// `rbigint.bit_length` / `rbigint.bit_count` scalar residual core.
///
/// Both methods are `@jit.elidable` and can raise OverflowError through
/// RPython's `ovfcheck`, so these wrappers publish the exception for the
/// trailing `GUARD_NO_EXCEPTION` instead of using a cannot-raise residual.
#[inline]
fn bigint_checked_scalar(value: &BigInt, bit_count: bool) -> i64 {
    let result = if bit_count {
        value.bit_count()
    } else {
        value.bit_length()
    };
    match result {
        Ok(value) => value,
        Err(majit_rlib::rbigint::RBigIntError::Overflow) => {
            crate::runtime_ops::jit_publish_exception(
                PyError::overflow_error("too many digits in integer").to_exc_object(),
            );
            0
        }
        Err(_) => unreachable!("bit_length/bit_count only raise OverflowError"),
    }
}

#[majit_macros::elidable]
pub extern "C" fn jit_bigint_bit_length(value: i64) -> i64 {
    let value = unsafe { &*(value as *const BigInt) };
    bigint_checked_scalar(value, false)
}

#[majit_macros::elidable]
pub extern "C" fn jit_bigint_bit_count(value: i64) -> i64 {
    let value = unsafe { &*(value as *const BigInt) };
    bigint_checked_scalar(value, true)
}

/// RPython `_divrem`'s truncated quotient over two bare RBigInt payloads.
/// Allocates the result via the COLLECTING nursery (a gcmap-rooted residual,
/// its operand pointers rooted across the alloc), matching the arithmetic
/// residuals. Returns a freshly heap-allocated `*mut BigInt` encoded as the
/// JIT's uniform i64 word; the MIR retarget keeps the result modeled as GcRef.
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_div(a: i64, b: i64) -> pyre_object::longobject::JitBigIntResult {
    let (a, b) = (a as *const BigInt, b as *const BigInt);
    unsafe {
        pyre_object::longobject::encode_jit_bigint_result(
            pyre_object::longobject::alloc_bigint_nursery_collecting(
                majit_rlib::rbigint::_divrem(&*a, &*b)
                    .expect("division by zero")
                    .0,
            ),
        )
    }
}

/// `_divrem`'s truncated remainder.
/// See [`jit_bigint_div`]; both project the same upstream helper.
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_rem(a: i64, b: i64) -> pyre_object::longobject::JitBigIntResult {
    let (a, b) = (a as *const BigInt, b as *const BigInt);
    unsafe {
        if (&*b).get_sign() != 0 && divrem_returns_input_as_remainder(&*a, &*b) {
            return pyre_object::longobject::encode_jit_bigint_result(a as *mut BigInt);
        }
        pyre_object::longobject::encode_jit_bigint_result(
            pyre_object::longobject::alloc_bigint_nursery_collecting(
                majit_rlib::rbigint::_divrem(&*a, &*b)
                    .expect("division by zero")
                    .1,
            ),
        )
    }
}

/// `rbigint.divmod`'s floored quotient projection. Allocates via the collecting nursery
/// (a gcmap-rooted residual, its operand pointers rooted across the alloc),
/// matching the arithmetic residuals. Returns a freshly heap-allocated
/// `*mut BigInt` encoded as the JIT's uniform i64 word; the MIR retarget keeps
/// the result modeled as a traced GcRef.
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_div_floor(a: i64, b: i64) -> pyre_object::longobject::JitBigIntResult {
    let (a, b) = (a as *const BigInt, b as *const BigInt);
    unsafe {
        pyre_object::longobject::encode_jit_bigint_result(
            pyre_object::longobject::alloc_bigint_nursery_collecting(
                (&*a).divmod(&*b).expect("division by zero").0,
            ),
        )
    }
}

/// Machine-int-divisor quotient (`longobject.py:418 _int_floordiv` →
/// `rbigint.int_floordiv`). `b` is a bare machine word, not a payload pointer.
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_int_div_floor(
    a: i64,
    b: i64,
) -> pyre_object::longobject::JitBigIntResult {
    let a = a as *const BigInt;
    unsafe {
        // rbigint.int_floordiv: a positive bigint divided by +1 returns
        // `self`, i.e. the identical translated GC reference.
        if b == 1 && (&*a).get_sign() == 1 {
            return pyre_object::longobject::encode_jit_bigint_result(a as *mut BigInt);
        }
        pyre_object::longobject::encode_jit_bigint_result(
            pyre_object::longobject::alloc_bigint_nursery_collecting(
                (&*a).int_floordiv(b).expect("division by zero"),
            ),
        )
    }
}

/// Machine-int-divisor remainder (`longobject.py:435 _int_mod` →
/// `rbigint.int_mod_int_result`). The remainder of a long by a machine int
/// always fits a machine int, so this residual returns the value itself and
/// allocates nothing.
#[majit_macros::elidable]
pub extern "C" fn jit_bigint_int_mod_int_result(a: i64, b: i64) -> i64 {
    let a = a as *const BigInt;
    unsafe { (&*a).int_mod_int_result(b).expect("division by zero") }
}

/// Both halves of a machine-int divmod (`longobject.py:451 _int_divmod` →
/// `rbigint.int_divmod`, rbigint.py:1050 `@jit.elidable`). `b` is a bare
/// machine word, not a payload pointer.
///
/// One call produces both results: `//` and `%` each reach `_divmod`, so
/// splitting this into two residuals would run the division twice. The result
/// is the RPython `tuple2` the elidable returns, and the caller reads its two
/// halves with `getfield_gc_r` before boxing each as a `W_LongObject` — the
/// wrappers and the interpreter tuple stay in traced code, where they remain
/// virtualizable.
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_int_divmod(
    a: i64,
    b: i64,
) -> pyre_object::longobject::JitBigIntPairResult {
    let a = a as *const BigInt;
    unsafe {
        let (div, modulo) = (&*a).int_divmod(b).expect("division by zero");
        pyre_object::longobject::encode_jit_bigint_pair_result(
            pyre_object::longobject::alloc_bigint_pair_nursery_collecting(div, modulo),
        )
    }
}

/// `rbigint.divmod`'s floored modulus projection.
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_mod_floor(a: i64, b: i64) -> pyre_object::longobject::JitBigIntResult {
    let (a, b) = (a as *const BigInt, b as *const BigInt);
    unsafe {
        let selfsign = (&*a).get_sign();
        let othersign = (&*b).get_sign();
        // `divmod` routes one-digit divisors through `int_divmod`, which
        // constructs its remainder. Otherwise `_divmod_small` preserves
        // `_divrem`'s `a` remainder when no opposite-sign correction occurs.
        if selfsign != 0
            && othersign != 0
            && selfsign == othersign
            && (&*b).numdigits() != 1
            && divrem_returns_input_as_remainder(&*a, &*b)
        {
            return pyre_object::longobject::encode_jit_bigint_result(a as *mut BigInt);
        }
        pyre_object::longobject::encode_jit_bigint_result(
            pyre_object::longobject::alloc_bigint_nursery_collecting(
                (&*a).divmod(&*b).expect("division by zero").1,
            ),
        )
    }
}

// ── BigInt binary-operator residuals ─────────────────────────────────
// Rust trait-operator shells (`<RBigInt as BitAnd>::bitand`, …) obscure the
// RPython method identity in LLBC, so a traced-into caller emits
// an unregistered `<Impl>` FunctionPath call the census Skips. `front::mir`
// retargets each such call — guarded on both operands resolving to the
// opaque `BigInt` ADT — to the matching residual below. Both operands and
// the result are the classdef-less `*mut BigInt` GcRef the front models a
// `BigInt` as: the operands arrive as i64-encoded pointers (a faithful ABI
// pass) and the result pointer is returned in the same uniform i64 word ABI;
// the retarget preserves the front's `Ref(None)` result type. Each
// allocates the fresh result in the collecting nursery, its operand pointers
// rooted across the alloc. These wrappers retain RPython's elidable effect:
// allocation-only operations use `EF_ELIDABLE_OR_MEMORYERROR`.

/// `rbigint.and_` payload — `&BigInt & &BigInt`. See the module note above.
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_and(a: i64, b: i64) -> pyre_object::longobject::JitBigIntResult {
    let (a, b) = (a as *const BigInt, b as *const BigInt);
    unsafe {
        pyre_object::longobject::encode_jit_bigint_result(
            pyre_object::longobject::alloc_bigint_nursery_collecting(&*a & &*b),
        )
    }
}

/// `rbigint.or_` payload — `&BigInt | &BigInt`. See [`jit_bigint_and`].
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_or(a: i64, b: i64) -> pyre_object::longobject::JitBigIntResult {
    let (a, b) = (a as *const BigInt, b as *const BigInt);
    unsafe {
        pyre_object::longobject::encode_jit_bigint_result(
            pyre_object::longobject::alloc_bigint_nursery_collecting(&*a | &*b),
        )
    }
}

/// `rbigint.xor_` payload — `&BigInt ^ &BigInt`. See [`jit_bigint_and`].
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_xor(a: i64, b: i64) -> pyre_object::longobject::JitBigIntResult {
    let (a, b) = (a as *const BigInt, b as *const BigInt);
    unsafe {
        pyre_object::longobject::encode_jit_bigint_result(
            pyre_object::longobject::alloc_bigint_nursery_collecting(&*a ^ &*b),
        )
    }
}

/// `rbigint.sub` payload — `&BigInt - &BigInt`. See [`jit_bigint_and`].
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_sub(a: i64, b: i64) -> pyre_object::longobject::JitBigIntResult {
    let (a, b) = (a as *const BigInt, b as *const BigInt);
    unsafe {
        if (&*b).get_sign() == 0 {
            return pyre_object::longobject::encode_jit_bigint_result(a as *mut BigInt);
        }
        pyre_object::longobject::encode_jit_bigint_result(
            pyre_object::longobject::alloc_bigint_nursery_collecting(&*a - &*b),
        )
    }
}

/// `rbigint.mul` payload — `&BigInt * &BigInt`. See [`jit_bigint_and`].
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_mul(a: i64, b: i64) -> pyre_object::longobject::JitBigIntResult {
    let (a, b) = (a as *const BigInt, b as *const BigInt);
    unsafe {
        pyre_object::longobject::encode_jit_bigint_result(
            pyre_object::longobject::alloc_bigint_nursery_collecting(&*a * &*b),
        )
    }
}

/// `rbigint.add` payload — `&BigInt + &BigInt`. See [`jit_bigint_and`].
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_add(a: i64, b: i64) -> pyre_object::longobject::JitBigIntResult {
    let (a, b) = (a as *const BigInt, b as *const BigInt);
    unsafe {
        if (&*a).get_sign() == 0 {
            return pyre_object::longobject::encode_jit_bigint_result(b as *mut BigInt);
        }
        if (&*b).get_sign() == 0 {
            return pyre_object::longobject::encode_jit_bigint_result(a as *mut BigInt);
        }
        pyre_object::longobject::encode_jit_bigint_result(
            pyre_object::longobject::alloc_bigint_nursery_collecting(&*a + &*b),
        )
    }
}

// ── BigInt/machine-int arithmetic residuals ─────────────────────────
//
// pypy/objspace/std/longobject.py:_make_generic_descr_binop and descr_sub
// call rbigint.int_{add,sub,mul,and_,or_,xor} whenever the other operand is
// a W_IntObject.  Those methods are @jit.elidable and return one rbigint GC
// reference.  The interpreter LLBC sees RBigInt as an opaque dependency ADT,
// so front::mir retargets the inherent method calls to these pointer-ABI
// residuals instead of allocating a temporary RBigInt for the machine word.

macro_rules! bigint_int_residual {
    ($name:ident, $method:ident) => {
        #[doc = "Bare-RBigInt/machine-int residual using the translated GC-reference ABI."]
        #[majit_macros::elidable_or_memerror]
        pub extern "C" fn $name(a: i64, b: i64) -> pyre_object::longobject::JitBigIntResult {
            let a = a as *const BigInt;
            unsafe {
                pyre_object::longobject::encode_jit_bigint_result(
                    pyre_object::longobject::alloc_bigint_nursery_collecting((&*a).$method(b)),
                )
            }
        }
    };
}

#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_int_add(a: i64, b: i64) -> pyre_object::longobject::JitBigIntResult {
    let a = a as *const BigInt;
    unsafe {
        if b == 0 && (&*a).get_sign() != 0 {
            return pyre_object::longobject::encode_jit_bigint_result(a as *mut BigInt);
        }
        pyre_object::longobject::encode_jit_bigint_result(
            pyre_object::longobject::alloc_bigint_nursery_collecting((&*a).int_add(b)),
        )
    }
}

#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_int_sub(a: i64, b: i64) -> pyre_object::longobject::JitBigIntResult {
    let a = a as *const BigInt;
    unsafe {
        if b == 0 {
            return pyre_object::longobject::encode_jit_bigint_result(a as *mut BigInt);
        }
        pyre_object::longobject::encode_jit_bigint_result(
            pyre_object::longobject::alloc_bigint_nursery_collecting((&*a).int_sub(b)),
        )
    }
}

#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_int_mul(a: i64, b: i64) -> pyre_object::longobject::JitBigIntResult {
    let a = a as *const BigInt;
    unsafe {
        if b == 1 && (&*a).get_sign() != 0 {
            return pyre_object::longobject::encode_jit_bigint_result(a as *mut BigInt);
        }
        pyre_object::longobject::encode_jit_bigint_result(
            pyre_object::longobject::alloc_bigint_nursery_collecting((&*a).int_mul(b)),
        )
    }
}

bigint_int_residual!(jit_bigint_int_and, int_and_);
bigint_int_residual!(jit_bigint_int_or, int_or_);
bigint_int_residual!(jit_bigint_int_xor, int_xor);

macro_rules! bigint_int_comparison_residual {
    ($name:ident, $method:ident) => {
        #[doc = "Bare-RBigInt/machine-int comparison residual using the translated GC-reference ABI."]
        // rbigint.py `_make_int_comparison`: these helpers only inspect the
        // existing digits and return a bool; unlike the arithmetic int_*
        // family above, they allocate nothing and cannot raise.
        #[majit_macros::elidable_cannot_raise]
        pub extern "C" fn $name(a: i64, b: i64) -> i64 {
            let a = a as *const BigInt;
            unsafe { (&*a).$method(b) as i64 }
        }
    };
}

bigint_int_comparison_residual!(jit_bigint_int_eq, int_eq);
bigint_int_comparison_residual!(jit_bigint_int_ne, int_ne);
bigint_int_comparison_residual!(jit_bigint_int_lt, int_lt);
bigint_int_comparison_residual!(jit_bigint_int_le, int_le);
bigint_int_comparison_residual!(jit_bigint_int_gt, int_gt);
bigint_int_comparison_residual!(jit_bigint_int_ge, int_ge);

/// `rbigint.pow(a, b, None)` after `longobject._pow_nomod` has rejected a
/// negative exponent and handled the 0/±1 fast paths.  In that domain the
/// only remaining explicit rbigint failure is MemoryError.  RPython carries
/// it as the implicit exception edge of an `EF_ELIDABLE_OR_MEMORYERROR` call;
/// publish the same backend exception and return an ignored null payload.
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_pow_nomod(a: i64, b: i64) -> pyre_object::longobject::JitBigIntResult {
    let (a, b) = (a as *const BigInt, b as *const BigInt);
    // rbigint.pow(..., modulus=None): exponent +1 returns `self`.
    if unsafe { (&*a).get_sign() != 0 && (&*b).int_eq(1) } {
        return pyre_object::longobject::encode_jit_bigint_result(a as *mut BigInt);
    }
    // The residual call can collect inside `rbigint.pow`; mirror the roots
    // inserted for both arguments by RPython's GC transform instead of
    // depending on a backend-specific native call map.
    let a = RBigIntGcRoot::new(unsafe { (&*a).translated_alias() });
    let b = RBigIntGcRoot::new(unsafe { (&*b).translated_alias() });
    match a.pow(&b, None) {
        Ok(value) => pyre_object::longobject::encode_jit_bigint_result(
            pyre_object::longobject::alloc_bigint_nursery_collecting(value),
        ),
        Err(_) => {
            crate::runtime_ops::jit_publish_exception(
                pyre_object::interp_exceptions::memory_error_singleton(),
            );
            pyre_object::longobject::encode_jit_bigint_result(std::ptr::null_mut())
        }
    }
}

/// Machine-int-exponent form of [`jit_bigint_pow_nomod`]
/// (`longobject.py:230` → `rbigint.int_pow`). `b` is a bare machine word.
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_int_pow_nomod(
    a: i64,
    b: i64,
) -> pyre_object::longobject::JitBigIntResult {
    let a = a as *const BigInt;
    // rbigint.int_pow(..., modulus=None): exponent 1 returns `self`.
    if b == 1 && unsafe { (&*a).get_sign() != 0 } {
        return pyre_object::longobject::encode_jit_bigint_result(a as *mut BigInt);
    }
    // `int_pow` retains the base across intermediate collecting allocations.
    let a = RBigIntGcRoot::new(unsafe { (&*a).translated_alias() });
    match a.int_pow(b, None) {
        Ok(value) => pyre_object::longobject::encode_jit_bigint_result(
            pyre_object::longobject::alloc_bigint_nursery_collecting(value),
        ),
        Err(_) => {
            crate::runtime_ops::jit_publish_exception(
                pyre_object::interp_exceptions::memory_error_singleton(),
            );
            pyre_object::longobject::encode_jit_bigint_result(std::ptr::null_mut())
        }
    }
}

/// `rbigint.lshift(a, machine_count)` after the Python-level sign/range
/// checks. RPython exposes only its implicit MemoryError edge.
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_lshift_count(
    a: i64,
    shift: i64,
) -> pyre_object::longobject::JitBigIntResult {
    let a = a as *const BigInt;
    // rbigint.lshift returns `self` for both a zero count and a zero base.
    if shift == 0 || unsafe { (&*a).get_sign() == 0 } {
        return pyre_object::longobject::encode_jit_bigint_result(a as *mut BigInt);
    }
    match unsafe { (&*a).lshift(shift) } {
        Ok(value) => pyre_object::longobject::encode_jit_bigint_result(
            pyre_object::longobject::alloc_bigint_nursery_collecting(value),
        ),
        Err(_) => {
            crate::runtime_ops::jit_publish_exception(
                pyre_object::interp_exceptions::memory_error_singleton(),
            );
            pyre_object::longobject::encode_jit_bigint_result(std::ptr::null_mut())
        }
    }
}

/// intobject.py's Signed×Signed specialized overflow leg. This is the direct
/// pointer-ABI form of [`bigint_lshift_int_int_result`].
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_lshift_int_int_result(
    iself: i64,
    shift: i64,
) -> pyre_object::longobject::JitBigIntResult {
    match BigInt::lshift_int_int_bigint_result(iself, shift) {
        Ok(value) => pyre_object::longobject::encode_jit_bigint_result(
            pyre_object::longobject::alloc_bigint_nursery_collecting(value),
        ),
        Err(_) => {
            crate::runtime_ops::jit_publish_exception(
                pyre_object::interp_exceptions::memory_error_singleton(),
            );
            pyre_object::longobject::encode_jit_bigint_result(std::ptr::null_mut())
        }
    }
}

/// `rbigint.neg` payload — `-&BigInt`. A unary operator, so a single operand
/// pointer; the result is a fresh negated `BigInt`. See [`jit_bigint_and`].
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_neg(a: i64) -> pyre_object::longobject::JitBigIntResult {
    let a = a as *const BigInt;
    unsafe {
        pyre_object::longobject::encode_jit_bigint_result(
            // rbigint.py:1299-1301 always constructs a fresh rbigint handle,
            // including for zero; only its immutable digits are shared.
            majit_rlib::rbigint::alloc_rbigint_clone_nursery_collecting((&*a).neg()),
        )
    }
}

/// Whether `rbigint.divmod(a, b)` reaches `_divrem`'s literal
/// `(NULLRBIGINT, a)` return without a floored-sign adjustment.
///
/// This pointer-only predicate is a translation seam for W_LongObject
/// consumers that must put the returned operand payload into a fresh wrapper.
/// Keeping it residual avoids rebuilding the opaque RBigInt field graph in
/// `long_mod` while retaining an elidable/cannot-raise operation.
#[majit_macros::elidable_cannot_raise]
pub extern "C" fn jit_bigint_divrem_returns_lhs_remainder(a: i64, b: i64) -> i64 {
    let (a, b) = (a as *const BigInt, b as *const BigInt);
    unsafe {
        let a = &*a;
        let b = &*b;
        let b_size = b.numdigits();
        (a.get_sign() != 0
            && a.get_sign() == b.get_sign()
            && b_size != 1
            && divrem_returns_input_as_remainder(a, b)) as i64
    }
}

/// `rbigint.invert` payload. Zero returns the canonical -1 prebuilt; every
/// other input follows `int_add(1)` and sign inversion.
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_invert(a: i64) -> pyre_object::longobject::JitBigIntResult {
    let a = a as *const BigInt;
    unsafe {
        pyre_object::longobject::encode_jit_bigint_result(
            pyre_object::longobject::alloc_bigint_nursery_collecting((&*a).invert()),
        )
    }
}

// ── BigInt shift-by-`usize` residuals ────────────────────────────────
// `<BigInt as Shl<usize>>::shl` / `Shr<usize>::shr` — the shift amount is a
// plain machine integer, NOT a `BigInt`, so `b` is the count value itself
// (not a pointer). `front::mir` retargets these separately, guarded on the
// first operand being the opaque `BigInt` ADT and the second being an
// integer. See the operator-residual module note above.

/// `rbigint.lshift` by a machine `usize` — `&BigInt << (b as usize)`.
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_shl(a: i64, b: i64) -> pyre_object::longobject::JitBigIntResult {
    let a = a as *const BigInt;
    unsafe {
        if b == 0 || (&*a).get_sign() == 0 {
            return pyre_object::longobject::encode_jit_bigint_result(a as *mut BigInt);
        }
        pyre_object::longobject::encode_jit_bigint_result(
            pyre_object::longobject::alloc_bigint_nursery_collecting(&*a << (b as usize)),
        )
    }
}

/// `rbigint.rshift` by a machine `usize` — `&BigInt >> (b as usize)`.
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_bigint_shr(a: i64, b: i64) -> pyre_object::longobject::JitBigIntResult {
    let a = a as *const BigInt;
    unsafe {
        if b == 0 {
            return pyre_object::longobject::encode_jit_bigint_result(a as *mut BigInt);
        }
        pyre_object::longobject::encode_jit_bigint_result(
            pyre_object::longobject::alloc_bigint_nursery_collecting(&*a >> (b as usize)),
        )
    }
}

#[majit_macros::elidable]
fn float_copysign(mag: f64, sign: f64) -> f64 {
    if sign.is_sign_negative() {
        -mag.abs()
    } else {
        mag.abs()
    }
}

#[majit_macros::elidable]
fn bigint_neg(a: &BigInt) -> BigInt {
    a.neg()
}

#[majit_macros::elidable]
fn bigint_clone(a: &BigInt) -> BigInt {
    a.clone()
}

#[majit_macros::elidable]
fn bigint_invert(a: &BigInt) -> BigInt {
    a.invert()
}

#[majit_macros::elidable]
pub(crate) fn bigint_eq(a: BigInt, b: BigInt) -> bool {
    a == b
}

#[majit_macros::elidable]
fn bigint_lt(a: BigInt, b: BigInt) -> bool {
    a < b
}

#[majit_macros::elidable]
fn bigint_gt(a: BigInt, b: BigInt) -> bool {
    a > b
}

#[majit_macros::elidable]
fn bigint_mod(a: BigInt, b: BigInt) -> BigInt {
    a % b
}

/// longobject.py:62-70 `_truediv` delegates directly to rbigint.truediv and
/// only translates its application-level exceptions.
#[majit_macros::elidable]
fn bigint_truediv(a: &BigInt, b: &BigInt) -> Result<f64, PyError> {
    a.truediv(b).map_err(|error| match error {
        majit_rlib::rbigint::RBigIntError::DivisionByZero => {
            PyError::zero_division(ZERO_DIVISION_MSG)
        }
        majit_rlib::rbigint::RBigIntError::FloatDivisionOverflow => {
            PyError::overflow_error("integer division result too large for a float")
        }
        majit_rlib::rbigint::RBigIntError::Memory => PyError::memory_error(""),
        _ => unreachable!("rbigint.truediv has no other exception edge"),
    })
}

// ── Arithmetic operations ─────────────────────────────────────────────

/// Integer addition fast path.
///
/// The JIT will specialize this via:
///   GuardClass(a, &INT_TYPE)
///   GuardClass(b, &INT_TYPE)
///   GetfieldGcI(a, intval_offset) → va
///   GetfieldGcI(b, intval_offset) → vb
///   IntAdd(va, vb) → result
///   New(W_IntObject) + SetfieldGcI(result)

unsafe fn int_add(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let va = int_value(a);
    let vb = int_value(b);
    match va.checked_add(vb) {
        Some(r) => Ok(w_int_new(r)),
        None => Ok(w_long_new(bigint_add(BigInt::from(va), BigInt::from(vb)))),
    }
}

unsafe fn int_sub(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let va = int_value(a);
    let vb = int_value(b);
    match va.checked_sub(vb) {
        Some(r) => Ok(w_int_new(r)),
        None => Ok(w_long_new(bigint_sub(BigInt::from(va), BigInt::from(vb)))),
    }
}

unsafe fn int_mul(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let va = int_value(a);
    let vb = int_value(b);
    match va.checked_mul(vb) {
        Some(r) => Ok(w_int_new(r)),
        None => Ok(w_long_new(bigint_mul(BigInt::from(va), BigInt::from(vb)))),
    }
}

unsafe fn int_floordiv(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let va = int_value(a);
    let vb = int_value(b);
    if vb == 0 {
        return Err(PyError::zero_division(ZERO_DIVISION_MSG));
    }
    // intobject.py `_floordiv`: `ovfcheck(x // y)` has exactly one
    // non-zero-divisor overflow on a signed machine word.
    if va == -9_223_372_036_854_775_808_i64 && vb == -1 {
        let va = BigInt::from(va);
        let vb = BigInt::from(vb);
        return Ok(bigint_result(bigint_floordiv_nonzero(&va, &vb)));
    }
    let q = va / vb;
    let r = va % vb;
    // Adjust: if remainder is nonzero and signs of operands differ, subtract 1.
    let q = if r != 0 && (r ^ vb) < 0 { q - 1 } else { q };
    Ok(w_int_new(q))
}

unsafe fn int_mod(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let va = int_value(a);
    let vb = int_value(b);
    if vb == 0 {
        return Err(PyError::zero_division(ZERO_DIVISION_MSG));
    }
    // intobject.py `_mod`: the matching machine-word overflow is
    // `MIN % -1`; bounce that one case to rbigint like `ovfcheck`.
    if va == -9_223_372_036_854_775_808_i64 && vb == -1 {
        let va = BigInt::from(va);
        let vb = BigInt::from(vb);
        return Ok(bigint_result(bigint_modulo_nonzero(&va, &vb)));
    }
    let r = va % vb;
    let r = if r != 0 && (r ^ vb) < 0 { r + vb } else { r };
    Ok(w_int_new(r))
}

// ── Long (BigInt) arithmetic operations ─────────────────────────────

unsafe fn long_add(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    // longobject.py:_make_generic_descr_binop('add'): preserve the dedicated
    // rbigint.int_add path in both commutative operand orders. PyPy's
    // W_BoolObject subclasses W_IntObject and shares `intval`; pyre stores
    // bool separately, so the one upstream W_IntObject arm has two storage
    // projections here.
    if is_long(a) && is_bool(b) {
        if !w_long_get_value(a).is_zero() && !w_bool_get_value(b) {
            return Ok(pyre_object::longobject::w_long_from_raw(
                w_long_get_raw_value(a),
            ));
        }
        return Ok(w_long_new(
            w_long_get_value(a).int_add(w_bool_get_value(b) as i64),
        ));
    }
    if is_long(a) && is_int(b) {
        if !w_long_get_value(a).is_zero() && w_int_get_value(b) == 0 {
            return Ok(pyre_object::longobject::w_long_from_raw(
                w_long_get_raw_value(a),
            ));
        }
        return Ok(w_long_new(w_long_get_value(a).int_add(w_int_get_value(b))));
    }
    if is_bool(a) && is_long(b) {
        if !w_long_get_value(b).is_zero() && !w_bool_get_value(a) {
            return Ok(pyre_object::longobject::w_long_from_raw(
                w_long_get_raw_value(b),
            ));
        }
        return Ok(w_long_new(
            w_long_get_value(b).int_add(w_bool_get_value(a) as i64),
        ));
    }
    if is_int(a) && is_long(b) {
        if !w_long_get_value(b).is_zero() && w_int_get_value(a) == 0 {
            return Ok(pyre_object::longobject::w_long_from_raw(
                w_long_get_raw_value(b),
            ));
        }
        return Ok(w_long_new(w_long_get_value(b).int_add(w_int_get_value(a))));
    }
    debug_assert!(is_long(a) && is_long(b));
    if w_long_get_value(a).is_zero() {
        return Ok(pyre_object::longobject::w_long_from_raw(
            w_long_get_raw_value(b),
        ));
    }
    if w_long_get_value(b).is_zero() {
        return Ok(pyre_object::longobject::w_long_from_raw(
            w_long_get_raw_value(a),
        ));
    }
    Ok(w_long_new(w_long_get_value(a).add(w_long_get_value(b))))
}

unsafe fn long_sub(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    // longobject.py:descr_sub specializes only `long - int`; descr_rsub keeps
    // `int - long` on the ordinary two-rbigint subtraction path.
    if is_long(a) && is_bool(b) {
        if !w_bool_get_value(b) {
            return Ok(pyre_object::longobject::w_long_from_raw(
                w_long_get_raw_value(a),
            ));
        }
        return Ok(w_long_new(
            w_long_get_value(a).int_sub(w_bool_get_value(b) as i64),
        ));
    }
    if is_long(a) && is_int(b) {
        if w_int_get_value(b) == 0 {
            return Ok(pyre_object::longobject::w_long_from_raw(
                w_long_get_raw_value(a),
            ));
        }
        return Ok(w_long_new(w_long_get_value(a).int_sub(w_int_get_value(b))));
    }
    if is_long(a) {
        debug_assert!(is_long(b));
        if w_long_get_value(b).is_zero() {
            return Ok(pyre_object::longobject::w_long_from_raw(
                w_long_get_raw_value(a),
            ));
        }
        return Ok(w_long_new(w_long_get_value(a).sub(w_long_get_value(b))));
    }
    // Reflected int/bool - long follows descr_rsub's ordinary rbigint path:
    // only the machine-word left operand needs materialising.
    debug_assert!(is_long(b));
    Ok(w_long_new(
        BigInt::from(int_value(a)).sub(w_long_get_value(b)),
    ))
}

unsafe fn long_mul(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    // longobject.py:_make_generic_descr_binop('mul'): commutative int_mul.
    if is_long(a) && is_bool(b) {
        if !w_long_get_value(a).is_zero() && w_bool_get_value(b) {
            return Ok(pyre_object::longobject::w_long_from_raw(
                w_long_get_raw_value(a),
            ));
        }
        return Ok(w_long_new(
            w_long_get_value(a).int_mul(w_bool_get_value(b) as i64),
        ));
    }
    if is_long(a) && is_int(b) {
        if !w_long_get_value(a).is_zero() && w_int_get_value(b) == 1 {
            return Ok(pyre_object::longobject::w_long_from_raw(
                w_long_get_raw_value(a),
            ));
        }
        return Ok(w_long_new(w_long_get_value(a).int_mul(w_int_get_value(b))));
    }
    if is_bool(a) && is_long(b) {
        if !w_long_get_value(b).is_zero() && w_bool_get_value(a) {
            return Ok(pyre_object::longobject::w_long_from_raw(
                w_long_get_raw_value(b),
            ));
        }
        return Ok(w_long_new(
            w_long_get_value(b).int_mul(w_bool_get_value(a) as i64),
        ));
    }
    if is_int(a) && is_long(b) {
        if !w_long_get_value(b).is_zero() && w_int_get_value(a) == 1 {
            return Ok(pyre_object::longobject::w_long_from_raw(
                w_long_get_raw_value(b),
            ));
        }
        return Ok(w_long_new(w_long_get_value(b).int_mul(w_int_get_value(a))));
    }
    debug_assert!(is_long(a) && is_long(b));
    Ok(w_long_new(w_long_get_value(a).mul(w_long_get_value(b))))
}

unsafe fn long_floordiv(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    // longobject.py:424 `_make_descr_binop(_floordiv, _int_floordiv)`: a
    // machine-int divisor takes the dedicated `rbigint.int_floordiv` leg.
    // PyPy's `_floordiv` still carries the 2.x "long ..." wording
    // (longobject.py:409), which a 3.x runtime does not.
    if is_int_like(b) {
        let vb = int_value(b);
        if vb == 0 {
            return Err(PyError::zero_division(ZERO_DIVISION_MSG));
        }
        debug_assert!(is_long(a));
        if w_long_get_value(a).get_sign() == 1 && vb == 1 {
            return Ok(pyre_object::longobject::w_long_from_raw(
                w_long_get_raw_value(a),
            ));
        }
        return Ok(w_long_new(bigint_int_floordiv_nonzero(
            w_long_get_value(a),
            vb,
        )));
    }
    debug_assert!(is_long(b));
    let vb = w_long_get_value(b);
    if !vb.tobool() {
        return Err(PyError::zero_division(ZERO_DIVISION_MSG));
    }
    let owned_a;
    let va = if is_long(a) {
        w_long_get_value(a)
    } else {
        owned_a = BigInt::from(int_value(a));
        &owned_a
    };
    // rbigint.floordiv → _divmod, returning the quotient half (rbigint.py:1001).
    // `_floordiv`/`_int_floordiv` both `newlong` the quotient, keeping a long.
    Ok(w_long_new(bigint_floordiv_nonzero(va, vb)))
}

unsafe fn long_mod(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    // longobject.py:441 `_make_descr_binop(_mod, _int_mod)`. `_int_mod`
    // (machine-int RHS) computes through `rbigint.int_mod_int_result` and
    // returns `space.newint` — the remainder of a long by a machine int always
    // fits — while `_mod` (long RHS) returns `newlong`.
    if is_int_like(b) {
        let vb = int_value(b);
        if vb == 0 {
            return Err(PyError::zero_division(ZERO_DIVISION_MSG));
        }
        debug_assert!(is_long(a));
        return Ok(w_int_new(bigint_int_modulo_int_result_nonzero(
            w_long_get_value(a),
            vb,
        )));
    }
    debug_assert!(is_long(b));
    let vb = w_long_get_value(b);
    if !vb.tobool() {
        return Err(PyError::zero_division(ZERO_DIVISION_MSG));
    }
    let owned_a;
    let va = if is_long(a) {
        w_long_get_value(a)
    } else {
        owned_a = BigInt::from(int_value(a));
        &owned_a
    };
    if is_long(a)
        && jit_bigint_divrem_returns_lhs_remainder(
            va as *const BigInt as i64,
            vb as *const BigInt as i64,
        ) != 0
    {
        return Ok(pyre_object::longobject::w_long_from_raw(
            w_long_get_raw_value(a),
        ));
    }
    // rbigint.mod → _divmod, returning the remainder half (rbigint.py:1001).
    Ok(w_long_new(bigint_modulo_nonzero(va, vb)))
}

/// PyPy longobject.py `_divmod` / `_int_divmod`: compute both halves with one
/// rbigint division and only then box the pair.  Calling `floordiv` and `mod`
/// separately would run `_divmod` twice.
unsafe fn integer_divmod_pair(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    if is_int_like(a) && is_int_like(b) {
        let va = int_value(a);
        let vb = int_value(b);
        debug_assert_ne!(vb, 0, "numeric_divmod checks the divisor");
        if va == i64::MIN && vb == -1 {
            let (q, r) = BigInt::from(va)
                .int_divmod(vb)
                .expect("divisor was checked nonzero");
            let q = RBigIntGcRoot::new(q);
            let r = RBigIntGcRoot::new(r);
            return Ok(w_tuple_new(vec![
                bigint_result(q.translated_alias()),
                bigint_result(r.translated_alias()),
            ]));
        }
        let mut q = va / vb;
        let mut r = va % vb;
        if r != 0 && (r ^ vb) < 0 {
            q -= 1;
            r += vb;
        }
        return Ok(w_tuple_new(vec![w_int_new(q), w_int_new(r)]));
    }

    // longobject.py `_make_descr_binop(_divmod, _int_divmod)` preserves a
    // dedicated long/int residual; reflected int/long follows `_divmod`.
    let remainder_aliases_a = if is_long(a) && is_long(b) {
        let va = w_long_get_value(a);
        let vb = w_long_get_value(b);
        jit_bigint_divrem_returns_lhs_remainder(
            va as *const BigInt as i64,
            vb as *const BigInt as i64,
        ) != 0
    } else {
        false
    };
    let (q, r) = if is_long(a) && is_int_like(b) {
        w_long_get_value(a)
            .int_divmod(int_value(b))
            .expect("divisor was checked nonzero")
    } else {
        debug_assert!(is_long(b));
        let owned_a;
        let va = if is_long(a) {
            w_long_get_value(a)
        } else {
            owned_a = BigInt::from(int_value(a));
            &owned_a
        };
        va.divmod(w_long_get_value(b))
            .expect("divisor was checked nonzero")
    };
    // `_divmod` produces two live rbigints before either wrapper is
    // allocated. RPython's GC transform roots both across those collecting
    // allocations.
    let q = RBigIntGcRoot::new(q);
    let r = RBigIntGcRoot::new(r);
    if is_long(a) || is_long(b) {
        let w_q = w_long_new(q.translated_alias());
        let w_r = if remainder_aliases_a {
            pyre_object::longobject::w_long_from_raw(w_long_get_raw_value(a))
        } else {
            w_long_new(r.translated_alias())
        };
        Ok(w_tuple_new(vec![w_q, w_r]))
    } else {
        Ok(w_tuple_new(vec![
            bigint_result(q.translated_alias()),
            bigint_result(r.translated_alias()),
        ]))
    }
}

/// `rbigint.floordiv` payload half (`longobject.py:409 _floordiv` →
/// `rbigint.floordiv` → `divmod`, `rbigint.py:1001 @jit.elidable`). Elidable
/// but CAN raise ZeroDivisionError on a zero divisor → `EF_ELIDABLE_CAN_RAISE`:
/// the trace records `CALL_PURE` + `GUARD_NO_EXCEPTION`. Returns a bare
/// `*mut BigInt` (Int) on success; on a zero divisor publishes the exception
/// and returns 0 so the trailing `GUARD_NO_EXCEPTION` deopts.
#[majit_macros::elidable]
pub extern "C" fn jit_w_long_floordiv_raw(a: i64, b: i64) -> i64 {
    let (a, b) = unsafe {
        (
            w_long_get_value(a as PyObjectRef),
            w_long_get_value(b as PyObjectRef),
        )
    };
    bigint_floordiv_core(a, b, false)
}

/// `rbigint.mod` over `W_LongObject` operands (no-collect, record-time). Same
/// `EF_ELIDABLE_CAN_RAISE` contract as [`jit_w_long_floordiv_raw`].
#[majit_macros::elidable]
pub extern "C" fn jit_w_long_mod_raw(a: i64, b: i64) -> i64 {
    let (a, b) = unsafe {
        (
            w_long_get_value(a as PyObjectRef),
            w_long_get_value(b as PyObjectRef),
        )
    };
    bigint_mod_core(a, b, false)
}

/// `rbigint.lshift` over `W_LongObject` operands (no-collect, record-time).
/// Elidable but CAN raise ValueError (negative shift) / OverflowError (shift too
/// large and base nonzero) → `EF_ELIDABLE_CAN_RAISE`. Walker-only (the trait
/// path defers shift to the generic residual).
#[majit_macros::elidable]
pub extern "C" fn jit_w_long_lshift_raw(a: i64, b: i64) -> i64 {
    let (a, b) = unsafe {
        (
            w_long_get_value(a as PyObjectRef),
            w_long_get_value(b as PyObjectRef),
        )
    };
    bigint_lshift_core(a, b, false)
}

/// `rbigint.rshift` over `W_LongObject` operands (no-collect, record-time). Like
/// [`jit_w_long_lshift_raw`] but a shift too large yields 0 / -1 (all bits
/// shifted out) instead of OverflowError; only a negative shift raises.
#[majit_macros::elidable]
pub extern "C" fn jit_w_long_rshift_raw(a: i64, b: i64) -> i64 {
    let (a, b) = unsafe {
        (
            w_long_get_value(a as PyObjectRef),
            w_long_get_value(b as PyObjectRef),
        )
    };
    bigint_rshift_core(a, b, false)
}

/// Allocate a result bigint, choosing the COLLECTING nursery (runtime payload
/// helpers, invoked from a gcmap-rooted residual call) or the NO-COLLECT nursery
/// (record-time wrappers, which hold the operands natively so a collection would
/// move them). See `longobject::alloc_bigint_nursery_collecting`.
#[inline]
fn alloc_result_bigint(value: BigInt, collecting: bool) -> i64 {
    if collecting {
        pyre_object::longobject::alloc_bigint_nursery_collecting(value) as i64
    } else {
        pyre_object::longobject::alloc_bigint_nursery(value) as i64
    }
}

/// `rbigint.floordiv`/`mod`/`lshift`/`rshift` cores over `&BigInt` operands,
/// shared by the collecting runtime payload helpers (`jit_bigint_*`) and the
/// no-collect record-time wrappers (`jit_w_long_*_raw`). Each returns a bare
/// `*mut BigInt` (Int, as i64) on success; a zero divisor / out-of-range shift
/// publishes the exception and returns 0 so the trailing `GUARD_NO_EXCEPTION`
/// deopts.
fn bigint_floordiv_core(a: &BigInt, b: &BigInt, collecting: bool) -> i64 {
    if pyre_object::longobject::jit_bigint_sign_i64(b) == 0 {
        crate::runtime_ops::jit_publish_exception(
            PyError::zero_division(ZERO_DIVISION_MSG).to_exc_object(),
        );
        return 0;
    }
    // rbigint.floordiv → _divmod, returning the quotient half (rbigint.py:1001).
    alloc_result_bigint(
        a.divmod(b).expect("divisor was checked nonzero").0,
        collecting,
    )
}

fn bigint_mod_core(a: &BigInt, b: &BigInt, collecting: bool) -> i64 {
    if pyre_object::longobject::jit_bigint_sign_i64(b) == 0 {
        crate::runtime_ops::jit_publish_exception(
            PyError::zero_division(ZERO_DIVISION_MSG).to_exc_object(),
        );
        return 0;
    }
    let selfsign = pyre_object::longobject::jit_bigint_sign_i64(a);
    let othersign = pyre_object::longobject::jit_bigint_sign_i64(b);
    if selfsign != 0
        && selfsign == othersign
        && b.numdigits() != 1
        && divrem_returns_input_as_remainder(a, b)
    {
        // `_divmod_small` leaves `_divrem`'s literal `a` remainder unchanged.
        return a as *const BigInt as i64;
    }
    // rbigint.mod → divmod, returning the remainder half.
    alloc_result_bigint(
        a.divmod(b).expect("divisor was checked nonzero").1,
        collecting,
    )
}

fn bigint_lshift_core(a: &BigInt, b: &BigInt, collecting: bool) -> i64 {
    if pyre_object::longobject::jit_bigint_sign_i64(b) < 0 {
        crate::runtime_ops::jit_publish_exception(
            PyError::value_error("negative shift count").to_exc_object(),
        );
        return 0;
    }
    // `rbigint.toint()` is a *signed* machine int (i64), so a count above
    // i64::MAX overflows here — not at usize::MAX — matching `_lshift`.
    let shift = if jit_bigint_to_i64_fits(b) != 0 {
        jit_bigint_to_i64_value(b)
    } else {
        if pyre_object::longobject::jit_bigint_sign_i64(a) == 0 {
            return a as *const BigInt as i64;
        }
        crate::runtime_ops::jit_publish_exception(
            PyError::overflow_error("shift count too large").to_exc_object(),
        );
        return 0;
    };
    if shift == 0 || pyre_object::longobject::jit_bigint_sign_i64(a) == 0 {
        return a as *const BigInt as i64;
    }
    match bigint_lshift(a, shift) {
        Ok(value) => alloc_result_bigint(value, collecting),
        Err(majit_rlib::rbigint::RBigIntError::Memory) => {
            crate::runtime_ops::jit_publish_exception(PyError::memory_error("").to_exc_object());
            0
        }
        Err(_) => unreachable!("shift sign/range was validated above"),
    }
}

fn bigint_rshift_core(a: &BigInt, b: &BigInt, collecting: bool) -> i64 {
    if pyre_object::longobject::jit_bigint_sign_i64(b) < 0 {
        crate::runtime_ops::jit_publish_exception(
            PyError::value_error("negative shift count").to_exc_object(),
        );
        return 0;
    }
    // `toint()` overflow (count > i64::MAX) takes this branch like `_rshift`;
    // for rshift the result (0 / -1) is the same as an actual huge shift.
    let shift = if jit_bigint_to_i64_fits(b) != 0 {
        jit_bigint_to_i64_value(b)
    } else {
        let val = if pyre_object::longobject::jit_bigint_sign_i64(a) < 0 {
            -1
        } else {
            0
        };
        return alloc_result_bigint(BigInt::from(val), collecting);
    };
    if shift == 0 {
        // rbigint.rshift(..., 0) returns `self`.
        return a as *const BigInt as i64;
    }
    alloc_result_bigint(bigint_rshift(a, shift), collecting)
}

/// `rbigint.floordiv`/`mod`/`lshift`/`rshift` payload halves on bare
/// `*const BigInt` operands — the divmod/shift the walker emits after reading
/// each `W_LongObject` operand's immutable `value` via `GetfieldGc`. Taking
/// the payloads (not the wrappers) keeps these elidable calls pure on the
/// immutable bigints so the optimizer never reorders them ahead of the boxing
/// `setfield_gc`. Allocates the result via the COLLECTING nursery (the call is a
/// gcmap-rooted residual `CallR` holding no unrooted pointer across the alloc).
/// `EF_ELIDABLE_CAN_RAISE`.
///
/// # Safety note: `extern "C"` over `i64`-encoded `*const BigInt` operands, live
/// for the duration of the call.
#[majit_macros::elidable]
pub extern "C" fn jit_bigint_floordiv(a: i64, b: i64) -> i64 {
    let (a, b) = unsafe { (&*(a as *const BigInt), &*(b as *const BigInt)) };
    bigint_floordiv_core(a, b, true)
}

/// `rbigint.mod` on bare payloads (collecting). See [`jit_bigint_floordiv`].
#[majit_macros::elidable]
pub extern "C" fn jit_bigint_mod(a: i64, b: i64) -> i64 {
    let (a, b) = unsafe { (&*(a as *const BigInt), &*(b as *const BigInt)) };
    bigint_mod_core(a, b, true)
}

/// `rbigint.lshift` on bare payloads (collecting). See [`jit_bigint_floordiv`].
#[majit_macros::elidable]
pub extern "C" fn jit_bigint_lshift(a: i64, b: i64) -> i64 {
    let (a, b) = unsafe { (&*(a as *const BigInt), &*(b as *const BigInt)) };
    bigint_lshift_core(a, b, true)
}

/// `rbigint.rshift` on bare payloads (collecting). See [`jit_bigint_floordiv`].
#[majit_macros::elidable]
pub extern "C" fn jit_bigint_rshift(a: i64, b: i64) -> i64 {
    let (a, b) = unsafe { (&*(a as *const BigInt), &*(b as *const BigInt)) };
    bigint_rshift_core(a, b, true)
}

/// `rbigint.truediv` payload half (`longobject.py:62-70 _truediv` →
/// `rbigint.truediv`, `rbigint.py:890`). Elidable but CAN raise
/// ZeroDivisionError / OverflowError → `EF_ELIDABLE_CAN_RAISE`: `CALL_PURE_F` +
/// `GUARD_NO_EXCEPTION`. Returns the correctly-rounded quotient as an `f64`
/// directly (a `CallPureF`, the float analogue of `rbigint.truediv` returning a
/// float); the walker then boxes it with `wrapfloat` (transparent
/// `new_with_vtable` + `setfield_gc_f`, mirroring `space.newfloat(f)`). On a
/// raising input publishes the exception and returns garbage (the guard
/// deopts). Walker-only, like the shift helpers.
#[majit_macros::elidable]
pub extern "C" fn jit_w_long_truediv_raw(a: i64, b: i64) -> f64 {
    let a = a as PyObjectRef;
    let b = b as PyObjectRef;
    unsafe {
        match bigint_truediv(w_long_get_value(a), w_long_get_value(b)) {
            Ok(f) => f,
            Err(mut e) => {
                crate::runtime_ops::jit_publish_exception(e.to_exc_object());
                0.0
            }
        }
    }
}

// ── Float arithmetic operations ──────────────────────────────────────

/// Coerce an operand to f64. Works for int, long, and float objects.
unsafe fn as_float(obj: PyObjectRef) -> f64 {
    if is_float(obj) {
        w_float_get_value(obj)
    } else if is_int(obj) {
        w_int_get_value(obj) as f64
    } else {
        // long → f64 (may lose precision for very large values)
        jit_bigint_to_f64_or_inf(w_long_get_value(obj))
    }
}

/// Reject an over-range `int` operand of a value-producing float operator:
/// `PyFloat_AsDouble` raises `OverflowError` when an `int`'s magnitude
/// exceeds f64 range. `v` is the already-extracted [`as_float`] value; a
/// genuine float infinity is preserved, only an over-range `int` raises.
///
/// Split from the f64 extraction so the operator bodies pass plain `f64`
/// values — never a `Result<f64, _>` — into the arithmetic. A `Result`
/// payload mixing `Float` (`Ok`) and `Ref` (`Err`) has no single register
/// kind, so the JIT codewriter cannot flatten it (`emit_list_of_kind`).
unsafe fn reject_float_coercion_overflow(obj: PyObjectRef, v: f64) -> Result<(), PyError> {
    if is_long(obj) && !v.is_finite() {
        return Err(PyError::overflow_error("int too large to convert to float"));
    }
    Ok(())
}

/// A `float`/`complex` power coerces each `int` operand to a double before the
/// power is computed, so an over-range `int` (base or exponent) raises
/// OverflowError up front — even for `1.0 ** huge`, which never reaches the
/// arithmetic. Only an over-range `int` raises; a genuine float infinity does
/// not.
pub(crate) unsafe fn reject_pow_operand_overflow(obj: PyObjectRef) -> Result<(), PyError> {
    if is_long(obj) && !jit_bigint_to_f64_or_inf(w_long_get_value(obj)).is_finite() {
        return Err(PyError::overflow_error("int too large to convert to float"));
    }
    Ok(())
}

/// True if both operands are numeric and at least one is float.

unsafe fn is_float_pair(a: PyObjectRef, b: PyObjectRef) -> bool {
    let a_num = is_int(a) || is_float(a) || is_long(a);
    let b_num = is_int(b) || is_float(b) || is_long(b);
    a_num && b_num && (is_float(a) || is_float(b))
}

unsafe fn float_add(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let va = as_float(a);
    reject_float_coercion_overflow(a, va)?;
    let vb = as_float(b);
    reject_float_coercion_overflow(b, vb)?;
    Ok(w_float_new(va + vb))
}

unsafe fn float_sub(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let va = as_float(a);
    reject_float_coercion_overflow(a, va)?;
    let vb = as_float(b);
    reject_float_coercion_overflow(b, vb)?;
    Ok(w_float_new(va - vb))
}

unsafe fn float_mul(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let va = as_float(a);
    reject_float_coercion_overflow(a, va)?;
    let vb = as_float(b);
    reject_float_coercion_overflow(b, vb)?;
    Ok(w_float_new(va * vb))
}

unsafe fn float_truediv(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let vb = as_float(b);
    reject_float_coercion_overflow(b, vb)?;
    if vb == 0.0 {
        return Err(PyError::zero_division(ZERO_DIVISION_MSG));
    }
    let va = as_float(a);
    reject_float_coercion_overflow(a, va)?;
    Ok(w_float_new(va / vb))
}

/// floatobject.py:508-512: descr_floordiv → _divmod_w()[0].
unsafe fn float_floordiv(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let x = as_float(a);
    reject_float_coercion_overflow(a, x)?;
    let y = as_float(b);
    reject_float_coercion_overflow(b, y)?;
    let (floordiv, _mod) = float_divmod_w(x, y)?;
    Ok(w_float_new(floordiv))
}

/// floatobject.py:520-540: descr_mod with math_fmod + sign correction.
unsafe fn float_mod(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let x = as_float(a);
    reject_float_coercion_overflow(a, x)?;
    let y = as_float(b);
    reject_float_coercion_overflow(b, y)?;
    if y == 0.0 {
        // floatobject.py:526
        return Err(PyError::zero_division(ZERO_DIVISION_MSG));
    }
    let mut m = jit_float_fmod(x, y);
    if m != 0.0 {
        // floatobject.py:529-531: ensure remainder has same sign as denominator
        if (y < 0.0) != (m < 0.0) {
            m += y;
        }
    } else {
        // floatobject.py:536-538: signed zero — copysign(0.0, y)
        m = float_copysign(0.0, y);
    }
    Ok(w_float_new(m))
}

/// floatobject.py:758-793: _divmod_w.
fn float_divmod_w(x: f64, y: f64) -> Result<(f64, f64), PyError> {
    if y == 0.0 {
        // floatobject.py:761
        return Err(PyError::zero_division(ZERO_DIVISION_MSG));
    }
    let mut m = jit_float_fmod(x, y);
    // floatobject.py:767: div = (x - mod) / y
    let mut div = (x - m) / y;
    if m != 0.0 {
        // floatobject.py:769-771: sign correction
        if (y < 0.0) != (m < 0.0) {
            m += y;
            div -= 1.0;
        }
    } else {
        // floatobject.py:776-778: signed zero
        // "mod *= mod" hides "+0" from optimizer, then negate if y < 0
        m = m * m; // hide from optimizer
        if y < 0.0 {
            m = -m;
        }
    }
    // floatobject.py:784-790: snap quotient to nearest integral value
    let floordiv = if div != 0.0 {
        let f = div.floor();
        if div - f > 0.5 { f + 1.0 } else { f }
    } else {
        // floatobject.py:789-790: zero with sign of true quotient
        let d = div * div; // hide from optimizer
        d * x / y
    };
    Ok((floordiv, m))
}

// ── Power ────────────────────────────────────────────────────────────

unsafe fn int_pow(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let va = int_value(a);
    let vb = int_value(b);
    if vb < 0 {
        // intobject.py:415-419 _pow_nomod raises ValueError for iw < 0,
        // descr_pow catches it and routes through float pow — which
        // carries the ZeroDivisionError guard from floatobject.py:910-913.
        return Ok(w_float_new(float_pow_raw(va as f64, vb as f64)?));
    }
    // intobject.py:415 / longobject.py:229: x ** 0 == 1 for any x.
    if vb == 0 {
        return Ok(w_int_new(1));
    }
    // longobject.py:224-231: rbigint.pow handles arbitrary exponents.
    match va {
        0 => return Ok(w_int_new(0)),
        1 => return Ok(w_int_new(1)),
        -1 => return Ok(w_int_new(if vb % 2 == 0 { 1 } else { -1 })),
        _ => {}
    }
    // intobject.py:414-435 `_pow_nomod`: exponentiation by squaring with an
    // overflow check at each machine multiplication. Keep this literal loop;
    // `checked_mul` is the Rust source spelling the MIR front lowers back to
    // RPython's `int_mul_ovf` exception edge.
    let mut temp = va;
    let mut ix = 1_i64;
    let mut iw = vb;
    let machine_result = loop {
        if iw & 1 != 0 {
            let Some(value) = ix.checked_mul(temp) else {
                break None;
            };
            ix = value;
        }
        iw >>= 1;
        if iw == 0 {
            break Some(ix);
        }
        let Some(value) = temp.checked_mul(temp) else {
            break None;
        };
        temp = value;
    };
    match machine_result {
        Some(r) => Ok(w_int_new(r)),
        None => {
            Ok(w_long_new(BigInt::from(va).int_pow(vb, None).map_err(
                |_| PyError::memory_error("exponent too large"),
            )?))
        }
    }
}

unsafe fn long_pow(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let vb_owned;
    let vb = if is_long(b) {
        w_long_get_value(b)
    } else {
        vb_owned = BigInt::from(int_value(b));
        &vb_owned
    };
    if vb.get_sign() < 0 {
        // longobject.py:219-222 calls descr_float on both integer operands
        // before float pow.  RBigInt::tofloat raises on an out-of-range value;
        // do not silently pass the infinity sentinel from as_float onward.
        reject_pow_operand_overflow(a)?;
        reject_pow_operand_overflow(b)?;
        let fa = as_float(a);
        let fb = as_float(b);
        return Ok(w_float_new(float_pow_raw(fa, fb)?));
    }
    // longobject.py:229: `if not exp_bigint: return int_pow(0)` → 1. `descr_pow`
    // wraps every branch as `W_LongObject`, so a long base keeps the long
    // representation across these trivial-base short-circuits too.
    if vb.get_sign() == 0 {
        return Ok(w_long_new(BigInt::from(1)));
    }
    // longobject.py:224-231: rbigint.pow handles arbitrary exponents.
    let va_owned;
    let va = if is_long(a) {
        w_long_get_value(a)
    } else {
        va_owned = BigInt::from(int_value(a));
        &va_owned
    };
    // Both rbigint.int_pow(1) and rbigint.pow(ONERBIGINT) return the base
    // reference after the zero-base check. W_LongObject adds only a wrapper.
    if is_long(a) && va.get_sign() != 0 && vb.int_eq(1) {
        return Ok(pyre_object::longobject::w_long_from_raw(
            w_long_get_raw_value(a),
        ));
    }
    if va.get_sign() == 0 {
        return Ok(w_long_new(BigInt::from(0)));
    }
    if va.int_eq(1) {
        return Ok(w_long_new(BigInt::from(1)));
    }
    if va.int_eq(-1) {
        let even = vb.digit(0) & 1 == 0;
        return Ok(w_long_new(BigInt::from(if even { 1 } else { -1 })));
    }
    // longobject.py:229-231: `descr_pow` keeps a `W_IntObject` exponent
    // unwrapped (`exp_bigint` stays None) and calls `rbigint.int_pow`; only a
    // long exponent reaches `rbigint.pow`.
    if is_int_like(b) {
        return Ok(w_long_new(bigint_int_pow_nomod(va, int_value(b))?));
    }
    Ok(w_long_new(bigint_pow_nomod(va, vb)?))
}

// ── Shift operations ─────────────────────────────────────────────────

unsafe fn int_lshift(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let va = int_value(a);
    let vb = int_value(b);
    if vb < 0 {
        return Err(PyError::value_error("negative shift count"));
    }
    // intobject.py:374-383 `_lshift`: use the machine-int result while the
    // shift is in range and `ovfcheck(a << b)` succeeds. Rust's
    // `checked_shl` checks only the count, so verify the arithmetic result by
    // shifting it back. The overflow recovery is the exact
    // `rbigint.lshift_int_int_bigint_result` helper used at
    // intobject.py:861-868.
    // RPython's target constant `LONG_BIT` is 64 for pyre's i64 word.
    if vb < 64 {
        let shifted = va << vb;
        if (shifted >> vb) == va {
            return Ok(w_int_new(shifted));
        }
    }
    if va == 0 {
        return Ok(w_int_new(0));
    }
    let big = bigint_lshift_int_int_result(va, vb)?;
    Ok(bigint_result(big))
}

unsafe fn int_rshift(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    // intobject.py:393-403 `_rshift(space, a, b)`:
    //   if r_uint(b) >= LONG_BIT:
    //       if b < 0: raise ValueError("negative shift count")
    //       # b >= LONG_BIT
    //       if a == 0: return wrapint(space, a)
    //       a = -1 if a < 0 else 0
    //   else: a = a >> b
    let va = int_value(a);
    let vb = int_value(b);
    if vb < 0 {
        return Err(PyError::value_error("negative shift count"));
    }
    if vb >= 64 {
        return Ok(w_int_new(if va < 0 { -1 } else { 0 }));
    }
    Ok(w_int_new(va >> vb))
}

unsafe fn long_lshift(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let vb_owned;
    let vb = if is_long(b) {
        w_long_get_value(b)
    } else {
        vb_owned = BigInt::from(int_value(b));
        &vb_owned
    };
    if vb.get_sign() < 0 {
        return Err(PyError::value_error("negative shift count"));
    }
    // longobject.py:375-380: `toint()` (signed machine int / i64) overflows
    // when the count exceeds i64::MAX → 0 if base is zero, OverflowError
    // otherwise.
    let shift = if jit_bigint_to_i64_fits(vb) != 0 {
        jit_bigint_to_i64_value(vb)
    } else {
        let base_is_zero = if is_long(a) {
            w_long_get_value(a).get_sign() == 0
        } else {
            int_value(a) == 0
        };
        if base_is_zero {
            // `_lshift` returns `self` (a W_LongObject) for a zero base.
            return Ok(if is_long(a) {
                a
            } else {
                // A reflected compact-int operand is first coerced to the
                // W_LongObject receiver in PyPy's `_make_descr_binop`.
                w_long_new(BigInt::from(0))
            });
        }
        return Err(PyError::overflow_error("shift count too large"));
    };
    // rbigint.lshift returns `self` for a zero count or zero receiver.
    // `W_LongObject._lshift` then puts that same payload into a fresh long
    // wrapper (except for the huge-count zero case above, which returns the
    // receiver wrapper itself).
    if is_long(a) && (shift == 0 || w_long_get_value(a).get_sign() == 0) {
        return Ok(pyre_object::longobject::w_long_from_raw(
            w_long_get_raw_value(a),
        ));
    }
    let va_owned;
    let va = if is_long(a) {
        w_long_get_value(a)
    } else {
        va_owned = BigInt::from(int_value(a));
        &va_owned
    };
    let shifted = bigint_lshift_count(va, shift)?;
    Ok(w_long_new(shifted))
}

unsafe fn long_rshift(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let vb_owned;
    let vb = if is_long(b) {
        w_long_get_value(b)
    } else {
        vb_owned = BigInt::from(int_value(b));
        &vb_owned
    };
    if vb.get_sign() < 0 {
        return Err(PyError::value_error("negative shift count"));
    }
    // longobject.py:393-397: `toint()` overflow (count > i64::MAX) → positive
    // yields 0, negative yields -1 (all bits shifted out).
    let shift = if jit_bigint_to_i64_fits(vb) != 0 {
        jit_bigint_to_i64_value(vb)
    } else {
        let negative = if is_long(a) {
            w_long_get_value(a).get_sign() < 0
        } else {
            int_value(a) < 0
        };
        return Ok(w_int_new(if negative { -1 } else { 0 }));
    };
    // rbigint.rshift(0) returns `self`; `newlong` adds only a fresh wrapper.
    if is_long(a) && shift == 0 {
        return Ok(pyre_object::longobject::w_long_from_raw(
            w_long_get_raw_value(a),
        ));
    }
    // `_rshift`/`_int_rshift` `newlong` the normal-count result, keeping a long.
    let va_owned;
    let va = if is_long(a) {
        w_long_get_value(a)
    } else {
        va_owned = BigInt::from(int_value(a));
        &va_owned
    };
    Ok(w_long_new(bigint_rshift(va, shift)))
}

// ── bool-as-int helpers ──────────────────────────────────────────────

/// True when obj is int or bool (bool is a subclass of int in Python).
#[inline]
pub(crate) unsafe fn is_int_like(obj: PyObjectRef) -> bool {
    is_int(obj) || is_bool(obj)
}

/// Extract i64 from an int or bool object.
#[inline]
pub(crate) unsafe fn int_value(obj: PyObjectRef) -> i64 {
    if is_bool(obj) {
        w_bool_get_value(obj) as i64
    } else {
        w_int_get_value(obj)
    }
}

// ── Bitwise operations ───────────────────────────────────────────────

// W_IntObject.descr_and/or/xor — always int; the bool result is produced
// by W_BoolObject's own descr_and/or/xor, not by the int path.
unsafe fn int_bitand(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_int_new(int_value(a) & int_value(b)))
}

unsafe fn int_bitor(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_int_new(int_value(a) | int_value(b)))
}

unsafe fn int_bitxor(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_int_new(int_value(a) ^ int_value(b)))
}

unsafe fn long_bitand(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    if is_long(a) && is_bool(b) {
        return Ok(w_long_new(
            w_long_get_value(a).int_and_(w_bool_get_value(b) as i64),
        ));
    }
    if is_long(a) && is_int(b) {
        return Ok(w_long_new(w_long_get_value(a).int_and_(w_int_get_value(b))));
    }
    if is_bool(a) && is_long(b) {
        return Ok(w_long_new(
            w_long_get_value(b).int_and_(w_bool_get_value(a) as i64),
        ));
    }
    if is_int(a) && is_long(b) {
        return Ok(w_long_new(w_long_get_value(b).int_and_(w_int_get_value(a))));
    }
    debug_assert!(is_long(a) && is_long(b));
    Ok(w_long_new(w_long_get_value(a).and_(w_long_get_value(b))))
}

unsafe fn long_bitor(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    if is_long(a) && is_bool(b) {
        return Ok(w_long_new(
            w_long_get_value(a).int_or_(w_bool_get_value(b) as i64),
        ));
    }
    if is_long(a) && is_int(b) {
        return Ok(w_long_new(w_long_get_value(a).int_or_(w_int_get_value(b))));
    }
    if is_bool(a) && is_long(b) {
        return Ok(w_long_new(
            w_long_get_value(b).int_or_(w_bool_get_value(a) as i64),
        ));
    }
    if is_int(a) && is_long(b) {
        return Ok(w_long_new(w_long_get_value(b).int_or_(w_int_get_value(a))));
    }
    debug_assert!(is_long(a) && is_long(b));
    Ok(w_long_new(w_long_get_value(a).or_(w_long_get_value(b))))
}

unsafe fn long_bitxor(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    if is_long(a) && is_bool(b) {
        return Ok(w_long_new(
            w_long_get_value(a).int_xor(w_bool_get_value(b) as i64),
        ));
    }
    if is_long(a) && is_int(b) {
        return Ok(w_long_new(w_long_get_value(a).int_xor(w_int_get_value(b))));
    }
    if is_bool(a) && is_long(b) {
        return Ok(w_long_new(
            w_long_get_value(b).int_xor(w_bool_get_value(a) as i64),
        ));
    }
    if is_int(a) && is_long(b) {
        return Ok(w_long_new(w_long_get_value(b).int_xor(w_int_get_value(a))));
    }
    debug_assert!(is_long(a) && is_long(b));
    Ok(w_long_new(w_long_get_value(a).xor(w_long_get_value(b))))
}

// ── String operations ────────────────────────────────────────────────

/// Concatenate two str objects.

pub(crate) unsafe fn str_concat(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    // `w_str_concat` joins the surrogate-aware WTF-8 views, so concatenating a
    // surrogateescape/surrogatepass-decoded string does not go through
    // `w_str_get_value` (which rejects lone surrogates).
    Ok(pyre_object::unicodeobject::w_str_concat(a, b))
}

/// Extract a non-negative repeat count from an int or long.
unsafe fn repeat_count(n: PyObjectRef) -> Result<usize, PyError> {
    if is_long(n) {
        let big = w_long_get_value(n);
        // The count coerces to a signed machine word (an index-sized integer):
        // a non-negative value exceeding `isize::MAX` overflows rather than
        // wrapping to a huge `usize` count, matching `getindex_w`. Negative
        // counts clamp to 0 only after they fit the index-sized word.
        match big.to_isize() {
            Some(v) => Ok(if v < 0 { 0 } else { v as usize }),
            None => Err(PyError::new(
                PyErrorKind::OverflowError,
                "cannot fit 'int' into an index-sized integer",
            )),
        }
    } else {
        let nv = w_int_get_value(n);
        Ok(if nv < 0 { 0 } else { nv as usize })
    }
}

/// tupleobject.py descr_mul
pub(crate) unsafe fn tuple_repeat(t: PyObjectRef, n: PyObjectRef) -> PyResult {
    let n = repeat_count(n)?;
    // tupleobject.py: `if times == 1 and space.type(self) == space.w_tuple:
    // return self`. Subclasses must still be copied to a base tuple.
    if n == 1 && is_exact_tuple(t) {
        return Ok(t);
    }
    let len = w_tuple_len(t);
    let cap = len
        .checked_mul(n)
        .ok_or_else(|| PyError::new(PyErrorKind::OverflowError, "tuple is too large"))?;
    let mut items: Vec<PyObjectRef> = Vec::new();
    items
        .try_reserve_exact(cap)
        .map_err(|_| PyError::new(PyErrorKind::MemoryError, ""))?;
    for _ in 0..n {
        for i in 0..len {
            if let Some(item) = w_tuple_getitem(t, i as i64) {
                items.push(item);
            }
        }
    }
    Ok(w_tuple_new(items))
}

/// The builtin sequences repeat through `sq_repeat`, never `nb_multiply`.
pub(crate) unsafe fn is_repeat_sequence(obj: PyObjectRef) -> bool {
    is_str(obj) || is_list(obj) || is_tuple(obj) || pyre_object::bytesobject::is_bytes_like(obj)
}

/// `sequence_repeat` for a receiver [`is_repeat_sequence`] accepted, with the
/// count already reduced through `__index__`.
unsafe fn sequence_repeat(seq: PyObjectRef, count: PyObjectRef) -> PyResult {
    if is_str(seq) {
        str_repeat(seq, count)
    } else if is_list(seq) {
        list_repeat(seq, count)
    } else if is_tuple(seq) {
        tuple_repeat(seq, count)
    } else {
        bytes_repeat(seq, count)
    }
}

/// unicodeobject.py:619-621 descr_mul
pub(crate) unsafe fn str_repeat(s: PyObjectRef, n: PyObjectRef) -> PyResult {
    // Repeat at the WTF-8 byte level — a repetition of valid WTF-8 is valid
    // WTF-8 — so a surrogate-bearing string repeats without going through
    // `w_str_get_value`.
    let bytes = w_str_get_wtf8(s).as_bytes();
    let count = repeat_count(n)?;
    if count == 1 {
        return Ok(crate::type_methods::str_result_unchanged(s));
    }
    // unicode_repeat: overflow is judged on the character count
    // (length * nchars), not the WTF-8 byte length. A non-ASCII character
    // is wider in WTF-8 than its Py_UCS storage, so the byte product may
    // exceed isize::MAX while the character product does not. That case is
    // a MemoryError from the allocation, not an OverflowError.
    let char_len = w_str_len(s);
    if char_len != 0 && count > isize::MAX as usize / char_len {
        return Err(PyError::new(
            PyErrorKind::OverflowError,
            "repeated string is too long",
        ));
    }
    let Some(total) = bytes.len().checked_mul(count) else {
        return Err(PyError::new(PyErrorKind::MemoryError, ""));
    };
    let mut out: Vec<u8> = Vec::new();
    out.try_reserve_exact(total)
        .map_err(|_| PyError::new(PyErrorKind::MemoryError, ""))?;
    for _ in 0..count {
        out.extend_from_slice(bytes);
    }
    let buf = Wtf8Buf::from_bytes(out).expect("repetition of WTF-8 is WTF-8");
    // Repetition churns fresh dynamic strings; make the result collectable.
    Ok(w_str_from_wtf8_managed(buf))
}

pub(crate) unsafe fn bytes_concat(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let Some(b_src) = crate::typedef::buffer_as_bytes_like(b)? else {
        return Err(PyError::type_error(format!(
            "can't concat {} to {}",
            crate::baseobjspace::object_functionstr_type_name(b),
            crate::baseobjspace::object_functionstr_type_name(a)
        )));
    };
    let a_data = pyre_object::bytesobject::bytes_like_data(a);
    let b_data = pyre_object::bytesobject::bytes_like_data(b_src);
    let mut result = a_data.to_vec();
    result.extend_from_slice(b_data);
    Ok(if pyre_object::bytesobject::is_bytes(a) {
        pyre_object::bytesobject::w_bytes_from_bytes(&result)
    } else {
        pyre_object::bytearrayobject::w_bytearray_from_bytes(&result)
    })
}

pub(crate) unsafe fn bytes_repeat(s: PyObjectRef, n: PyObjectRef) -> PyResult {
    let data = pyre_object::bytesobject::bytes_like_data(s);
    let count = repeat_count(n)?;
    // A count of 1 on exact `bytes` (immutable) returns the receiver unchanged;
    // a subclass yields a fresh base `bytes`, and mutable `bytearray` copies.
    if count == 1 && pyre_object::pyobject::is_exact_type(s, &pyre_object::bytesobject::BYTES_TYPE)
    {
        return Ok(s);
    }
    let cap = data
        .len()
        .checked_mul(count)
        .ok_or_else(|| PyError::new(PyErrorKind::OverflowError, "repeated bytes are too long"))?;
    if cap > isize::MAX as usize {
        return Err(PyError::new(
            PyErrorKind::OverflowError,
            "repeated bytes are too long",
        ));
    }
    let mut buf: Vec<u8> = Vec::new();
    buf.try_reserve_exact(cap)
        .map_err(|_| PyError::new(PyErrorKind::MemoryError, ""))?;
    for _ in 0..count {
        buf.extend_from_slice(data);
    }
    Ok(if pyre_object::bytesobject::is_bytes(s) {
        pyre_object::bytesobject::w_bytes_from_bytes(&buf)
    } else {
        pyre_object::bytearrayobject::w_bytearray_from_bytes(&buf)
    })
}

pub(crate) unsafe fn list_concat(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let len_a = w_list_len(a);
    let len_b = w_list_len(b);
    let mut items = Vec::with_capacity(len_a + len_b);
    for i in 0..len_a {
        if let Some(item) = w_list_getitem(a, i as i64) {
            items.push(item);
        }
    }
    for i in 0..len_b {
        if let Some(item) = w_list_getitem(b, i as i64) {
            items.push(item);
        }
    }
    Ok(w_list_new(items))
}

pub(crate) unsafe fn tuple_concat(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let len_a = w_tuple_len(a);
    let len_b = w_tuple_len(b);
    let mut items = Vec::with_capacity(len_a + len_b);
    for i in 0..len_a {
        if let Some(item) = w_tuple_getitem(a, i as i64) {
            items.push(item);
        }
    }
    for i in 0..len_b {
        if let Some(item) = w_tuple_getitem(b, i as i64) {
            items.push(item);
        }
    }
    Ok(w_tuple_new(items))
}

/// listobject.py:638-641 descr_mul
pub(crate) unsafe fn list_repeat(list: PyObjectRef, n: PyObjectRef) -> PyResult {
    let count = repeat_count(n)?;
    let len = w_list_len(list);
    let cap = len
        .checked_mul(count)
        .ok_or_else(|| PyError::new(PyErrorKind::OverflowError, "list is too large"))?;
    // CPython 3.14 `list_resize`: the element count may fit Py_ssize_t while
    // `new_allocated * sizeof(PyObject*)` does not (gh-97616).
    if cap > (isize::MAX as usize) / std::mem::size_of::<PyObjectRef>() {
        return Err(PyError::new(PyErrorKind::MemoryError, ""));
    }
    let mut items: Vec<PyObjectRef> = Vec::new();
    items
        .try_reserve_exact(cap)
        .map_err(|_| PyError::new(PyErrorKind::MemoryError, ""))?;
    for _ in 0..count {
        for i in 0..len {
            if let Some(item) = w_list_getitem(list, i as i64) {
                items.push(item);
            }
        }
    }
    Ok(w_list_new(items))
}

/// listobject.py:645-648 descr_inplace_mul — repeat the list in place; the
/// list object identity is preserved.  Count and overflow handling mirror
/// `list_repeat`, but the extra copies are appended into the existing
/// storage instead of building a fresh list.
pub(crate) unsafe fn list_inplace_repeat(list: PyObjectRef, n: PyObjectRef) -> Result<(), PyError> {
    let count = repeat_count(n)?;
    if count == 0 {
        w_list_clear(list);
        return Ok(());
    }
    let len = w_list_len(list);
    if count == 1 || len == 0 {
        return Ok(());
    }
    let cap = len
        .checked_mul(count)
        .ok_or_else(|| PyError::new(PyErrorKind::OverflowError, "list is too large"))?;
    if cap > (isize::MAX as usize) / std::mem::size_of::<PyObjectRef>() {
        return Err(PyError::new(PyErrorKind::MemoryError, ""));
    }
    // Snapshot the original items so the growing list is not re-read while
    // the copies are appended.  Holding the refs across `w_list_append` is
    // the same idiom `list_method_extend` uses for its iterable branch.
    let snapshot = w_list_items_copy_as_vec(list);
    pyre_object::listobject::w_list_reserve_for_extend(list, cap - len);
    for _ in 1..count {
        for &item in &snapshot {
            pyre_object::listobject::w_list_append_preallocated(list, item);
        }
    }
    Ok(())
}

/// bytearrayobject.py descr_inplace_mul — repeat the bytearray in place,
/// preserving object identity.  A resize while the buffer is exported raises
/// BufferError, mirroring the `__iadd__` export guard; a count of 1 leaves the
/// length unchanged and needs no resize.
pub(crate) unsafe fn bytearray_inplace_repeat(
    ba: PyObjectRef,
    n: PyObjectRef,
) -> Result<(), PyError> {
    let count = repeat_count(n)?;
    let len = pyre_object::bytearrayobject::w_bytearray_len(ba);
    if count == 1 {
        return Ok(());
    }
    let new_size = len
        .checked_mul(count)
        .ok_or_else(|| PyError::new(PyErrorKind::OverflowError, "repeated bytes are too long"))?;
    // Only an actual length change touches the buffer size; an exported buffer
    // blocks it.
    if new_size != len {
        crate::builtins::bytearray_check_exports(ba)?;
    }
    if count == 0 {
        pyre_object::bytearrayobject::w_bytearray_vec_mut(ba).clear();
        pyre_object::bytearrayobject::w_bytearray_sync_alloc(ba, len);
        return Ok(());
    }
    if len == 0 {
        return Ok(());
    }
    let snapshot = pyre_object::bytearrayobject::w_bytearray_data(ba).to_vec();
    let vec = pyre_object::bytearrayobject::w_bytearray_vec_mut(ba);
    vec.try_reserve_exact(new_size - len)
        .map_err(|_| PyError::new(PyErrorKind::MemoryError, ""))?;
    for _ in 1..count {
        vec.extend_from_slice(&snapshot);
    }
    pyre_object::bytearrayobject::w_bytearray_sync_alloc(ba, len);
    Ok(())
}

// ── Comparison operations ─────────────────────────────────────────────

unsafe fn int_lt(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_bool_from(int_value(a) < int_value(b)))
}

unsafe fn int_le(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_bool_from(int_value(a) <= int_value(b)))
}

unsafe fn int_gt(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_bool_from(int_value(a) > int_value(b)))
}

unsafe fn int_ge(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_bool_from(int_value(a) >= int_value(b)))
}

unsafe fn int_eq(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_bool_from(int_value(a) == int_value(b)))
}

unsafe fn int_ne(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    Ok(w_bool_from(int_value(a) != int_value(b)))
}

/// pypy/objspace/std/longobject.py `_make_descr_cmp`: compare a W_LongObject's
/// rbigint payload directly with a W_IntObject's machine word.
unsafe fn long_int_compare(long: PyObjectRef, iother: i64, op: CompareOp) -> bool {
    let value = w_long_get_value(long);
    match op {
        CompareOp::Lt => value.int_lt(iother),
        CompareOp::Le => value.int_le(iother),
        CompareOp::Gt => value.int_gt(iother),
        CompareOp::Ge => value.int_ge(iother),
        CompareOp::Eq => value.int_eq(iother),
        CompareOp::Ne => value.int_ne(iother),
    }
}

/// Read a total-order result under `op` — the shape every `_memcmp`-based
/// comparison ends in, where the common prefix and then the lengths have
/// already been folded into one `Ordering`.
#[inline]
pub(crate) fn ordering_satisfies(ordering: std::cmp::Ordering, op: CompareOp) -> bool {
    match op {
        CompareOp::Lt => ordering.is_lt(),
        CompareOp::Le => ordering.is_le(),
        CompareOp::Gt => ordering.is_gt(),
        CompareOp::Ge => ordering.is_ge(),
        CompareOp::Eq => ordering.is_eq(),
        CompareOp::Ne => ordering.is_ne(),
    }
}

#[inline]
fn reverse_compare_op(op: CompareOp) -> CompareOp {
    match op {
        CompareOp::Lt => CompareOp::Gt,
        CompareOp::Le => CompareOp::Ge,
        CompareOp::Gt => CompareOp::Lt,
        CompareOp::Ge => CompareOp::Le,
        CompareOp::Eq => CompareOp::Eq,
        CompareOp::Ne => CompareOp::Ne,
    }
}

#[inline]
fn compare_f64(f1: f64, f2: f64, op: CompareOp) -> bool {
    match op {
        CompareOp::Lt => f1 < f2,
        CompareOp::Le => f1 <= f2,
        CompareOp::Gt => f1 > f2,
        CompareOp::Ge => f1 >= f2,
        CompareOp::Eq => f1 == f2,
        CompareOp::Ne => f1 != f2,
    }
}

/// `specialisedtupleobject.py:113-127 descr_eq`, the arm where both operands
/// are the SAME specialised class: the value slots compare raw, so neither
/// side pays the box `getitem` would have to build for an `_ii` / `_ff` slot.
///
/// `None` means the pair is not same-class — a mixed pair (one specialised,
/// one array-backed) still walks elementwise, which is what upstream does too.
///
/// # Safety
/// `a` and `b` must point to valid tuple objects.
unsafe fn specialised_tuple_same_class_eq(
    a: PyObjectRef,
    b: PyObjectRef,
) -> Result<Option<bool>, PyError> {
    if is_specialised_tuple_ii(a) && is_specialised_tuple_ii(b) {
        let equal = (0..2).all(|i| {
            w_specialised_tuple_ii_getvalue(a, i) == w_specialised_tuple_ii_getvalue(b, i)
        });
        return Ok(Some(equal));
    }
    if is_specialised_tuple_ff(a) && is_specialised_tuple_ff(b) {
        let equal = (0..2).all(|i| {
            let va = w_specialised_tuple_ff_getvalue(a, i);
            let vb = w_specialised_tuple_ff_getvalue(b, i);
            // Two NaNs compare unequal as doubles, but a tuple checks element
            // identity first, so the same NaN in both slots must still be
            // equal — `float2longlong` upstream, the raw bits here. `+0.0`
            // and `-0.0` differ in bits and are caught by the value compare.
            va == vb || va.to_bits() == vb.to_bits()
        });
        return Ok(Some(equal));
    }
    if is_specialised_tuple_oo(a) && is_specialised_tuple_oo(b) {
        for i in 0..2 {
            if !crate::baseobjspace::eq_w(
                w_specialised_tuple_oo_getvalue(a, i),
                w_specialised_tuple_oo_getvalue(b, i),
            )? {
                return Ok(Some(false));
            }
        }
        return Ok(Some(true));
    }
    Ok(None)
}

/// floatobject.py:106-129 `do_compare_bigint` — compare a float against a
/// bigint without converting the bigint to a double, which would round it.
fn do_compare_bigint(f1: f64, b2: &BigInt, op: CompareOp) -> bool {
    if matches!(op, CompareOp::Eq | CompareOp::Ne) {
        let ne = matches!(op, CompareOp::Ne);
        // A non-finite or fractional float is never equal to an integer.
        if !f1.is_finite() || f1.floor() != f1 {
            return ne;
        }
        return BigInt::_fromfloat_finite(f1).eq(b2) != ne;
    }
    if !f1.is_finite() {
        return compare_f64(f1, 0.0, op);
    }
    let f1 = if matches!(op, CompareOp::Gt | CompareOp::Le) {
        // 'float > long'  <==> 'ceil(float) > long'
        // 'float <= long' <==> 'ceil(float) <= long'
        f1.ceil()
    } else {
        // 'float < long'  <==> 'floor(float) < long'
        // 'float >= long' <==> 'floor(float) >= long'
        f1.floor()
    };
    let b1 = BigInt::_fromfloat_finite(f1);
    match op {
        CompareOp::Lt => b1.lt(b2),
        CompareOp::Le => b1.le(b2),
        CompareOp::Gt => b1.gt(b2),
        CompareOp::Ge => b1.ge(b2),
        CompareOp::Eq | CompareOp::Ne => unreachable!("handled above"),
    }
}

/// floatobject.py:132-148 `_compare` — the float side of a numeric
/// comparison.  `w_float` is a float; `w_other` is a float, an int, a bool
/// or a long.  The relation is evaluated exactly: only an int small enough
/// that a double represents it losslessly takes the plain f64 path.
unsafe fn float_compare(w_float: PyObjectRef, w_other: PyObjectRef, op: CompareOp) -> bool {
    let f1 = w_float_get_value(w_float);
    if is_float(w_other) {
        return compare_f64(f1, w_float_get_value(w_other), op);
    }
    if is_long(w_other) {
        return do_compare_bigint(f1, w_long_get_value(w_other), op);
    }
    let i2 = if is_bool(w_other) {
        w_bool_get_value(w_other) as i64
    } else {
        w_int_get_value(w_other)
    };
    // (double-)floats have always at least 48 bits of precision, so an int
    // whose bit 48 and above are a plain sign extension converts exactly.
    // `int_between(-1, i2 >> 48, 1)` (rarithmetic.py) is `-1 <= i2 >> 48 < 1`.
    let top = i2 >> 48;
    if !(-1..1).contains(&top) {
        return do_compare_bigint(f1, &BigInt::from(i2), op);
    }
    compare_f64(f1, i2 as f64, op)
}

// ── Complex arithmetic operations ────────────────────────────────────

/// `complexobject.c _PyHASH_IMAG` — the imaginary-part hash multiplier.
const HASH_IMAG: i64 = 1_000_003;

/// `(real, imag)` for any numeric operand (`complex` / `int` / `long` /
/// `float` / `bool`), else `None`.
pub(crate) unsafe fn complex_val(obj: PyObjectRef) -> Option<(f64, f64)> {
    if is_complex(obj) {
        Some((w_complex_get_real(obj), w_complex_get_imag(obj)))
    } else if is_bool(obj) {
        Some((w_bool_get_value(obj) as i64 as f64, 0.0))
    } else if is_int(obj) || is_long(obj) || is_float(obj) {
        Some((as_float(obj), 0.0))
    } else {
        None
    }
}

/// True if both operands are numeric and at least one is `complex`.
unsafe fn is_complex_pair(a: PyObjectRef, b: PyObjectRef) -> bool {
    (is_complex(a) || is_complex(b)) && complex_val(a).is_some() && complex_val(b).is_some()
}

unsafe fn complex_add(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let (ar, ai) = complex_val(a).unwrap();
    let (br, bi) = complex_val(b).unwrap();
    reject_float_coercion_overflow(a, ar)?;
    reject_float_coercion_overflow(b, br)?;
    // CPython 3.14 complexobject.c COMPLEX_BINOP(add, sum): mixed real /
    // complex addition uses _Py_cr_sum / _Py_rc_sum and leaves the complex
    // imaginary lane untouched.  Besides matching C11 Annex G mixed-mode
    // arithmetic, this preserves the sign of an imaginary zero.
    if is_complex(a) && is_complex(b) {
        Ok(w_complex_new(ar + br, ai + bi))
    } else if is_complex(a) {
        Ok(w_complex_new(ar + br, ai))
    } else {
        Ok(w_complex_new(ar + br, bi))
    }
}

unsafe fn complex_sub(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let (ar, ai) = complex_val(a).unwrap();
    let (br, bi) = complex_val(b).unwrap();
    reject_float_coercion_overflow(a, ar)?;
    reject_float_coercion_overflow(b, br)?;
    // CPython 3.14 _Py_c_diff / _Py_cr_diff / _Py_rc_diff.  In the mixed
    // cases only the real lane is combined; real-complex negates the complex
    // imaginary lane directly instead of subtracting it from +0.0.
    if is_complex(a) && is_complex(b) {
        Ok(w_complex_new(ar - br, ai - bi))
    } else if is_complex(a) {
        Ok(w_complex_new(ar - br, ai))
    } else {
        Ok(w_complex_new(ar - br, -bi))
    }
}

unsafe fn complex_mul(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let (ar, ai) = complex_val(a).unwrap();
    let (br, bi) = complex_val(b).unwrap();
    reject_float_coercion_overflow(a, ar)?;
    reject_float_coercion_overflow(b, br)?;
    if is_complex(a) && is_complex(b) {
        Ok(complex_prod(ar, ai, br, bi))
    } else if is_complex(a) {
        // CPython 3.14 _Py_cr_prod: multiply each existing complex lane by
        // the real operand, without manufacturing a zero imaginary lane.
        Ok(w_complex_new(ar * br, ai * br))
    } else {
        // _Py_rc_prod(a, b) delegates to _Py_cr_prod(b, a).
        Ok(w_complex_new(br * ar, bi * ar))
    }
}

/// CPython 3.14 `complexobject.c _Py_c_prod`, including the C11 Annex G.5.1
/// recovery for infinities that first produce `nan+nanj`.
unsafe fn complex_prod(mut a: f64, mut b: f64, mut c: f64, mut d: f64) -> PyObjectRef {
    let ac = a * c;
    let bd = b * d;
    let ad = a * d;
    let bc = b * c;
    let mut real = ac - bd;
    let mut imag = ad + bc;
    if real.is_nan() && imag.is_nan() {
        let mut recalc = false;
        if a.is_infinite() || b.is_infinite() {
            a = float_copysign(if a.is_infinite() { 1.0 } else { 0.0 }, a);
            b = float_copysign(if b.is_infinite() { 1.0 } else { 0.0 }, b);
            if c.is_nan() {
                c = float_copysign(0.0, c);
            }
            if d.is_nan() {
                d = float_copysign(0.0, d);
            }
            recalc = true;
        }
        if c.is_infinite() || d.is_infinite() {
            c = float_copysign(if c.is_infinite() { 1.0 } else { 0.0 }, c);
            d = float_copysign(if d.is_infinite() { 1.0 } else { 0.0 }, d);
            if a.is_nan() {
                a = float_copysign(0.0, a);
            }
            if b.is_nan() {
                b = float_copysign(0.0, b);
            }
            recalc = true;
        }
        if !recalc && (ac.is_infinite() || bd.is_infinite() || ad.is_infinite() || bc.is_infinite())
        {
            if a.is_nan() {
                a = float_copysign(0.0, a);
            }
            if b.is_nan() {
                b = float_copysign(0.0, b);
            }
            if c.is_nan() {
                c = float_copysign(0.0, c);
            }
            if d.is_nan() {
                d = float_copysign(0.0, d);
            }
            recalc = true;
        }
        if recalc {
            real = f64::INFINITY * (a * c - b * d);
            imag = f64::INFINITY * (a * d + b * c);
        }
    }
    w_complex_new(real, imag)
}

/// CPython 3.14 `complexobject.c _Py_c_quot`: Smith's stable division plus
/// the C11 Annex G.5.2 recovery for infinite numerators and denominators.
unsafe fn complex_quot(ar: f64, ai: f64, br: f64, bi: f64) -> PyObjectRef {
    let abs_br = br.abs();
    let abs_bi = bi.abs();
    let mut real: f64;
    let mut imag: f64;
    if abs_br >= abs_bi {
        if abs_br == 0.0 {
            // `_Py_c_quot` writes 0+0j and signals EDOM through errno.  The
            // caller performs that side-channel check before using the pair.
            return w_complex_new(0.0, 0.0);
        }
        let ratio = bi / br;
        let denom = br + bi * ratio;
        real = (ar + ai * ratio) / denom;
        imag = (ai - ar * ratio) / denom;
    } else if abs_bi >= abs_br {
        let ratio = br / bi;
        let denom = br * ratio + bi;
        real = (ar * ratio + ai) / denom;
        imag = (ai * ratio - ar) / denom;
    } else {
        real = f64::NAN;
        imag = f64::NAN;
    }
    if real.is_nan() && imag.is_nan() {
        if (ar.is_infinite() || ai.is_infinite()) && br.is_finite() && bi.is_finite() {
            let x = float_copysign(if ar.is_infinite() { 1.0 } else { 0.0 }, ar);
            let y = float_copysign(if ai.is_infinite() { 1.0 } else { 0.0 }, ai);
            real = f64::INFINITY * (x * br + y * bi);
            imag = f64::INFINITY * (y * br - x * bi);
        } else if (abs_br.is_infinite() || abs_bi.is_infinite()) && ar.is_finite() && ai.is_finite()
        {
            let x = float_copysign(if br.is_infinite() { 1.0 } else { 0.0 }, br);
            let y = float_copysign(if bi.is_infinite() { 1.0 } else { 0.0 }, bi);
            real = 0.0 * (ar * x + ai * y);
            imag = 0.0 * (ai * x - ar * y);
        }
    }
    w_complex_new(real, imag)
}

/// CPython 3.14 `_Py_rc_quot`: real divided by complex, with its distinct
/// signed-zero recovery when the denominator contains an infinity.
unsafe fn real_complex_quot(a: f64, br: f64, bi: f64) -> PyObjectRef {
    let abs_br = br.abs();
    let abs_bi = bi.abs();
    let mut real: f64;
    let mut imag: f64;
    if abs_br >= abs_bi {
        if abs_br == 0.0 {
            return w_complex_new(0.0, 0.0);
        }
        let ratio = bi / br;
        let denom = br + bi * ratio;
        real = a / denom;
        imag = (-a * ratio) / denom;
    } else if abs_bi >= abs_br {
        let ratio = br / bi;
        let denom = br * ratio + bi;
        real = (a * ratio) / denom;
        imag = (-a) / denom;
    } else {
        real = f64::NAN;
        imag = f64::NAN;
    }
    if real.is_nan()
        && imag.is_nan()
        && a.is_finite()
        && (abs_br.is_infinite() || abs_bi.is_infinite())
    {
        let x = float_copysign(if br.is_infinite() { 1.0 } else { 0.0 }, br);
        let y = float_copysign(if bi.is_infinite() { 1.0 } else { 0.0 }, bi);
        real = 0.0 * (a * x);
        imag = 0.0 * (-a * y);
    }
    w_complex_new(real, imag)
}

unsafe fn complex_truediv(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let (ar, ai) = complex_val(a).unwrap();
    let (br, bi) = complex_val(b).unwrap();
    reject_float_coercion_overflow(a, ar)?;
    reject_float_coercion_overflow(b, br)?;
    if br == 0.0 && bi == 0.0 {
        return Err(PyError::zero_division(ZERO_DIVISION_MSG));
    }
    if is_complex(a) && is_complex(b) {
        Ok(complex_quot(ar, ai, br, bi))
    } else if is_complex(a) {
        // CPython 3.14 _Py_cr_quot.
        Ok(w_complex_new(ar / br, ai / br))
    } else {
        Ok(real_complex_quot(ar, br, bi))
    }
}

unsafe fn complex_powi(a: PyObjectRef, exponent: i64) -> PyResult {
    let (mut base_real, mut base_imag) = complex_val(a).unwrap();
    let mut result_real = 1.0;
    let mut result_imag = 0.0;
    let mut n = exponent.unsigned_abs();
    while n != 0 {
        if n & 1 != 0 {
            // CPython 3.14 c_powu calls _Py_c_prod for every multiply, so
            // the Annex G infinity recovery is shared with ordinary `*`.
            let product = complex_prod(result_real, result_imag, base_real, base_imag);
            result_real = w_complex_get_real(product);
            result_imag = w_complex_get_imag(product);
        }
        n >>= 1;
        if n != 0 {
            let square = complex_prod(base_real, base_imag, base_real, base_imag);
            base_real = w_complex_get_real(square);
            base_imag = w_complex_get_imag(square);
        }
    }
    let result = if exponent < 0 {
        complex_truediv(
            w_complex_new(1.0, 0.0),
            w_complex_new(result_real, result_imag),
        )?
    } else {
        w_complex_new(result_real, result_imag)
    };
    // complexobject.c complex_pow: `_Py_ADJUST_ERANGE2` forces ERANGE when
    // either component is infinite; the numeric slot reports that as
    // `OverflowError("complex exponentiation")`.
    if w_complex_get_real(result).is_infinite() || w_complex_get_imag(result).is_infinite() {
        Err(PyError::overflow_error("complex exponentiation"))
    } else {
        Ok(result)
    }
}

/// `complexobject.c _Py_c_pow`.
unsafe fn complex_pow(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let (ar, ai) = complex_val(a).unwrap();
    let (br, bi) = complex_val(b).unwrap();
    let (real, imag) = if br == 0.0 && bi == 0.0 {
        (1.0, 0.0)
    } else if ar == 0.0 && ai == 0.0 {
        if bi != 0.0 || br < 0.0 {
            return Err(PyError::zero_division(
                "zero to a negative or complex power",
            ));
        }
        (0.0, 0.0)
    } else if bi == 0.0 && (-100.0..=100.0).contains(&br) && br == br.trunc() {
        return complex_powi(a, br as i64);
    } else {
        let vabs = ar.hypot(ai);
        let mut len = vabs.powf(br);
        let at = ai.atan2(ar);
        let mut phase = at * br;
        if bi != 0.0 {
            len /= (at * bi).exp();
            phase += bi * vabs.ln();
        }
        (len * phase.cos(), len * phase.sin())
    };
    if real.is_infinite() || imag.is_infinite() {
        Err(PyError::overflow_error("complex exponentiation"))
    } else {
        Ok(w_complex_new(real, imag))
    }
}

unsafe fn complex_neg(a: PyObjectRef) -> PyResult {
    let (ar, ai) = complex_val(a).unwrap();
    Ok(w_complex_new(-ar, -ai))
}

/// `abs(complex)` → the float magnitude `hypot(real, imag)`.
pub(crate) unsafe fn complex_abs(a: PyObjectRef) -> PyResult {
    let (ar, ai) = complex_val(a).unwrap();
    let result = ar.hypot(ai);
    if result.is_infinite() && ar.is_finite() && ai.is_finite() {
        return Err(PyError::overflow_error("absolute value too large"));
    }
    Ok(w_float_new(result))
}

/// Complex equality: `==`/`!=` only (no ordering).  Mixed numeric
/// operands compare equal when the imaginary part is zero.
unsafe fn complex_richcompare(a: PyObjectRef, b: PyObjectRef, op: CompareOp) -> PyResult {
    if !matches!(op, CompareOp::Eq | CompareOp::Ne) {
        return Ok(w_not_implemented());
    }
    let equal = if is_complex(a) && is_complex(b) {
        w_complex_get_real(a) == w_complex_get_real(b)
            && w_complex_get_imag(a) == w_complex_get_imag(b)
    } else {
        // `complexobject.py descr_eq`: compare a real-only complex through
        // float/int equality.  Do not round an arbitrary-size integer to f64;
        // convert the integral float lane to BigInt and compare exactly.
        let (z, other) = if is_complex(a) { (a, b) } else { (b, a) };
        let real = w_complex_get_real(z);
        if w_complex_get_imag(z) != 0.0 {
            false
        } else if is_float(other) {
            real == w_float_get_value(other)
        } else if is_int(other) || is_long(other) || is_bool(other) {
            real.is_finite()
                && real.fract() == 0.0
                && BigInt::from_f64(real).is_some_and(|value| {
                    if is_long(other) {
                        value.eq(w_long_get_value(other))
                    } else {
                        value.int_eq(int_value(other))
                    }
                })
        } else {
            return Ok(w_not_implemented());
        }
    };
    Ok(w_bool_from(if matches!(op, CompareOp::Eq) {
        equal
    } else {
        !equal
    }))
}

/// `complexobject.c complex_hash` — `hash(real) + _PyHASH_IMAG * hash(imag)`.
pub(crate) fn complex_hash(obj: PyObjectRef, real: f64, imag: f64) -> i64 {
    // `complexobject.py descr_hash`: each NaN lane uses the containing
    // complex object's identity hash, not the float HASH_NAN sentinel.
    let identity = || crate::typedef::default_identity_hash_value(obj);
    let hr = if real.is_nan() {
        identity()
    } else {
        crate::builtins::_hash_float(real)
    };
    let hi = if imag.is_nan() {
        identity()
    } else {
        crate::builtins::_hash_float(imag)
    };
    let combined = hr.wrapping_add(hi.wrapping_mul(HASH_IMAG));
    if combined == -1 { -2 } else { combined }
}

// ── Public dispatch API ───────────────────────────────────────────────

/// Check if w_type is a subtype of cls — delegates to the single MRO
/// membership scan in `pyre_object::w_type_issubtype`.
unsafe fn issubtype_cached(w_type: PyObjectRef, cls: PyObjectRef) -> bool {
    pyre_object::w_type_issubtype(w_type, cls)
}

/// Comparison special-method dispatch.
///
/// PyPy: `descroperation.py:_make_comparison_impl`.  Generic instances use
/// the complete MRO lookup, including inherited `object.__eq__` /
/// `object.__ne__`; the latter deliberately calls the receiver's live
/// `__eq__`.  Builtin layouts retain the override-only lookup because their
/// inherited slots delegate straight back into [`compare`] and would recurse
/// forever.  The reflected method of a proper subclass has priority.  PyPy
/// 3.11 suppresses the second call for equal user-defined types, but CPython
/// 3.14 `Objects/object.c:do_richcompare` tries `tp_richcompare` in both
/// operand directions when the first call returns `NotImplemented`; pyre's
/// Python 3.14 compatibility rule takes precedence here.
unsafe fn try_compare_override(
    a: PyObjectRef,
    b: PyObjectRef,
    op: CompareOp,
) -> Result<Option<PyObjectRef>, PyError> {
    let dunder = match op {
        CompareOp::Lt => "__lt__",
        CompareOp::Le => "__le__",
        CompareOp::Gt => "__gt__",
        CompareOp::Ge => "__ge__",
        CompareOp::Eq => "__eq__",
        CompareOp::Ne => "__ne__",
    };
    let rdunder = reverse_dunder(dunder).unwrap_or(dunder);
    let comparison_method = |obj: PyObjectRef, name: &str| {
        if is_instance(obj) {
            let w_type = crate::typedef::r#type(obj)?;
            let method = lookup_in_type_where(w_type.as_ptr(), name)?;
            Some((method, w_type.as_ptr()))
        } else {
            crate::baseobjspace::subclass_special_override(obj, name)
        }
    };
    let a_type = crate::typedef::r#type(a);
    let b_type = crate::typedef::r#type(b);
    let a_ov = comparison_method(a, dunder);
    let b_ov = comparison_method(b, rdunder);
    if a_ov.is_none() && b_ov.is_none() {
        return Ok(None);
    }
    // Python's "subclass reflected op takes priority": if `b`'s type is a
    // proper subclass of `a`'s type and `b` overrides the reflected op, run it
    // first.
    let b_first = b_ov.is_some()
        && match (a_type, b_type) {
            (Some(at), Some(bt)) => at != bt && issubtype_cached(bt.as_ptr(), at.as_ptr()),
            _ => false,
        };
    let order = if b_first {
        [(b_ov, b, a), (a_ov, a, b)]
    } else {
        [(a_ov, a, b), (b_ov, b, a)]
    };
    for (ov, recv, other) in order {
        if let Some((method, w_type)) = ov
            && let Some(result) = unsafe { invoke_comparison(method, recv, w_type, other) }?
        {
            return Ok(Some(result));
        }
    }
    Ok(None)
}

/// PyPy `descroperation.py:_invoke_comparison`.
///
/// A plain function is called directly, so every exception from its body is
/// observable. Other descriptors bind through `space.get`; only an
/// `AttributeError` raised by that binding step means that this comparison
/// implementation is absent (notably `__eq__ = property(...)`).
unsafe fn invoke_comparison(
    w_descr: PyObjectRef,
    w_obj: PyObjectRef,
    w_type: PyObjectRef,
    w_other: PyObjectRef,
) -> Result<Option<PyObjectRef>, PyError> {
    let direct_function = std::ptr::eq(
        unsafe { (*w_descr).ob_type },
        &crate::function::FUNCTION_TYPE as *const _,
    ) || std::ptr::eq(
        unsafe { (*w_descr).ob_type },
        &crate::function::METHOD_DESCRIPTOR_TYPE as *const _,
    );
    let result = if direct_function {
        crate::call::call_function_impl_result(w_descr, &[w_obj, w_other])?
    } else {
        let w_impl = match unsafe { crate::baseobjspace::get(w_descr, w_obj, w_type) } {
            Ok(Some(w_impl)) => w_impl,
            Ok(None) => w_descr,
            Err(err) if err.kind == PyErrorKind::AttributeError => return Ok(None),
            Err(err) => return Err(err),
        };
        crate::call::call_function_impl_result(w_impl, &[w_other])?
    };
    if is_not_implemented(result) {
        Ok(None)
    } else {
        Ok(Some(result))
    }
}

/// Map forward dunder to reverse dunder.
/// PyPy: descroperation.py `_make_binop_impl` generates both directions.
fn reverse_dunder(dunder: &str) -> Option<&'static str> {
    Some(match dunder {
        // Arithmetic — PyPy: descroperation.py _make_binop_impl
        "__add__" => "__radd__",
        "__sub__" => "__rsub__",
        "__mul__" => "__rmul__",
        "__truediv__" => "__rtruediv__",
        "__floordiv__" => "__rfloordiv__",
        "__mod__" => "__rmod__",
        "__matmul__" => "__rmatmul__",
        "__pow__" => "__rpow__",
        "__lshift__" => "__rlshift__",
        "__rshift__" => "__rrshift__",
        "__and__" => "__rand__",
        "__or__" => "__ror__",
        "__xor__" => "__rxor__",
        // Comparison reflected — PyPy: descroperation.py _cmp_dispatch
        "__lt__" => "__gt__",
        "__le__" => "__ge__",
        "__gt__" => "__lt__",
        "__ge__" => "__le__",
        "__eq__" => "__eq__",
        "__ne__" => "__ne__",
        _ => return None,
    })
}

/// Try to call a unary dunder on an instance.
///
/// PyPy: `ObjSpace.call_function(space.lookup(w_obj, dunder), w_obj)`
/// The Python-level OperationError must propagate to the caller; use the
/// Result-returning call path so PENDING_CALL_ERROR is consumed.
unsafe fn try_instance_unaryop(
    a: PyObjectRef,
    dunder: &str,
) -> Result<Option<PyObjectRef>, PyError> {
    if is_instance(a)
        && let Some(method) = lookup(a, dunder)
    {
        let Some(w_type) = crate::typedef::r#type(a) else {
            return Ok(None);
        };
        return Ok(Some(crate::baseobjspace::get_and_call_function(
            method,
            a,
            w_type.as_ptr(),
            &[],
        )?));
    }
    Ok(None)
}

/// True when `obj`'s type defines `dunder` in a class other than the
/// builtin type object `tp` — i.e. a subclass overrides it.  Builtin
/// `str`/`list`/`tuple` install `__add__`/`__radd__` on their own type;
/// an inherited (non-overridden) lookup resolves back to `tp`.
unsafe fn dunder_overridden(obj: PyObjectRef, dunder: &str, tp: PyObjectRef) -> bool {
    match crate::typedef::r#type(obj)
        .and_then(|t| lookup_where_with_method_cache(t.as_ptr(), dunder))
    {
        Some((src, _)) => !std::ptr::eq(src, tp),
        None => false,
    }
}

/// Builtin sequence base selected by [`needs_seq_binop_dispatch`].  The
/// caller passes this discriminant instead of a `&STR_TYPE`/… static so
/// the type-static load stays inside the residual helper, off the traced
/// `add` graph.
#[derive(Clone, Copy)]
enum SeqBase {
    Str,
    List,
    Tuple,
}

/// descroperation.py:708 `binop_impl` shortcut — the builtin sequence
/// fast path (`str`/`list`/`tuple` concat) bypasses `__op__`/`__rop__`
/// dispatch unless one operand is a subclass that actually overrides the
/// forward or reflected special method (descroperation.py:664 "unicode +
/// string subclass").  Returns `false` when no override exists so the
/// caller concatenates directly — this also avoids re-entering the
/// builtin `__add__` slot, which would recurse back into `add`.
///
/// `dont_look_inside`: the builtin-base type static is loaded here, so a
/// traced caller emits a residual call instead of carrying an
/// unresolvable `LoadStatic` into its JitCode.
#[majit_macros::dont_look_inside]
unsafe fn needs_seq_binop_dispatch(
    a: PyObjectRef,
    b: PyObjectRef,
    base: SeqBase,
    fwd: &str,
    rev: &str,
) -> bool {
    let tp: *const pyre_object::PyType = match base {
        SeqBase::Str => &pyre_object::STR_TYPE,
        SeqBase::List => &pyre_object::LIST_TYPE,
        SeqBase::Tuple => &pyre_object::TUPLE_TYPE,
    };
    let Some(t) = crate::typedef::gettypefor(tp) else {
        return false;
    };
    let t = t.as_ptr();
    dunder_overridden(a, fwd, t)
        || dunder_overridden(a, rev, t)
        || dunder_overridden(b, fwd, t)
        || dunder_overridden(b, rev, t)
}

/// `bytes`/`bytearray` analog of `needs_seq_binop_dispatch`.  The two
/// builtin types share one `+` branch, so each operand is judged against
/// its own builtin base (`bytes` vs `bytearray`).  `dont_look_inside`
/// keeps the base type-static loads in the residual call, off the traced
/// `add` graph.
#[majit_macros::dont_look_inside]
unsafe fn needs_bytes_binop_dispatch(a: PyObjectRef, b: PyObjectRef, fwd: &str, rev: &str) -> bool {
    bytes_operand_overrides(a, fwd, rev) || bytes_operand_overrides(b, fwd, rev)
}

/// True when `obj`'s type overrides `fwd`/`rev` relative to its builtin
/// base (`bytes` or `bytearray`).  Only reached from the residual
/// `needs_bytes_binop_dispatch`, so the type-static loads never enter a
/// traced graph.
unsafe fn bytes_operand_overrides(obj: PyObjectRef, fwd: &str, rev: &str) -> bool {
    let tp: *const pyre_object::PyType = if pyre_object::bytesobject::is_bytes(obj) {
        &pyre_object::bytesobject::BYTES_TYPE
    } else {
        &pyre_object::bytearrayobject::BYTEARRAY_TYPE
    };
    let Some(t) = crate::typedef::gettypefor(tp) else {
        return false;
    };
    let t = t.as_ptr();
    dunder_overridden(obj, fwd, t) || dunder_overridden(obj, rev, t)
}

/// `needs_seq_binop_dispatch` for the `sq_repeat` branches of [`mul`], where
/// only one operand is the sequence: a subclass that overrides the multiply
/// specials relative to its own builtin base has to run that override instead
/// of repeating, exactly as the concat branches of [`add`] are gated.
///
/// Judged one operand at a time rather than as a pair, because a repeat's other
/// operand is the multiplier — an `int`, whose own `__mul__` lives on `int` and
/// would read as an override against any sequence base.  A *non*-overriding
/// subclass keeps the repeat path, which is what makes `L([1]) * 2` still a
/// plain list repetition.
///
/// `dont_look_inside`: the builtin-base type statics are loaded here, so a
/// traced caller emits a residual call rather than an unresolvable
/// `LoadStatic`.
#[majit_macros::dont_look_inside]
pub(crate) unsafe fn seq_repeat_override(obj: PyObjectRef, dunders: &[&str]) -> bool {
    if pyre_object::is_exact_builtin_instance(obj) {
        return false;
    }
    let tp: *const pyre_object::PyType = if is_str(obj) {
        &pyre_object::STR_TYPE
    } else if is_list(obj) {
        &pyre_object::LIST_TYPE
    } else if is_tuple(obj) {
        &pyre_object::TUPLE_TYPE
    } else if pyre_object::bytesobject::is_bytes_like(obj) {
        if pyre_object::bytesobject::is_bytes(obj) {
            &pyre_object::bytesobject::BYTES_TYPE
        } else {
            &pyre_object::bytearrayobject::BYTEARRAY_TYPE
        }
    } else {
        return false;
    };
    let Some(t) = crate::typedef::gettypefor(tp) else {
        return false;
    };
    let t = t.as_ptr();
    dunders
        .iter()
        .any(|dunder| dunder_overridden(obj, dunder, t))
}

/// True when `obj` is an exact builtin numeric instance
/// (`int`/`long`/`float`/`complex`/`bool`, not a subclass).  These types
/// define no in-place special method (`__iadd__` etc.), so
/// [`try_inplace_special`] can skip the in-place lookup for them.  A
/// subclass ([`is_exact_builtin_instance`] false) may override the slot, so
/// it is excluded.
unsafe fn is_exact_numeric_builtin(obj: PyObjectRef) -> bool {
    pyre_object::is_exact_builtin_instance(obj)
        && (is_int(obj) || is_long(obj) || is_float(obj) || is_complex(obj))
}

/// The builtin numeric base type backing `obj`'s storage — `int` for
/// int/long, `float` for float, `complex` for complex, `None` for a
/// non-numeric operand.
unsafe fn numeric_base_type(obj: PyObjectRef) -> Option<*const pyre_object::PyType> {
    if is_int(obj) || is_long(obj) {
        Some(&pyre_object::INT_TYPE)
    } else if is_float(obj) {
        Some(&pyre_object::FLOAT_TYPE)
    } else if is_complex(obj) {
        Some(&pyre_object::COMPLEX_TYPE)
    } else {
        None
    }
}

/// The builtin numeric base type object of `obj`, returned only when `obj`
/// is a subclass instance that may override a special method.  An exact
/// builtin numeric instance ([`is_exact_builtin_instance`]) cannot override
/// any special method, so it yields `None`, skipping the string-keyed
/// method-cache lookups in [`dunder_overridden`].  Those lookups (one
/// UTF-8-keyed cache probe per dunder) are the dominant per-iteration cost
/// of the interpreter numeric fast path.  This mirrors the exact-builtin
/// gate already opening `subclass_special_override` / `is_true`.
unsafe fn numeric_base_type_of_overriding_subclass(
    obj: PyObjectRef,
) -> Option<std::ptr::NonNull<pyre_object::PyObject>> {
    if pyre_object::is_exact_builtin_instance(obj) {
        return None;
    }
    let base = numeric_base_type(obj)?;
    crate::typedef::gettypefor(base)
}

/// True when numeric operand `obj` (int/long/float storage) has a Python
/// class that overrides `dunder` relative to its builtin base — int/long
/// against `int`, float against `float`, complex against `complex`.  Mirrors [`dunder_overridden`]
/// with the numeric base selected per operand.  Only an *overriding*
/// subclass routes through dispatch: a non-overriding subclass keeps the
/// Rust fast path, which both matches the builtin result and avoids
/// re-entering the inherited slot (it would recurse back into this op).
unsafe fn numeric_operand_overrides(obj: PyObjectRef, dunder: &str, rdunder: &str) -> bool {
    let Some(t) = numeric_base_type_of_overriding_subclass(obj) else {
        return false;
    };
    let t = t.as_ptr();
    dunder_overridden(obj, dunder, t) || dunder_overridden(obj, rdunder, t)
}

/// descroperation.py:708 `binop_impl` shortcut — the builtin numeric
/// (int/long/float) fast path bypasses `__op__`/`__rop__` dispatch unless
/// an operand is a subclass that actually overrides the forward or
/// reflected special method.  The seq/bytes analogs are
/// [`needs_seq_binop_dispatch`]/[`needs_bytes_binop_dispatch`].
///
/// `dont_look_inside`: the type-static + typeobject-registry loads stay in
/// this residual helper, off the traced numeric graph; the hot int path is
/// specialized separately via `guard_class` in the JIT.
#[majit_macros::dont_look_inside]
unsafe fn needs_numeric_binop_dispatch(
    a: PyObjectRef,
    b: PyObjectRef,
    fwd: &str,
    rev: &str,
) -> bool {
    numeric_operand_overrides(a, fwd, rev) || numeric_operand_overrides(b, fwd, rev)
}

/// Set/frozenset analogue of the numeric override gate.  The storage fast
/// paths below are valid for exact builtins, but a heap subclass must enter
/// `_call_binop_impl` so its forward/reflected override and the reflected-
/// subclass priority are observed before inherited set semantics.
///
/// `dont_look_inside`: `is_exact_type` loads the builtin `SET_TYPE`/
/// `FROZENSET_TYPE` statics here, so keep that load in this residual helper
/// off the traced graph — mirrors `needs_seq`/`needs_bytes`/`needs_numeric`.
#[majit_macros::dont_look_inside]
unsafe fn needs_set_binop_dispatch(a: PyObjectRef, b: PyObjectRef) -> bool {
    let is_exact_setlike = |obj| {
        pyre_object::is_exact_type(obj, &pyre_object::setobject::SET_TYPE)
            || pyre_object::is_exact_type(obj, &pyre_object::setobject::FROZENSET_TYPE)
    };
    (pyre_object::is_set_or_frozenset(a) && !is_exact_setlike(a))
        || (pyre_object::is_set_or_frozenset(b) && !is_exact_setlike(b))
}

/// Unary analog: true when numeric operand `a` overrides the unary
/// special `dunder` relative to its builtin base.
#[majit_macros::dont_look_inside]
unsafe fn needs_numeric_unaryop_dispatch(a: PyObjectRef, dunder: &str) -> bool {
    let Some(t) = numeric_base_type_of_overriding_subclass(a) else {
        return false;
    };
    dunder_overridden(a, dunder, t.as_ptr())
}

/// Call the overriding unary special on a numeric subclass operand before
/// the Rust fast path.  Returns `None` when `a` is an exact builtin
/// numeric or does not override `dunder`, so the caller falls through.
unsafe fn try_numeric_unaryop_override(
    a: PyObjectRef,
    dunder: &str,
) -> Result<Option<PyObjectRef>, PyError> {
    if !needs_numeric_unaryop_dispatch(a, dunder) {
        return Ok(None);
    }
    let Some(t) = crate::typedef::r#type(a) else {
        return Ok(None);
    };
    let Some(method) = lookup_in_type_where(t.as_ptr(), dunder) else {
        return Ok(None);
    };
    Ok(Some(crate::call::call_function_impl_result(method, &[a])?))
}

pub fn add(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        let numeric_override = needs_numeric_binop_dispatch(a, b, "__add__", "__radd__");
        if numeric_override
            && let Some(result) = try_dispatch_binary_special(a, b, "__add__", "__radd__")?
        {
            return Ok(result);
        }
        if !numeric_override {
            if is_int_like(a) && is_int_like(b) {
                return int_add(a, b);
            }
            if is_int_or_long(a) && is_int_or_long(b) {
                return long_add(a, b);
            }
            if is_float_pair(a, b) {
                return float_add(a, b);
            }
            if is_complex_pair(a, b) {
                return complex_add(a, b);
            }
        }
        if is_str(a) && is_str(b) {
            // descroperation.py:664 "unicode + string subclass" — a str
            // subclass overriding `__add__`/`__radd__` must reach the
            // reflected dispatch; otherwise concat directly.
            if needs_seq_binop_dispatch(a, b, SeqBase::Str, "__add__", "__radd__")
                && let Some(result) = try_dispatch_binary_special(a, b, "__add__", "__radd__")?
            {
                return Ok(result);
            }
            return str_concat(a, b);
        }
        if is_list(a) && is_list(b) {
            if needs_seq_binop_dispatch(a, b, SeqBase::List, "__add__", "__radd__")
                && let Some(result) = try_dispatch_binary_special(a, b, "__add__", "__radd__")?
            {
                return Ok(result);
            }
            return list_concat(a, b);
        }
        if is_tuple(a) && is_tuple(b) {
            if needs_seq_binop_dispatch(a, b, SeqBase::Tuple, "__add__", "__radd__")
                && let Some(result) = try_dispatch_binary_special(a, b, "__add__", "__radd__")?
            {
                return Ok(result);
            }
            return tuple_concat(a, b);
        }
        // `bytes`/`bytearray` `__add__` accepts any buffer on the right (a
        // memoryview included), and the result type follows the left operand:
        // `bytes + <buffer>` is bytes, `bytearray + <buffer>` is bytearray.
        if pyre_object::bytesobject::is_bytes_like(a) {
            if let Some(b_src) = crate::typedef::buffer_as_bytes_like(b)? {
                // Only a real bytes-like rhs can carry a subclass `__radd__`;
                // a memoryview cannot, so dispatch only when both are bytes-like.
                if pyre_object::bytesobject::is_bytes_like(b)
                    && needs_bytes_binop_dispatch(a, b, "__add__", "__radd__")
                    && let Some(result) = try_dispatch_binary_special(a, b, "__add__", "__radd__")?
                {
                    return Ok(result);
                }
                return bytes_concat(a, b_src);
            }
            if let Some(result) = try_dispatch_binary_special(a, b, "__add__", "__radd__")? {
                return Ok(result);
            }
            // A non-buffer rhs is rejected with the generic operator TypeError
            // (bytes `descr_add` returns NotImplemented), not "can't concat".
            let a_name = crate::baseobjspace::object_functionstr_type_name(a);
            let b_name = crate::baseobjspace::object_functionstr_type_name(b);
            return Err(PyError::type_error(format!(
                "unsupported operand type(s) for +: '{a_name}' and '{b_name}'"
            )));
        }
        // Forward `__add__` + reflected `__radd__` per
        // `descroperation.py:_make_binop_impl` — try_dispatch_binary_special
        // already implements the reflected-first reordering rule for
        // subclass operands.
        if !numeric_override
            && let Some(result) = try_dispatch_binary_special(a, b, "__add__", "__radd__")?
        {
            return Ok(result);
        }
        let a_name = crate::baseobjspace::object_functionstr_type_name(a);
        let b_name = crate::baseobjspace::object_functionstr_type_name(b);
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for +: '{}' and '{}'",
            a_name, b_name,
        )))
    }
}

pub fn matmul(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        if let Some(result) = try_dispatch_binary_special(a, b, "__matmul__", "__rmatmul__")? {
            return Ok(result);
        }
        let a_name = (*ll_type(a)).name;
        let b_name = (*ll_type(b)).name;
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for @: '{a_name}' and '{b_name}'"
        )))
    }
}

pub fn sub(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        let set_override = needs_set_binop_dispatch(a, b);
        if set_override
            && let Some(result) = try_dispatch_binary_special(a, b, "__sub__", "__rsub__")?
        {
            return Ok(result);
        }
        let numeric_override = needs_numeric_binop_dispatch(a, b, "__sub__", "__rsub__");
        if numeric_override
            && let Some(result) = try_dispatch_binary_special(a, b, "__sub__", "__rsub__")?
        {
            return Ok(result);
        }
        if !numeric_override {
            if is_int_like(a) && is_int_like(b) {
                return int_sub(a, b);
            }
            if is_int_or_long(a) && is_int_or_long(b) {
                return long_sub(a, b);
            }
            if is_float_pair(a, b) {
                return float_sub(a, b);
            }
            if is_complex_pair(a, b) {
                return complex_sub(a, b);
            }
        }
        // set / frozenset difference — PyPy: setobject.py W_BaseSetObject.descr_sub.
        // descr_sub returns NotImplemented for a non-set rhs, so `-` requires
        // both operands to be sets (the `difference` method takes iterables).
        if !set_override
            && pyre_object::is_set_or_frozenset(a)
            && pyre_object::is_set_or_frozenset(b)
        {
            return crate::typedef::set_method_difference(&[a, b]);
        }
        if !numeric_override
            && let Some(result) = try_dispatch_binary_special(a, b, "__sub__", "__rsub__")?
        {
            return Ok(result);
        }
        let a_name = crate::baseobjspace::object_functionstr_type_name(a);
        let b_name = crate::baseobjspace::object_functionstr_type_name(b);
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for -: '{a_name}' and '{b_name}'"
        )))
    }
}

pub fn mul(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        let numeric_override = needs_numeric_binop_dispatch(a, b, "__mul__", "__rmul__");
        if numeric_override
            && let Some(result) = try_dispatch_binary_special(a, b, "__mul__", "__rmul__")?
        {
            return Ok(result);
        }
        if !numeric_override {
            if is_int_like(a) && is_int_like(b) {
                return int_mul(a, b);
            }
            if is_int_or_long(a) && is_int_or_long(b) {
                return long_mul(a, b);
            }
            if is_float_pair(a, b) {
                return float_mul(a, b);
            }
            if is_complex_pair(a, b) {
                return complex_mul(a, b);
            }
        }
        // The `sq_repeat` fast paths below are valid for exact builtin
        // sequences.  A sequence subclass that overrides `__mul__`/`__rmul__`
        // must reach its override first — `LM([1]) * 2` is `LM.__mul__`, not a
        // list repetition — the same gate the concat branches of `add` apply.
        const MUL_SPECIALS: &[&str] = &["__mul__", "__rmul__"];
        if (seq_repeat_override(a, MUL_SPECIALS) || seq_repeat_override(b, MUL_SPECIALS))
            && let Some(result) = try_dispatch_binary_special(a, b, "__mul__", "__rmul__")?
        {
            return Ok(result);
        }
        if is_str(a) && is_int_or_long(b) {
            return str_repeat(a, b);
        }
        if is_int_or_long(a) && is_str(b) {
            return str_repeat(b, a);
        }
        if is_list(a) && is_int_or_long(b) {
            return list_repeat(a, b);
        }
        if is_int_or_long(a) && is_list(b) {
            return list_repeat(b, a);
        }
        // tupleobject.py descr_mul
        if is_tuple(a) && is_int_or_long(b) {
            return tuple_repeat(a, b);
        }
        if is_int_or_long(a) && is_tuple(b) {
            return tuple_repeat(b, a);
        }
        // bytesobject.py descr_mul / bytearrayobject.py descr_mul
        if pyre_object::bytesobject::is_bytes_like(a) && is_int_or_long(b) {
            return bytes_repeat(a, b);
        }
        if is_int_or_long(a) && pyre_object::bytesobject::is_bytes_like(b) {
            return mul(b, a);
        }
        // `PyNumber_Multiply`: none of the builtin sequences implements
        // `nb_multiply`, so their `__mul__` / `__rmul__` slot wrappers take no
        // part in the operator dispatch — only the other operand can supply a
        // numeric implementation.
        let a_seq = is_repeat_sequence(a);
        let b_seq = is_repeat_sequence(b);
        let dispatched = match (a_seq, b_seq) {
            (true, true) => None,
            (true, false) => match lookup_type_special(b, "__rmul__") {
                Some(method) => try_call_special(method, &[b, a])?,
                None => None,
            },
            (false, true) => match lookup_type_special(a, "__mul__") {
                Some(method) => try_call_special(method, &[a, b])?,
                None => None,
            },
            (false, false) if !numeric_override => {
                try_dispatch_binary_special(a, b, "__mul__", "__rmul__")?
            }
            (false, false) => None,
        };
        if let Some(result) = dispatched {
            return Ok(result);
        }
        let a_name = crate::baseobjspace::object_functionstr_type_name(a);
        let b_name = crate::baseobjspace::object_functionstr_type_name(b);
        // `sequence_repeat`: the count goes through `__index__`, and an
        // operand that has none is reported by its own type — the sequence is
        // never the one named.
        if a_seq || b_seq {
            let (seq, other, other_name) = if a_seq {
                (a, b, b_name)
            } else {
                (b, a, a_name)
            };
            if !(a_seq && b_seq) && crate::baseobjspace::lookup(other, "__index__").is_some() {
                return sequence_repeat(seq, crate::baseobjspace::getindex_repeat(other)?);
            }
            return Err(PyError::type_error(format!(
                "can't multiply sequence by non-int of type '{other_name}'"
            )));
        }
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for *: '{a_name}' and '{b_name}'"
        )))
    }
}

pub fn floordiv(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        let numeric_override = needs_numeric_binop_dispatch(a, b, "__floordiv__", "__rfloordiv__");
        if numeric_override
            && let Some(result) =
                try_dispatch_binary_special(a, b, "__floordiv__", "__rfloordiv__")?
        {
            return Ok(result);
        }
        if !numeric_override {
            if is_int_like(a) && is_int_like(b) {
                return int_floordiv(a, b);
            }
            if is_int_or_long(a) && is_int_or_long(b) {
                return long_floordiv(a, b);
            }
            if is_float_pair(a, b) {
                return float_floordiv(a, b);
            }
        }
        if !numeric_override
            && let Some(result) =
                try_dispatch_binary_special(a, b, "__floordiv__", "__rfloordiv__")?
        {
            return Ok(result);
        }
        let a_name = crate::baseobjspace::object_functionstr_type_name(a);
        let b_name = crate::baseobjspace::object_functionstr_type_name(b);
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for //: '{a_name}' and '{b_name}'"
        )))
    }
}

pub fn mod_(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        let numeric_override = needs_numeric_binop_dispatch(a, b, "__mod__", "__rmod__");
        if numeric_override
            && let Some(result) = try_dispatch_binary_special(a, b, "__mod__", "__rmod__")?
        {
            return Ok(result);
        }
        if !numeric_override {
            if is_int_like(a) && is_int_like(b) {
                return int_mod(a, b);
            }
            if is_int_or_long(a) && is_int_or_long(b) {
                return long_mod(a, b);
            }
            if is_float_pair(a, b) {
                return float_mod(a, b);
            }
        }
        let is_str_lhs = is_str(a);
        let is_bytes_lhs = pyre_object::bytesobject::is_bytes_like(a);
        // str/bytes % args — reflected-subclass priority: a subclass on the
        // right overriding __rmod__ is tried before the built-in formatter.
        if is_str_lhs || is_bytes_lhs {
            if let Some((method, w_type)) =
                crate::baseobjspace::subclass_special_override(b, "__rmod__")
            {
                let priority = match (crate::typedef::r#type(a), crate::typedef::r#type(b)) {
                    (Some(at), Some(bt)) => at != bt && issubtype_cached(bt.as_ptr(), at.as_ptr()),
                    _ => false,
                };
                if priority {
                    match crate::baseobjspace::get_and_call_function(method, b, w_type, &[a]) {
                        Ok(result) if !is_not_implemented(result) => return Ok(result),
                        Ok(_) => {}
                        Err(e) => return Err(e),
                    }
                }
            }
            return if is_str_lhs {
                crate::objspace::std::formatting::str_format_percent(a, b)
            } else {
                crate::objspace::std::formatting::bytes_format_percent(a, b)
            };
        }
        if !numeric_override
            && let Some(result) = try_dispatch_binary_special(a, b, "__mod__", "__rmod__")?
        {
            return Ok(result);
        }
        let a_name = crate::baseobjspace::object_functionstr_type_name(a);
        let b_name = crate::baseobjspace::object_functionstr_type_name(b);
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for %: '{a_name}' and '{b_name}'"
        )))
    }
}

/// True division (`/` operator) — always produces a float result.
///
/// intobject.py:332-345 `_truediv` raises "division by zero" for int/int;
/// floatobject.py:519 `_floatdiv` raises "float division by zero" once
/// any operand is a float.
/// longobject.py:62-70 `_truediv` catches OverflowError from
/// `rbigint.truediv` and reissues it as
/// "integer division result too large for a float".
pub fn truediv(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        let numeric_override = needs_numeric_binop_dispatch(a, b, "__truediv__", "__rtruediv__");
        if numeric_override
            && let Some(result) = try_dispatch_binary_special(a, b, "__truediv__", "__rtruediv__")?
        {
            return Ok(result);
        }
        let a_num = is_int(a) || is_float(a) || is_long(a);
        let b_num = is_int(b) || is_float(b) || is_long(b);
        if !numeric_override && a_num && b_num {
            if is_float(a) || is_float(b) {
                return float_truediv(a, b);
            }
            if !is_long(b) && as_float(b) == 0.0 {
                return Err(PyError::zero_division(ZERO_DIVISION_MSG));
            }
            // intobject.py:332 `_truediv`: machine ints wider than the
            // binary64 mantissa deliberately overflow into the rbigint path
            // so division is rounded once, after exact integer arithmetic.
            let wide_int = (!is_long(a) && int_value(a).unsigned_abs() >> 53 != 0)
                || (!is_long(b) && int_value(b).unsigned_abs() >> 53 != 0);
            if is_long(a) || is_long(b) || wide_int {
                let a_owned;
                let va = if is_long(a) {
                    w_long_get_value(a)
                } else {
                    a_owned = BigInt::from(int_value(a));
                    &a_owned
                };
                let b_owned;
                let vb = if is_long(b) {
                    w_long_get_value(b)
                } else {
                    b_owned = BigInt::from(int_value(b));
                    &b_owned
                };
                let r = bigint_truediv(va, vb)?;
                return Ok(w_float_new(r));
            }
            return Ok(w_float_new(as_float(a) / as_float(b)));
        }
        if !numeric_override && is_complex_pair(a, b) {
            return complex_truediv(a, b);
        }
        if !numeric_override
            && let Some(result) = try_dispatch_binary_special(a, b, "__truediv__", "__rtruediv__")?
        {
            return Ok(result);
        }
        let a_name = crate::baseobjspace::object_functionstr_type_name(a);
        let b_name = crate::baseobjspace::object_functionstr_type_name(b);
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for /: '{a_name}' and '{b_name}'"
        )))
    }
}

/// `descroperation.py:425 pow_binary` — return `None` when neither numeric
/// fast paths nor `__pow__` / `__rpow__` produce a result.
fn pow_binary(a: PyObjectRef, b: PyObjectRef) -> Result<Option<PyObjectRef>, PyError> {
    unsafe {
        let numeric_override = needs_numeric_binop_dispatch(a, b, "__pow__", "__rpow__");
        if numeric_override {
            if let Some(result) = try_dispatch_binary_special(a, b, "__pow__", "__rpow__")? {
                return Ok(Some(result));
            }
            return Ok(None);
        }
        if is_int_like(a) && is_int_like(b) {
            return int_pow(a, b).map(Some);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_pow(a, b).map(Some);
        }
        if is_float_pair(a, b) {
            reject_pow_operand_overflow(a)?;
            reject_pow_operand_overflow(b)?;
            return float_pow_impl(as_float(a), as_float(b)).map(Some);
        }
        if is_complex_pair(a, b) {
            reject_pow_operand_overflow(a)?;
            reject_pow_operand_overflow(b)?;
            return complex_pow(a, b).map(Some);
        }
        try_dispatch_binary_special(a, b, "__pow__", "__rpow__")
    }
}

/// Power operation dispatch (`**` operator).
pub fn pow(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    if let Some(result) = pow_binary(a, b)? {
        return Ok(result);
    }
    let a_name = crate::baseobjspace::object_functionstr_type_name(a);
    let b_name = crate::baseobjspace::object_functionstr_type_name(b);
    Err(PyError::type_error(format!(
        "unsupported operand type(s) for ** or pow(): '{a_name}' and '{b_name}'"
    )))
}

/// `descroperation.py:486-499 inplace_pow` — unlike the generated in-place
/// binary operations, power has its own fallback error spelling.
pub fn inplace_pow(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    if let Some(result) = try_inplace_special(a, b, "__ipow__", None, false)? {
        return Ok(result);
    }
    if let Some(result) = pow_binary(a, b)? {
        return Ok(result);
    }
    Err(binary_builtin_type_error("**=", a, b))
}

// ── Numeric type-slot builtins ────────────────────────────────────────
//
// The `int`/`float` numeric special methods (`int.__add__`,
// `float.__and__`, …) resolve to these concrete computations, not to the
// operator dispatch above.  The operator (`add`, `and_`, …) drives the
// forward+reflected protocol and, when an operand is a numeric subclass
// that overrides the special method, re-dispatches through that operand's
// type slot.  Wiring the slot back to the operator would re-enter it and
// recurse without bound; wiring the slot to the concrete computation
// terminates after computing the result or returning NotImplemented for
// the reflected method to handle.  The `is_int_or_long`/`is_float` macro
// guards in typedef.rs pre-filter the operand kinds, so the trailing
// NotImplemented is reached only defensively.

pub(crate) fn add_builtin(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        if is_int_like(a) && is_int_like(b) {
            return int_add(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_add(a, b);
        }
        if is_float_pair(a, b) {
            return float_add(a, b);
        }
        if is_complex_pair(a, b) {
            return complex_add(a, b);
        }
        Ok(w_not_implemented())
    }
}

pub(crate) fn sub_builtin(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        if is_int_like(a) && is_int_like(b) {
            return int_sub(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_sub(a, b);
        }
        if is_float_pair(a, b) {
            return float_sub(a, b);
        }
        if is_complex_pair(a, b) {
            return complex_sub(a, b);
        }
        Ok(w_not_implemented())
    }
}

pub(crate) fn mul_builtin(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        if is_int_like(a) && is_int_like(b) {
            return int_mul(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_mul(a, b);
        }
        if is_float_pair(a, b) {
            return float_mul(a, b);
        }
        if is_complex_pair(a, b) {
            return complex_mul(a, b);
        }
        Ok(w_not_implemented())
    }
}

pub(crate) fn truediv_builtin(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        let a_num = is_int(a) || is_float(a) || is_long(a);
        let b_num = is_int(b) || is_float(b) || is_long(b);
        if a_num && b_num {
            if is_float(a) || is_float(b) {
                return float_truediv(a, b);
            }
            if !is_long(b) && as_float(b) == 0.0 {
                return Err(PyError::zero_division(ZERO_DIVISION_MSG));
            }
            // Match `_truediv`'s overflow-to-rbigint leg for i64 values that
            // are not exactly representable in a binary64 mantissa.
            let wide_int = (!is_long(a) && int_value(a).unsigned_abs() >> 53 != 0)
                || (!is_long(b) && int_value(b).unsigned_abs() >> 53 != 0);
            if is_long(a) || is_long(b) || wide_int {
                let a_owned;
                let va = if is_long(a) {
                    w_long_get_value(a)
                } else {
                    a_owned = BigInt::from(int_value(a));
                    &a_owned
                };
                let b_owned;
                let vb = if is_long(b) {
                    w_long_get_value(b)
                } else {
                    b_owned = BigInt::from(int_value(b));
                    &b_owned
                };
                let r = bigint_truediv(va, vb)?;
                return Ok(w_float_new(r));
            }
            return Ok(w_float_new(as_float(a) / as_float(b)));
        }
        if is_complex_pair(a, b) {
            return complex_truediv(a, b);
        }
        Ok(w_not_implemented())
    }
}

pub(crate) fn floordiv_builtin(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        if is_int_like(a) && is_int_like(b) {
            return int_floordiv(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_floordiv(a, b);
        }
        if is_float_pair(a, b) {
            return float_floordiv(a, b);
        }
        Ok(w_not_implemented())
    }
}

pub(crate) fn mod_builtin(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        if is_int_like(a) && is_int_like(b) {
            return int_mod(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_mod(a, b);
        }
        if is_float_pair(a, b) {
            return float_mod(a, b);
        }
        Ok(w_not_implemented())
    }
}

pub(crate) fn pow_builtin(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        if is_int_like(a) && is_int_like(b) {
            return int_pow(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_pow(a, b);
        }
        if is_float_pair(a, b) {
            reject_pow_operand_overflow(a)?;
            reject_pow_operand_overflow(b)?;
            return float_pow_impl(as_float(a), as_float(b));
        }
        if is_complex_pair(a, b) {
            reject_pow_operand_overflow(a)?;
            reject_pow_operand_overflow(b)?;
            return complex_pow(a, b);
        }
        Ok(w_not_implemented())
    }
}

pub(crate) fn divmod_builtin(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        let lhs_num = is_int(a) || is_long(a) || is_float(a);
        let rhs_num = is_int(b) || is_long(b) || is_float(b);
        if lhs_num && rhs_num {
            // Python 3.14 reports the builtin operation's uniform spelling for
            // every numeric zero divisor.  This intentionally differs from
            // PyPy 3.11's int/float-specific divmod and modulo messages.
            if !is_true(b)? {
                return Err(PyError::zero_division(ZERO_DIVISION_MSG));
            }
            if is_float_pair(a, b) {
                let x = as_float(a);
                reject_float_coercion_overflow(a, x)?;
                let y = as_float(b);
                reject_float_coercion_overflow(b, y)?;
                let (q, r) = float_divmod_w(x, y)?;
                return Ok(w_tuple_new(vec![w_float_new(q), w_float_new(r)]));
            }
            return integer_divmod_pair(a, b);
        }
    }
    Ok(w_not_implemented())
}

pub(crate) fn lshift_builtin(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        if is_int_like(a) && is_int_like(b) {
            return int_lshift(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_lshift(a, b);
        }
        Ok(w_not_implemented())
    }
}

pub(crate) fn rshift_builtin(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        if is_int_like(a) && is_int_like(b) {
            return int_rshift(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_rshift(a, b);
        }
        Ok(w_not_implemented())
    }
}

pub(crate) fn and_builtin(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        // int.__and__ — bool operands are treated as ints; the bool-typed
        // result is produced by bool.__and__ (init_bool_type), not here.
        if is_int(a) && is_int(b) {
            return int_bitand(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_bitand(a, b);
        }
        Ok(w_not_implemented())
    }
}

pub(crate) fn or_builtin(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        if is_int(a) && is_int(b) {
            return int_bitor(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_bitor(a, b);
        }
        Ok(w_not_implemented())
    }
}

pub(crate) fn xor_builtin(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        if is_int(a) && is_int(b) {
            return int_bitxor(a, b);
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            return long_bitxor(a, b);
        }
        Ok(w_not_implemented())
    }
}

// ── descroperation helpers — pypy/objspace/descroperation.py ──────────
//
// These helpers implement the standard "forward + reverse with
// NotImplemented fallback" dispatch that PyPy generates from
// `_make_binop_impl` / `_make_descr_unaryop`. They were originally in
// `baseobjspace` (not in `builtins`) because they are space-level
// semantics shared between the builtin module, the weakproxy wrappers,
// and any future opcode dispatch — every caller needs the same rule
// or NotImplemented from the forward path silently swallows the
// reflected operand.

/// `space.lookup(w_obj, dunder)` — descroperation.py.
pub(crate) unsafe fn lookup_type_special(obj: PyObjectRef, dunder: &str) -> Option<PyObjectRef> {
    crate::typedef::r#type(obj).and_then(|tp| lookup_in_type(tp.as_ptr(), dunder))
}

/// Call a raw type-MRO special-method descriptor and treat NotImplemented as
/// "no result", per descroperation.py `_invoke_binop` and
/// `_check_notimplemented`.  `args[0]` is the receiver; custom descriptors
/// must bind through `space.get_and_call_function` rather than receiving that
/// object as an explicitly injected argument.
pub(crate) fn try_call_special(
    method: PyObjectRef,
    args: &[PyObjectRef],
) -> Result<Option<PyObjectRef>, PyError> {
    debug_assert!(!args.is_empty());
    let receiver = args[0];
    let Some(w_type) = crate::typedef::r#type(receiver) else {
        return Ok(None);
    };
    let result = unsafe {
        crate::baseobjspace::get_and_call_function(method, receiver, w_type.as_ptr(), &args[1..])?
    };
    if unsafe { is_not_implemented(result) } {
        Ok(None)
    } else {
        Ok(Some(result))
    }
}

/// descroperation.py:648 `_call_binop_impl` — resolve the forward
/// (`dunder`) and reflected (`rdunder`) special methods through
/// `lookup_where`, decide whether to try the reflected operand first by
/// comparing the two defining classes, then invoke forward-then-reverse.
pub(crate) fn try_dispatch_binary_special(
    lhs: PyObjectRef,
    rhs: PyObjectRef,
    dunder: &str,
    rdunder: &str,
) -> Result<Option<PyObjectRef>, PyError> {
    // descroperation.py:687 `seq_bug_compat = (symbol == '+' or symbol == '*')`.
    let seq_bug_compat = dunder == "__add__" || dunder == "__mul__";
    unsafe {
        let Some(w_typ1) = crate::typedef::r#type(lhs) else {
            return Ok(None);
        };
        let Some(w_typ2) = crate::typedef::r#type(rhs) else {
            return Ok(None);
        };
        let (w_left_src, mut w_left_impl) =
            match lookup_where_with_method_cache(w_typ1.as_ptr(), dunder) {
                Some((src, imp)) => (Some(src), Some(imp)),
                None => (None, None),
            };
        let mut w_obj1 = lhs;
        let mut w_obj2 = rhs;
        let mut w_right_impl: Option<PyObjectRef> = None;
        // descroperation.py:652 — same type means the reflected method is
        // never considered.
        if w_typ1 != w_typ2 {
            let (w_right_src, wri) = match lookup_where_with_method_cache(w_typ2.as_ptr(), rdunder)
            {
                Some((src, imp)) => (Some(src), Some(imp)),
                None => (None, None),
            };
            w_right_impl = wri;
            // descroperation.py:662 — both `__op__` and `__rop__` are
            // found, in different MRO classes.
            if let (Some(rsrc), Some(lsrc)) = (w_right_src, w_left_src)
                && !std::ptr::eq(lsrc, rsrc)
            {
                // descroperation.py:667-670.
                let prefer_reverse = (seq_bug_compat
                    && crate::baseobjspace::flag_sequence_bug_compat(w_typ1.as_ptr())
                    && !crate::baseobjspace::flag_sequence_bug_compat(w_typ2.as_ptr()))
                    || issubtype_w(w_typ2.as_ptr(), w_typ1.as_ptr());
                // descroperation.py:671-672.
                if prefer_reverse
                    && !p_abstract_issubclass_w(lsrc, rsrc)?
                    && !p_abstract_issubclass_w(w_typ1.as_ptr(), rsrc)?
                {
                    std::mem::swap(&mut w_obj1, &mut w_obj2);
                    std::mem::swap(&mut w_left_impl, &mut w_right_impl);
                }
            }
        }
        // descroperation.py:676 — _invoke_binop(w_left_impl, w_obj1, w_obj2).
        if let Some(method) = w_left_impl
            && let Some(result) = try_call_special(method, &[w_obj1, w_obj2])?
        {
            return Ok(Some(result));
        }
        // descroperation.py:679 — _invoke_binop(w_right_impl, w_obj2, w_obj1).
        if let Some(method) = w_right_impl
            && let Some(result) = try_call_special(method, &[w_obj2, w_obj1])?
        {
            return Ok(Some(result));
        }
        Ok(None)
    }
}

/// CPython 3.14 `typeobject.c:slot_nb_power` — the three-argument copy of
/// `SLOT1BINFULL`.  Unlike PyPy 3.11's `space.pow`, Python 3.14 also offers
/// the exponent's reflected method the modulus argument.  A proper subtype's
/// distinct `__rpow__` is tried first, then the base's `__pow__`, then the
/// reflected method when it has not already been tried.
fn try_dispatch_ternary_pow_special(
    base: PyObjectRef,
    exp: PyObjectRef,
    modulus: PyObjectRef,
) -> Result<Option<PyObjectRef>, PyError> {
    unsafe {
        let Some(w_base_type) = crate::typedef::r#type(base) else {
            return Ok(None);
        };
        let Some(w_exp_type) = crate::typedef::r#type(exp) else {
            return Ok(None);
        };
        let (w_base_src, w_base_impl) =
            match lookup_where_with_method_cache(w_base_type.as_ptr(), "__pow__") {
                Some((src, imp)) => (Some(src), Some(imp)),
                None => (None, None),
            };
        let (w_exp_src, mut w_exp_impl) = if w_base_type != w_exp_type {
            match lookup_where_with_method_cache(w_exp_type.as_ptr(), "__rpow__") {
                Some((src, imp)) => (Some(src), Some(imp)),
                None => (None, None),
            }
        } else {
            (None, None)
        };

        // slot_nb_power: a proper exponent subtype whose reflected method is
        // distinct from the base type's definition gets the first chance.
        if issubtype_w(w_exp_type.as_ptr(), w_base_type.as_ptr())
            && let (Some(exp_src), Some(base_src), Some(exp_impl)) =
                (w_exp_src, w_base_src, w_exp_impl)
            && !std::ptr::eq(base_src, exp_src)
            && !p_abstract_issubclass_w(base_src, exp_src)?
            && !p_abstract_issubclass_w(w_base_type.as_ptr(), exp_src)?
        {
            if let Some(result) = try_call_special(exp_impl, &[exp, base, modulus])? {
                return Ok(Some(result));
            }
            // CPython clears `do_other` after the subtype-first call,
            // including when it returns NotImplemented.
            w_exp_impl = None;
        }

        if let Some(method) = w_base_impl
            && let Some(result) = try_call_special(method, &[base, exp, modulus])?
        {
            return Ok(Some(result));
        }
        if let Some(method) = w_exp_impl
            && let Some(result) = try_call_special(method, &[exp, base, modulus])?
        {
            return Ok(Some(result));
        }
        Ok(None)
    }
}

/// descroperation.py:825 `inplace_impl` — try the in-place special
/// (`__iadd__` etc.) on the lhs.  Returns `None` when the type has no
/// such method or the call yields `NotImplemented`, so the caller falls
/// back to the corresponding binary operation.
pub(crate) fn try_inplace_special(
    lhs: PyObjectRef,
    rhs: PyObjectRef,
    idunder: &str,
    rdunder: Option<&str>,
    seq_bug_compat: bool,
) -> Result<Option<PyObjectRef>, PyError> {
    // An exact builtin numeric lhs (int/long/float/complex/bool) defines no
    // in-place special, so the lookup below is always a miss.  Skip it — the
    // string-keyed method-cache probe (a UTF-8 dunder-name intern) is the
    // dominant per-iteration cost of `total += …`.  Only exact numerics are
    // gated: builtin sequences (list/bytearray) define `__iadd__`/`__imul__`,
    // and a subclass may override the in-place slot, so both must fall
    // through to the lookup.
    if unsafe { is_exact_numeric_builtin(lhs) } {
        return Ok(None);
    }
    // descroperation.py:826 — only when the lhs in-place method exists.
    if let Some(method) = unsafe { lookup_type_special(lhs, idunder) } {
        // descroperation.py:831 seq_bug_compat — for `+=` / `*=` where the
        // lhs is a builtin sequence and the rhs is not, try the rhs
        // reflected method before the lhs in-place method.
        if seq_bug_compat
            && let Some(rd) = rdunder
            && let (Some(lhs_type), Some(rhs_type)) =
                (crate::typedef::r#type(lhs), crate::typedef::r#type(rhs))
            && crate::baseobjspace::flag_sequence_bug_compat(lhs_type.as_ptr())
            && !crate::baseobjspace::flag_sequence_bug_compat(rhs_type.as_ptr())
            && let Some(rmethod) = unsafe { lookup_type_special(rhs, rd) }
            && let Some(result) = try_call_special(rmethod, &[rhs, lhs])?
        {
            return Ok(Some(result));
        }
        if let Some(result) = try_call_special(method, &[lhs, rhs])? {
            return Ok(Some(result));
        }
    }
    Ok(None)
}

/// `(int|long) ** (int|long) % (int|long)` fast path used by `space.pow`
/// when a modulus is supplied — longobject.py `int_pow`.
pub(crate) fn try_int_long_pow_with_modulo(
    base: PyObjectRef,
    exp: PyObjectRef,
    modulus: PyObjectRef,
) -> Result<Option<PyObjectRef>, PyError> {
    unsafe {
        if !is_int_or_long(base) || !is_int_or_long(exp) || !is_int_or_long(modulus) {
            return Ok(None);
        }

        // `descr_pow` wraps its 3-arg result as `W_LongObject` whenever the
        // receiver is a long; the int path (`W_IntObject.descr_pow`) reaches
        // that same long path via `_pow_ovf2long` when the exponent or modulus
        // is a long. So the result stays a long unless all three operands are
        // machine ints, in which case `space.newint` demotes it.
        let all_int_like = is_int_like(base) && is_int_like(exp) && is_int_like(modulus);

        let base_owned;
        let base = if is_long(base) {
            w_long_get_value(base)
        } else {
            base_owned = BigInt::from(int_value(base));
            &base_owned
        };
        let exp_owned;
        let exp = if is_long(exp) {
            w_long_get_value(exp)
        } else {
            exp_owned = BigInt::from(int_value(exp));
            &exp_owned
        };
        let modulus_owned;
        let modulus = if is_long(modulus) {
            w_long_get_value(modulus)
        } else {
            modulus_owned = BigInt::from(int_value(modulus));
            &modulus_owned
        };

        if modulus.get_sign() == 0 {
            return Err(PyError::value_error("pow() 3rd argument cannot be 0"));
        }
        if exp.get_sign() < 0 {
            // 3-arg pow with a negative exponent: raise the modular
            // inverse of `base` to `-exp` (longobject.c long_pow →
            // long_invmod).  The inverse exists only when `base` is
            // coprime to the modulus.
            let negative_modulus = modulus.get_sign() < 0;
            let abs_modulus_owned;
            let abs_modulus = if negative_modulus {
                abs_modulus_owned = modulus.neg();
                &abs_modulus_owned
            } else {
                modulus
            };
            let inverse = bigint_mod_inverse(base, abs_modulus)?;
            let pos_exp = exp.neg();
            let mut result = inverse
                .pow(&pos_exp, Some(abs_modulus))
                .map_err(|_| PyError::memory_error("exponent too large"))?;
            if negative_modulus && result.get_sign() > 0 {
                result = result.sub(abs_modulus);
            }
            return Ok(Some(pow_mod_result(result, all_int_like)));
        }
        if exp.get_sign() == 0 {
            // `x ** 0 % m` is `1 % m` under floor semantics, so a negative
            // modulus yields a negative residue (`pow(2, 0, -13) == -12`).
            return Ok(Some(pow_mod_result(
                BigInt::one()
                    .r#mod(modulus)
                    .expect("modulus was checked nonzero"),
                all_int_like,
            )));
        }

        let negative_modulus = modulus.get_sign() < 0;
        let abs_modulus_owned;
        let abs_modulus = if negative_modulus {
            abs_modulus_owned = modulus.neg();
            &abs_modulus_owned
        } else {
            modulus
        };
        let mut result = base
            .pow(exp, Some(abs_modulus))
            .map_err(|_| PyError::memory_error("exponent too large"))?;
        if negative_modulus && result.get_sign() > 0 {
            result = result.sub(abs_modulus);
        }
        Ok(Some(pow_mod_result(result, all_int_like)))
    }
}

/// Box a 3-arg `pow` result: `space.newint` (demote) when every operand was a
/// machine int, else `W_LongObject` (a long receiver keeps the long
/// representation). The demote arm always fits — `result < |modulus|`.
fn pow_mod_result(value: BigInt, all_int_like: bool) -> PyObjectRef {
    if all_int_like {
        w_int_new(jit_bigint_to_i64_value(&value))
    } else {
        w_long_new(value)
    }
}

/// `%T` — the runtime type name of `obj` (`space.type(obj).name`), falling
/// back to the layout-level type for objects without a Python type.
fn operand_type_name(obj: PyObjectRef) -> String {
    unsafe {
        match crate::typedef::r#type(obj) {
            Some(tp) => pyre_object::w_type_get_name(tp.as_ptr()).to_string(),
            None => (*ll_type(obj)).name.to_string(),
        }
    }
}

pub(crate) fn binary_builtin_type_error(
    opname: &str,
    lhs: PyObjectRef,
    rhs: PyObjectRef,
) -> PyError {
    let lhs_name = operand_type_name(lhs);
    let rhs_name = operand_type_name(rhs);
    PyError::type_error(format!(
        "unsupported operand type(s) for {opname}: '{lhs_name}' and '{rhs_name}'"
    ))
}

/// The three-operand form of [`binary_builtin_type_error`] for `pow(a, b, c)`
/// — descroperation.py:469 `unsupported operand type(s) for pow(): T, T, T`.
pub(crate) fn ternary_builtin_type_error(
    opname: &str,
    a: PyObjectRef,
    b: PyObjectRef,
    c: PyObjectRef,
) -> PyError {
    let a_name = operand_type_name(a);
    let b_name = operand_type_name(b);
    let c_name = operand_type_name(c);
    PyError::type_error(format!(
        "unsupported operand type(s) for {opname}: '{a_name}', '{b_name}', '{c_name}'"
    ))
}

/// 3-arg `pow(a, b, c)` dispatch. PyPy 3.11's
/// `pypy/objspace/descroperation.py:441` tries only the base's `__pow__`;
/// Python 3.14 changed this to offer the exponent's `__rpow__` the modulus
/// too (`Objects/typeobject.c:slot_nb_power`).
pub fn pow3(base: PyObjectRef, exp: PyObjectRef, modulus: PyObjectRef) -> PyResult {
    if unsafe { is_none(modulus) } {
        return pow(base, exp);
    }
    if let Some(result) = try_dispatch_ternary_pow_special(base, exp, modulus)? {
        return Ok(result);
    }
    Err(ternary_builtin_type_error(
        "** or pow()",
        base,
        exp,
        modulus,
    ))
}

/// `divmod(a, b)` dispatch — pypy/interpreter/baseobjspace.py
/// `('divmod', 'divmod', 2, ['__divmod__', '__rdivmod__'])`. Numeric
/// fast path then forward + reverse special-method dispatch with the
/// standard NotImplemented fallback.
pub fn divmod(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    let numeric_override;
    unsafe {
        numeric_override = needs_numeric_binop_dispatch(a, b, "__divmod__", "__rdivmod__");
        if numeric_override
            && let Some(result) = try_dispatch_binary_special(a, b, "__divmod__", "__rdivmod__")?
        {
            return Ok(result);
        }
        let lhs_num = is_int(a) || is_long(a) || is_float(a);
        let rhs_num = is_int(b) || is_long(b) || is_float(b);
        if !numeric_override && lhs_num && rhs_num {
            // Python 3.14 target-version spelling; see `divmod_builtin` above
            // for the PyPy 3.11 difference.
            if !is_true(b)? {
                return Err(PyError::zero_division(ZERO_DIVISION_MSG));
            }
            if is_float_pair(a, b) {
                let x = as_float(a);
                reject_float_coercion_overflow(a, x)?;
                let y = as_float(b);
                reject_float_coercion_overflow(b, y)?;
                let (q, r) = float_divmod_w(x, y)?;
                return Ok(w_tuple_new(vec![w_float_new(q), w_float_new(r)]));
            }
            return integer_divmod_pair(a, b);
        }
    }
    if !numeric_override
        && let Some(result) = try_dispatch_binary_special(a, b, "__divmod__", "__rdivmod__")?
    {
        return Ok(result);
    }
    Err(binary_builtin_type_error("divmod()", a, b))
}

/// floatobject.py:862 `PowDomainError` — sentinel for a negative base raised to
/// a fractional power, which `descr_pow` promotes to a complex result.
enum FloatPowError {
    Domain,
    ZeroDivision,
    Overflow,
}

/// Float scalar helper for the rtyper residual surface. Mirrors RPython's
/// lltype-level math helper pattern (`rpython/rtyper/lltypesystem/module/
/// ll_math.py`) while keeping Rust's `f64::abs` intrinsic ABI out of the
/// two-phase rtyper.
#[majit_macros::dont_look_inside]
pub fn jit_float_abs(v: f64) -> f64 {
    v.abs()
}

/// `rpython.rlib.rfloat.rfloat` / `math.fmod` scalar residual.  PyPy's float
/// modulo, divmod, and power code call `math_fmod` directly; Rust's `%`
/// spelling would instead expose a `mod(SomeFloat, SomeFloat)` operation that
/// RPython's annotator intentionally does not define.
#[majit_macros::dont_look_inside]
pub fn jit_float_fmod(x: f64, y: f64) -> f64 {
    x % y
}

/// `float_pow`: libm sets `ERANGE` when a finite base produces an
/// out-of-range result.  An infinite base is excluded because `pow(±inf, y)`
/// is answered by the special cases above rather than by libm, so its infinity
/// is the exact result and not a range error.
fn float_pow_range_check(z: f64, base: f64) -> Result<f64, FloatPowError> {
    if z.is_infinite() && !base.is_infinite() {
        return Err(FloatPowError::Overflow);
    }
    Ok(z)
}

/// 3.14 surfaces the `ERANGE` libm sets for `float_pow` through
/// `PyErr_SetFromErrno` as the `(errno, strerror)` pair.
/// `floatobject.py:937-943` instead lets its own `math.pow` OverflowError
/// through as the message `"float power"`.
fn float_pow_overflow_error() -> PyError {
    // 34 on every platform pyre targets; spelled out rather than taken from
    // `libc`, which does not export the errno constants for `wasm32`.
    const ERANGE: i32 = 34;
    PyError::errno_pair(
        crate::PyErrorKind::OverflowError,
        pyre_object::interp_exceptions::ExcKind::OverflowError,
        ERANGE,
    )
}

/// floatobject.py:865 `_pow`.
fn float_pow_inner(x: f64, y: f64) -> Result<f64, FloatPowError> {
    // floatobject.py:800-801
    if y == 2.0 {
        return float_pow_range_check(x * x, x);
    }
    // floatobject.py:803-804
    if y == 0.0 {
        return Ok(1.0);
    }
    // floatobject.py:806-807
    if x.is_nan() {
        return Ok(x);
    }
    // floatobject.py:809-814
    if y.is_nan() {
        return Ok(if x == 1.0 { 1.0 } else { y });
    }
    // floatobject.py:815-827
    if y.is_infinite() {
        let ax = jit_float_abs(x);
        if ax == 1.0 {
            return Ok(1.0);
        }
        return Ok(if (y > 0.0) == (ax > 1.0) {
            f64::INFINITY
        } else {
            0.0
        });
    }
    // floatobject.py:828-842
    if x.is_infinite() {
        let y_is_odd = jit_float_fmod(jit_float_abs(y), 2.0) == 1.0;
        return Ok(if y > 0.0 {
            if y_is_odd { x } else { jit_float_abs(x) }
        } else if y_is_odd {
            float_copysign(0.0, x)
        } else {
            0.0
        });
    }
    // floatobject.py:844-847
    if x == 0.0 && y < 0.0 {
        return Err(FloatPowError::ZeroDivision);
    }
    // floatobject.py:849-862
    let mut negate_result = false;
    let mut bx = x;
    if bx < 0.0 {
        if y.floor() != y {
            return Err(FloatPowError::Domain);
        }
        bx = -bx;
        negate_result = jit_float_fmod(jit_float_abs(y), 2.0) == 1.0;
    }
    // floatobject.py:864-869
    if bx == 1.0 {
        return Ok(if negate_result { -1.0 } else { 1.0 });
    }
    // floatobject.py:871-877
    let z = float_pow_range_check(bx.powf(y), bx)?;
    // floatobject.py:879-881
    Ok(if negate_result { -z } else { z })
}

/// Raw `x ** y` as an `f64` for the `int`/`long` negative-exponent path. The
/// negative-base fractional case cannot arise with an integral exponent, but is
/// mapped back to the ValueError that `_pow` raises via `PowDomainError`.
pub fn float_pow_raw(x: f64, y: f64) -> Result<f64, PyError> {
    match float_pow_inner(x, y) {
        Ok(z) => Ok(z),
        Err(FloatPowError::Domain) => Err(PyError::value_error(
            "negative number cannot be raised to a fractional power",
        )),
        Err(FloatPowError::ZeroDivision) => Err(PyError::zero_division("zero to a negative power")),
        Err(FloatPowError::Overflow) => Err(float_pow_overflow_error()),
    }
}

/// floatobject.py:584 `W_FloatObject.descr_pow`.
fn float_pow_impl(x: f64, y: f64) -> PyResult {
    match float_pow_inner(x, y) {
        Ok(z) => Ok(w_float_new(z)),
        // Negative numbers raised to fractional powers become complex.
        Err(FloatPowError::Domain) => unsafe {
            complex_pow(w_complex_new(x, 0.0), w_complex_new(y, 0.0))
        },
        Err(FloatPowError::ZeroDivision) => Err(PyError::zero_division("zero to a negative power")),
        Err(FloatPowError::Overflow) => Err(float_pow_overflow_error()),
    }
}

/// Left shift dispatch (`<<` operator).

pub fn lshift(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        let numeric_override = needs_numeric_binop_dispatch(a, b, "__lshift__", "__rlshift__");
        if numeric_override
            && let Some(result) = try_dispatch_binary_special(a, b, "__lshift__", "__rlshift__")?
        {
            return Ok(result);
        }
        if !numeric_override {
            if is_int_like(a) && is_int_like(b) {
                return int_lshift(a, b);
            }
            if is_int_or_long(a) && is_int_or_long(b) {
                return long_lshift(a, b);
            }
        }
        if !numeric_override
            && let Some(result) = try_dispatch_binary_special(a, b, "__lshift__", "__rlshift__")?
        {
            return Ok(result);
        }
        let a_name = crate::baseobjspace::object_functionstr_type_name(a);
        let b_name = crate::baseobjspace::object_functionstr_type_name(b);
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for <<: '{a_name}' and '{b_name}'"
        )))
    }
}

/// Right shift dispatch (`>>` operator).

pub fn rshift(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        let numeric_override = needs_numeric_binop_dispatch(a, b, "__rshift__", "__rrshift__");
        if numeric_override
            && let Some(result) = try_dispatch_binary_special(a, b, "__rshift__", "__rrshift__")?
        {
            return Ok(result);
        }
        if !numeric_override {
            if is_int_like(a) && is_int_like(b) {
                return int_rshift(a, b);
            }
            if is_int_or_long(a) && is_int_or_long(b) {
                return long_rshift(a, b);
            }
        }
        if !numeric_override
            && let Some(result) = try_dispatch_binary_special(a, b, "__rshift__", "__rrshift__")?
        {
            return Ok(result);
        }
        let a_name = crate::baseobjspace::object_functionstr_type_name(a);
        let b_name = crate::baseobjspace::object_functionstr_type_name(b);
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for >>: '{a_name}' and '{b_name}'"
        )))
    }
}

/// Bitwise AND dispatch (`&` operator).

pub fn and_(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        let set_override = needs_set_binop_dispatch(a, b);
        if set_override
            && let Some(result) = try_dispatch_binary_special(a, b, "__and__", "__rand__")?
        {
            return Ok(result);
        }
        let numeric_override = needs_numeric_binop_dispatch(a, b, "__and__", "__rand__");
        if numeric_override
            && let Some(result) = try_dispatch_binary_special(a, b, "__and__", "__rand__")?
        {
            return Ok(result);
        }
        // boolobject.py:74 W_BoolObject.descr_and — both operands bool
        // → space.newbool(op(a, b)). MRO ensures this runs before the
        // W_IntObject.descr_and fallback in int_bitand.
        if !numeric_override {
            if is_bool(a) && is_bool(b) {
                return Ok(pyre_object::bool_descr_and(a, b));
            }
            if is_int(a) && is_int(b) {
                return int_bitand(a, b);
            }
            if is_int_or_long(a) && is_int_or_long(b) {
                return long_bitand(a, b);
            }
        }
        // set / frozenset intersection — PyPy: setobject.py W_BaseSetObject.descr_and.
        // descr_and returns NotImplemented for a non-set rhs, so `&` requires
        // both operands to be sets (the `intersection` method takes iterables).
        if !set_override
            && pyre_object::is_set_or_frozenset(a)
            && pyre_object::is_set_or_frozenset(b)
        {
            return crate::typedef::set_method_intersection(&[a, b]);
        }
        if !numeric_override
            && let Some(result) = try_dispatch_binary_special(a, b, "__and__", "__rand__")?
        {
            return Ok(result);
        }
        let a_name = crate::baseobjspace::object_functionstr_type_name(a);
        let b_name = crate::baseobjspace::object_functionstr_type_name(b);
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for &: '{a_name}' and '{b_name}'"
        )))
    }
}

/// Check if an object can participate in `X | Y` union syntax.
///
/// PyPy equivalent: _unionable() in _pypy_generic_alias.py
#[inline]
pub(crate) fn unionable(obj: PyObjectRef) -> bool {
    unsafe {
        is_none(obj)
            || is_type(obj)
            || pyre_object::is_union(obj)
            || pyre_object::is_generic_alias(obj)
    }
}

/// Bitwise OR dispatch (`|` operator).

pub fn or_(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    // `pypy/objspace/std/dictproxyobject.py:51 descr_or` /
    // `pypy/objspace/std/dictproxyobject.py:60 descr_ror` —
    // mappingproxy `|` dispatches by copying the proxy's wrapped
    // mapping then `update`-ing with the other operand.  Pre-unwrap
    // each side so the dict-arm below sees plain dicts and produces
    // the same merge result.  The proxy-on-rhs case mirrors
    // `descr_ror` (proxy wraps the rhs operand inside `__or__`).
    let a = unsafe {
        if pyre_object::is_dict_proxy(a) {
            pyre_object::w_dict_proxy_get_mapping(a)
        } else {
            a
        }
    };
    let b = unsafe {
        if pyre_object::is_dict_proxy(b) {
            pyre_object::w_dict_proxy_get_mapping(b)
        } else {
            b
        }
    };
    unsafe {
        let set_override = needs_set_binop_dispatch(a, b);
        let numeric = needs_numeric_binop_dispatch(a, b, "__or__", "__ror__");
        if set_override
            && let Some(result) = try_dispatch_binary_special(a, b, "__or__", "__ror__")?
        {
            return Ok(result);
        }
        if numeric && let Some(result) = try_dispatch_binary_special(a, b, "__or__", "__ror__")? {
            return Ok(result);
        }
        // boolobject.py:75 W_BoolObject.descr_or — both operands bool
        // → space.newbool(op(a, b)).
        if !numeric {
            if is_bool(a) && is_bool(b) {
                return Ok(pyre_object::bool_descr_or(a, b));
            }
            if is_int(a) && is_int(b) {
                return int_bitor(a, b);
            }
            if is_int_or_long(a) && is_int_or_long(b) {
                return long_bitor(a, b);
            }
        }
        // set / frozenset union — PyPy: setobject.py W_BaseSetObject.descr_or.
        // descr_or returns NotImplemented unless w_other is a set/frozenset,
        // so the binary `|` operator requires both operands to be sets
        // (the `union` method accepts arbitrary iterables, descr_or does not).
        if !set_override
            && pyre_object::is_set_or_frozenset(a)
            && pyre_object::is_set_or_frozenset(b)
        {
            return crate::typedef::set_method_union(&[a, b]);
        }
        // dict | dict — PEP 584 merge. PyPy: dictmultiobject.py descr_or.
        // Returns a new dict built from `a`'s items, then updated with `b`'s.
        if pyre_object::is_dict(a) && pyre_object::is_dict(b) {
            let new_dict = pyre_object::w_dict_new();
            for (k, v) in pyre_object::w_dict_items(a) {
                pyre_object::w_dict_store(new_dict, k, v);
            }
            for (k, v) in pyre_object::w_dict_items(b) {
                pyre_object::w_dict_store(new_dict, k, v);
            }
            return Ok(new_dict);
        }
        // user-class + typedef (dict_view, …) dispatch: forward __or__ then reflected
        // __ror__, exactly once.  Skipped when a gated special above already ran, so a
        // set-/numeric-subclass override is never re-invoked.
        if !set_override
            && !numeric
            && let Some(result) = try_dispatch_binary_special(a, b, "__or__", "__ror__")?
        {
            return Ok(result);
        }
        // type | type — PEP 604 union types (Python 3.10+)
        // PyPy: typeobject.py descr_or → _pypy_generic_alias._create_union,
        // which collapses identical operands (`int | int` is `int`).
        // `None | type` reaches the type's reflected union descriptor, but
        // `None | None` has no type / UnionType / GenericAlias descriptor on
        // either side and must remain an unsupported bitwise operation.
        if unionable(a) && unionable(b) && !(is_none(a) && is_none(b)) {
            return crate::_pypy_generic_alias::create_union(a, b);
        }
        let a_name = crate::baseobjspace::object_functionstr_type_name(a);
        let b_name = crate::baseobjspace::object_functionstr_type_name(b);
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for |: '{a_name}' and '{b_name}'"
        )))
    }
}

/// Bitwise XOR dispatch (`^` operator).

pub fn xor(a: PyObjectRef, b: PyObjectRef) -> PyResult {
    unsafe {
        let set_override = needs_set_binop_dispatch(a, b);
        if set_override
            && let Some(result) = try_dispatch_binary_special(a, b, "__xor__", "__rxor__")?
        {
            return Ok(result);
        }
        let numeric_override = needs_numeric_binop_dispatch(a, b, "__xor__", "__rxor__");
        if numeric_override
            && let Some(result) = try_dispatch_binary_special(a, b, "__xor__", "__rxor__")?
        {
            return Ok(result);
        }
        if !numeric_override {
            if is_bool(a) && is_bool(b) {
                return Ok(pyre_object::bool_descr_xor(a, b));
            }
            if is_int(a) && is_int(b) {
                return int_bitxor(a, b);
            }
            if is_int_or_long(a) && is_int_or_long(b) {
                return long_bitxor(a, b);
            }
        }
        // set / frozenset symmetric difference — `pypy/objspace/std/
        // setobject.py W_BaseSetObject.descr_xor`.  Mirrors `and_`'s
        // intersection arm: walk both sides, keep elements present in
        // exactly one set.  Result type follows the left operand
        // (frozenset stays frozenset).
        if !set_override
            && pyre_object::is_set_or_frozenset(a)
            && pyre_object::is_set_or_frozenset(b)
        {
            return crate::typedef::set_method_symmetric_difference(&[a, b]);
        }
        if !numeric_override
            && let Some(result) = try_dispatch_binary_special(a, b, "__xor__", "__rxor__")?
        {
            return Ok(result);
        }
        let a_name = crate::baseobjspace::object_functionstr_type_name(a);
        let b_name = crate::baseobjspace::object_functionstr_type_name(b);
        Err(PyError::type_error(format!(
            "unsupported operand type(s) for ^: '{a_name}' and '{b_name}'"
        )))
    }
}

/// Comparison operation dispatch.

pub fn compare(a: PyObjectRef, b: PyObjectRef, op: CompareOp) -> PyResult {
    // RPython inserts a stack check on this recursive object-space call.
    // Container comparisons recurse without pushing a Python frame (for
    // example two distinct self-referential lists), so keep the same guard
    // explicitly in the Rust port and raise RecursionError before exhausting
    // the native stack.
    crate::stack_check::stack_check()?;
    // A builtin subclass overriding the comparison dunder dispatches the
    // override first (with reflected-subclass priority); exact builtins and
    // non-overriding subclasses fall through to the by-layout comparison slot,
    // which gives the inherited builtin comparison.
    unsafe {
        if let Some(result) = try_compare_override(a, b, op)? {
            return Ok(result);
        }
        // PyPy `descroperation.py:_make_comparison_impl` swaps the operands
        // whenever the right-hand type is a proper subtype, before invoking
        // the inherited comparison implementation.  `try_compare_override`
        // already performs that ordering for generic instances and genuine
        // builtin overrides; this is the corresponding path for a builtin
        // subclass which inherits (rather than overrides) its comparison
        // slot, such as unittest.mock's `_CallList(list)`.
        // Keep generic instances out of this slot-only path: PyPy preserves
        // `w_orig_obj1` / `w_orig_obj2` when subtype-first methods all return
        // NotImplemented, so the final ordering TypeError still names the
        // original operator and operand order.
        if !is_instance(a)
            && !is_instance(b)
            && let (Some(a_type), Some(b_type)) =
                (crate::typedef::r#type(a), crate::typedef::r#type(b))
        {
            if a_type != b_type && issubtype_cached(b_type.as_ptr(), a_type.as_ptr()) {
                return compare_slot(b, a, reverse_compare_op(op));
            }
        }
    }
    compare_slot(a, b, op)
}

/// The builtin comparison slot body: rich-comparison dispatch by concrete
/// layout.  Reached from the operator [`compare`] for exact builtins and
/// non-overriding subclasses, and bound by the `cmp_dunder!` slots so a
/// subclass override's `super().__eq__` (etc.) resolves to the inherited
/// builtin comparison instead of re-entering override dispatch (which would
/// recurse).
pub fn compare_slot(a: PyObjectRef, b: PyObjectRef, op: CompareOp) -> PyResult {
    unsafe {
        if is_int_like(a) && is_int_like(b) {
            return match op {
                CompareOp::Lt => int_lt(a, b),
                CompareOp::Le => int_le(a, b),
                CompareOp::Gt => int_gt(a, b),
                CompareOp::Ge => int_ge(a, b),
                CompareOp::Eq => int_eq(a, b),
                CompareOp::Ne => int_ne(a, b),
            };
        }
        // longobject.py `_make_descr_cmp` and intobject.py
        // `_make_descr_cmp`: both mixed orders call an rbigint.int_* method
        // on the long payload. The int-left order uses the reversed relation.
        // W_BoolObject shares W_IntObject storage upstream; pyre's distinct
        // bool layout requires the paired projections below.
        if is_long(a) && is_bool(b) {
            return Ok(w_bool_from(long_int_compare(
                a,
                w_bool_get_value(b) as i64,
                op,
            )));
        }
        if is_long(a) && is_int(b) {
            return Ok(w_bool_from(long_int_compare(a, w_int_get_value(b), op)));
        }
        if is_bool(a) && is_long(b) {
            return Ok(w_bool_from(long_int_compare(
                b,
                w_bool_get_value(a) as i64,
                reverse_compare_op(op),
            )));
        }
        if is_int(a) && is_long(b) {
            return Ok(w_bool_from(long_int_compare(
                b,
                w_int_get_value(a),
                reverse_compare_op(op),
            )));
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            // All machine/mixed integer pairs returned above, so this is the
            // longobject.py `_make_descr_cmp` long/long arm. `asbigint()`
            // returns each W_LongObject's immutable payload upstream; borrow
            // those payloads rather than allocating two translated clones.
            debug_assert!(is_long(a) && is_long(b));
            let va = w_long_get_value(a);
            let vb = w_long_get_value(b);
            return Ok(w_bool_from(match op {
                CompareOp::Lt => va.lt(vb),
                CompareOp::Le => va.le(vb),
                CompareOp::Gt => va.gt(vb),
                CompareOp::Ge => va.ge(vb),
                CompareOp::Eq => va.eq(vb),
                CompareOp::Ne => va.ne(vb),
            }));
        }
        if is_float_pair(a, b) {
            // `_compare` is a method on the float operand; the int-left order
            // reaches it through the reflected comparison, with the relation
            // reversed.
            return Ok(w_bool_from(if is_float(a) {
                float_compare(a, b, op)
            } else {
                float_compare(b, a, reverse_compare_op(op))
            }));
        }
        // complexobject.py only implements equality here.  Its ordering
        // dunders return NotImplemented, which must continue through the
        // reflected comparison and the generic TypeError fallback below.
        if is_complex_pair(a, b) && matches!(op, CompareOp::Eq | CompareOp::Ne) {
            return complex_richcompare(a, b, op);
        }
        if is_str(a) && is_str(b) {
            // Compare the WTF-8 bytes: for surrogate-free strings this is the
            // UTF-8 byte order (= code point order), and WTF-8 keeps lone
            // surrogates in code-point order too, so a surrogate-bearing
            // string compares correctly without going through
            // `w_str_get_value`.
            let sa = w_str_get_wtf8(a).as_bytes();
            let sb = w_str_get_wtf8(b).as_bytes();
            return Ok(w_bool_from(match op {
                CompareOp::Lt => sa < sb,
                CompareOp::Le => sa <= sb,
                CompareOp::Gt => sa > sb,
                CompareOp::Ge => sa >= sb,
                CompareOp::Eq => sa == sb,
                CompareOp::Ne => sa != sb,
            }));
        }
        // bytesobject.py W_BytesObject.descr_eq / _lt / ... and the
        // bytearray counterparts — lexicographic comparison on the raw
        // bytes.  bytes and bytearray compare by content
        // (b"a" == bytearray(b"a")), so both operands route through
        // bytes_like_data.
        if pyre_object::bytesobject::is_bytes_like(a) && pyre_object::bytesobject::is_bytes_like(b)
        {
            let da = pyre_object::bytesobject::bytes_like_data(a);
            let db = pyre_object::bytesobject::bytes_like_data(b);
            return Ok(w_bool_from(ordering_satisfies(da.cmp(db), op)));
        }
        // Tuple lexicographic comparison — PyPy: tupleobject.py descr_lt / _eq / etc.
        if is_tuple(a) && is_tuple(b) {
            let la = w_tuple_len(a);
            let lb = w_tuple_len(b);
            if matches!(op, CompareOp::Eq | CompareOp::Ne)
                && let Some(equal) = specialised_tuple_same_class_eq(a, b)?
            {
                return Ok(w_bool_from(if matches!(op, CompareOp::Ne) {
                    !equal
                } else {
                    equal
                }));
            }
            let min_len = la.min(lb);
            for i in 0..min_len {
                let ea = w_tuple_getitem(a, i as i64).unwrap_or(PY_NULL);
                let eb = w_tuple_getitem(b, i as i64).unwrap_or(PY_NULL);
                // tupleobject.py:137 `if not space.eq_w(items1[p], items2[p]):
                //     return getattr(space, name)(items1[p], items2[p])`
                if !crate::baseobjspace::eq_w(ea, eb)? {
                    return compare(ea, eb, op);
                }
            }
            return Ok(w_bool_from(match op {
                CompareOp::Lt => la < lb,
                CompareOp::Le => la <= lb,
                CompareOp::Gt => la > lb,
                CompareOp::Ge => la >= lb,
                CompareOp::Eq => la == lb,
                CompareOp::Ne => la != lb,
            }));
        }
        // dict equality — `pypy/objspace/std/dictmultiobject.py
        // W_DictMultiObject.descr_eq` is order-independent: same length
        // AND each key-value pair in `a` exists with equal value in `b`.
        // CPython only defines == / != for dicts (no ordering), so we
        // restrict to those ops; other ops fall through to the dunder
        // dispatch which currently raises TypeError, matching the
        // unimplemented `__lt__` etc. on plain dict.
        if is_exact_type(a, &pyre_object::DICT_TYPE)
            && is_exact_type(b, &pyre_object::DICT_TYPE)
            && matches!(op, CompareOp::Eq | CompareOp::Ne)
        {
            let la = pyre_object::w_dict_len(a);
            let lb = pyre_object::w_dict_len(b);
            let mut equal = la == lb;
            if equal {
                for (k, v) in pyre_object::w_dict_items(a) {
                    match pyre_object::dictmultiobject::w_dict_lookup_checked(b, k)
                        .map_err(|_| crate::baseobjspace::take_pending_dict_key_error(k))?
                    {
                        Some(other_v) => {
                            // dictmultiobject.py:664 `if not space.eq_w(w_val,
                            // w_rightval): return space.w_False`
                            if !crate::baseobjspace::eq_w(v, other_v)? {
                                equal = false;
                                break;
                            }
                        }
                        None => {
                            equal = false;
                            break;
                        }
                    }
                }
            }
            return Ok(w_bool_from(match op {
                CompareOp::Eq => equal,
                CompareOp::Ne => !equal,
                _ => unreachable!(),
            }));
        }
        // `dictmultiobject.py:1619-1623 _is_set_like` parity — when
        // one side is a set/frozenset and the other is a set-like
        // dict_view (Keys / Items), the comparison reduces to the
        // set-set arm with the dict_view materialised through its
        // snapshot.  Without this arm, `set == d.keys()` would fall
        // through to `object.__eq__`'s identity check and return
        // False even when the contents match.
        if (pyre_object::is_set_or_frozenset(a) || pyre_object::is_set_or_frozenset(b))
            && (pyre_object::dictmultiobject::is_dict_view(a)
                || pyre_object::dictmultiobject::is_dict_view(b))
        {
            let view_set_like = |obj: PyObjectRef| -> bool {
                if pyre_object::is_set_or_frozenset(obj) {
                    return true;
                }
                if pyre_object::dictmultiobject::is_dict_view(obj) {
                    let kind = pyre_object::dictmultiobject::w_dict_view_get_kind(obj);
                    return matches!(
                        kind,
                        pyre_object::dictmultiobject::DictViewKind::Keys
                            | pyre_object::dictmultiobject::DictViewKind::Items
                    );
                }
                false
            };
            if view_set_like(a) && view_set_like(b) {
                // A set operand stands in for itself; only a view is walked and
                // hashed into one. Rebuilding a set from its own elements would
                // hand each of them back to a user `__hash__`.
                let as_set = |obj: PyObjectRef| -> Result<PyObjectRef, PyError> {
                    if pyre_object::is_set_or_frozenset(obj) {
                        return Ok(obj);
                    }
                    crate::builtins::builtin_set_from_items(
                        &crate::type_methods::dict_view_snapshot(obj),
                    )
                };
                let a_set = as_set(a)?;
                let b_set = as_set(b)?;
                let la = pyre_object::w_set_len(a_set);
                let lb = pyre_object::w_set_len(b_set);
                let a_subset_b = || crate::typedef::set_is_subset_of(a_set, b_set);
                let b_subset_a = || crate::typedef::set_is_subset_of(b_set, a_set);
                return Ok(w_bool_from(match op {
                    CompareOp::Eq => la == lb && a_subset_b()?,
                    CompareOp::Ne => la != lb || !a_subset_b()?,
                    CompareOp::Le => la <= lb && a_subset_b()?,
                    CompareOp::Lt => la < lb && a_subset_b()?,
                    CompareOp::Ge => la >= lb && b_subset_a()?,
                    CompareOp::Gt => la > lb && b_subset_a()?,
                }));
            }
        }
        // set / frozenset comparison — subset / superset / equality.
        // PyPy: setobject.py W_BaseSetObject.descr_eq, descr_le, descr_lt
        if pyre_object::is_set_or_frozenset(a) && pyre_object::is_set_or_frozenset(b) {
            // The subset walks probe with the digest each element was stored
            // under (`setobject.py _issubset_unwrapped`), so a
            // comparison hashes nothing and an `eq_w` raised from a bucket
            // probe propagates instead of reading as "not a subset".
            let la = pyre_object::w_set_len(a);
            let lb = pyre_object::w_set_len(b);
            let a_subset_b = || crate::typedef::set_is_subset_of(a, b);
            let b_subset_a = || crate::typedef::set_is_subset_of(b, a);
            return Ok(w_bool_from(match op {
                CompareOp::Eq => la == lb && a_subset_b()?,
                CompareOp::Ne => la != lb || !a_subset_b()?,
                CompareOp::Le => la <= lb && a_subset_b()?,
                CompareOp::Lt => la < lb && a_subset_b()?,
                CompareOp::Ge => la >= lb && b_subset_a()?,
                CompareOp::Gt => la > lb && b_subset_a()?,
            }));
        }
        // List comparison. Unlike tuples, element comparison may mutate either
        // operand, so every loop boundary and the final size comparison read
        // the live lists (CPython 3.14 `list_richcompare_impl`; PyPy
        // `list_eq` / `_compare_unwrappeditems`).
        if is_list(a) && is_list(b) {
            if matches!(op, CompareOp::Eq | CompareOp::Ne)
                && pyre_object::w_list_len(a) != pyre_object::w_list_len(b)
            {
                return Ok(w_bool_from(matches!(op, CompareOp::Ne)));
            }

            let mut i = 0usize;
            while i < pyre_object::w_list_len(a) && i < pyre_object::w_list_len(b) {
                let ea = pyre_object::w_list_getitem(a, i as i64).unwrap_or(PY_NULL);
                let eb = pyre_object::w_list_getitem(b, i as i64).unwrap_or(PY_NULL);
                if !crate::baseobjspace::eq_w(ea, eb)? {
                    break;
                }
                i += 1;
            }
            let la = pyre_object::w_list_len(a);
            let lb = pyre_object::w_list_len(b);
            if i >= la || i >= lb {
                return Ok(w_bool_from(match op {
                    CompareOp::Lt => la < lb,
                    CompareOp::Le => la <= lb,
                    CompareOp::Eq => la == lb,
                    CompareOp::Ne => la != lb,
                    CompareOp::Gt => la > lb,
                    CompareOp::Ge => la >= lb,
                }));
            }
            if matches!(op, CompareOp::Eq | CompareOp::Ne) {
                return Ok(w_bool_from(matches!(op, CompareOp::Ne)));
            }
            // CPython deliberately fetches the live items again: equality may
            // have replaced either element before the ordering comparison.
            let ea = pyre_object::w_list_getitem(a, i as i64).unwrap_or(PY_NULL);
            let eb = pyre_object::w_list_getitem(b, i as i64).unwrap_or(PY_NULL);
            return compare(ea, eb, op);
        }
        // range value comparison — functional.py W_Range.descr_eq:
        // two ranges are equal iff they generate the same sequence
        // (equal length, and for non-empty ranges equal start and —
        // for length > 1 — equal step).  Only `==` / `!=` are defined;
        // ordering falls through to the dunder dispatch (TypeError).
        if pyre_object::is_w_range(a)
            && pyre_object::is_w_range(b)
            && matches!(op, CompareOp::Eq | CompareOp::Ne)
        {
            let equal = pyre_object::w_range_eq(a, b);
            return Ok(w_bool_from(match op {
                CompareOp::Eq => equal,
                CompareOp::Ne => !equal,
                _ => unreachable!(),
            }));
        }
        let dunder = match op {
            CompareOp::Lt => "__lt__",
            CompareOp::Le => "__le__",
            CompareOp::Gt => "__gt__",
            CompareOp::Ge => "__ge__",
            CompareOp::Eq => "__eq__",
            CompareOp::Ne => "__ne__",
        };
        // `SetLikeDictView` operands expose comparison dunders through the
        // typedef. Instance-shaped operands were already handled by
        // `try_compare_override`, so only non-instance operands dispatch here.
        // Reflected: if RHS is a dict view, try `b.dunder(a)` —
        // PyPy's `_is_set_like(other)` short-circuits the LHS-side
        // descr_eq when the other is set-like, so the reflected call
        // path is the one that succeeds for `set == d.keys()`.
        if !is_instance(a)
            && let Some(a_type) = crate::typedef::r#type(a)
            && let Some(method) = lookup_in_type_where(a_type.as_ptr(), dunder)
        {
            // A raised exception (not NotImplemented) propagates; only
            // NotImplemented falls through to the reflected comparison.
            let result = crate::call::call_function_impl_result(method, &[a, b])?;
            if !is_not_implemented(result) {
                return Ok(result);
            }
        }
        if !is_instance(b)
            && let Some(rdunder) = reverse_dunder(dunder)
            && let Some(b_type) = crate::typedef::r#type(b)
            && let Some(method) = lookup_in_type_where(b_type.as_ptr(), rdunder)
        {
            let result = crate::call::call_function_impl_result(method, &[b, a])?;
            if !is_not_implemented(result) {
                return Ok(result);
            }
        }
        // Identity comparison fallback for == and !=
        if matches!(op, CompareOp::Eq) {
            return Ok(w_bool_from(std::ptr::eq(a, b)));
        }
        if matches!(op, CompareOp::Ne) {
            return Ok(w_bool_from(!std::ptr::eq(a, b)));
        }
        let a_name = crate::baseobjspace::object_functionstr_type_name(a);
        let b_name = crate::baseobjspace::object_functionstr_type_name(b);
        let op_symbol = op.symbol();
        Err(PyError::type_error(format!(
            "'{op_symbol}' not supported between instances of '{a_name}' and '{b_name}'"
        )))
    }
}

/// Comparison operator enum (mirrors RustPython's ComparisonOperator).
#[derive(Debug, Clone, Copy)]
pub enum CompareOp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

impl CompareOp {
    /// The Python operator symbol (`<`, `<=`, `>`, `>=`, `==`, `!=`) used in
    /// the "not supported between instances of" TypeError message.
    pub fn symbol(self) -> &'static str {
        match self {
            CompareOp::Lt => "<",
            CompareOp::Le => "<=",
            CompareOp::Gt => ">",
            CompareOp::Ge => ">=",
            CompareOp::Eq => "==",
            CompareOp::Ne => "!=",
        }
    }
}

/// Unary positive (`+a`).

pub fn pos(a: PyObjectRef) -> PyResult {
    unsafe {
        if let Some(result) = try_numeric_unaryop_override(a, "__pos__")? {
            return Ok(result);
        }
        if is_int(a) || is_bool(a) {
            return Ok(w_int_new(int_value(a)));
        }
        if is_long(a) {
            // intobject.py:182-191 `_self_unaryop('pos')` delegates to
            // `self.int(space)`, which returns `self` only for the exact
            // builtin representation.  A subclass returns a plain-int copy
            // (long_pos: exact → self, else `_PyLong_Copy`), so leaking the
            // subclass instance would be wrong; an exact long returns `self`
            // without allocating another wrapper or shallow-cloned payload.
            if pyre_object::is_exact_builtin_instance(a) {
                return Ok(a);
            }
            return Ok(pyre_object::longobject::w_long_new_fresh_rbigint_handle(
                bigint_clone(w_long_get_value(a)),
            ));
        }
        if is_float(a) {
            return Ok(w_float_new(w_float_get_value(a)));
        }
        if is_complex(a) {
            let (ar, ai) = complex_val(a).unwrap();
            return Ok(w_complex_new(ar, ai));
        }
        if let Some(result) = try_instance_unaryop(a, "__pos__")? {
            return Ok(result);
        }
        if a.is_null() {
            return Err(PyError::type_error(
                "unsupported operand type for unary pos: 'NoneType'",
            ));
        }
        Err(PyError::type_error(format!(
            "unsupported operand type for unary pos: '{}'",
            crate::baseobjspace::object_functionstr_type_name(a),
        )))
    }
}

/// Unary negation.

pub fn neg(a: PyObjectRef) -> PyResult {
    unsafe {
        if let Some(result) = try_numeric_unaryop_override(a, "__neg__")? {
            return Ok(result);
        }
        if is_int(a) || is_bool(a) {
            let v = int_value(a);
            return match v.checked_neg() {
                Some(r) => Ok(w_int_new(r)),
                None => Ok(pyre_object::longobject::w_long_new_fresh_rbigint_handle(
                    bigint_neg(&BigInt::from(v)),
                )),
            };
        }
        if is_long(a) {
            return Ok(pyre_object::longobject::w_long_new_fresh_rbigint_handle(
                bigint_neg(w_long_get_value(a)),
            ));
        }
        if is_float(a) {
            return Ok(w_float_new(-w_float_get_value(a)));
        }
        if is_complex(a) {
            return complex_neg(a);
        }
        // Instance __neg__
        if let Some(result) = try_instance_unaryop(a, "__neg__")? {
            return Ok(result);
        }
        if a.is_null() {
            return Err(PyError::type_error(
                "unsupported operand type for unary neg: 'NoneType'",
            ));
        }
        Err(PyError::type_error(format!(
            "unsupported operand type for unary neg: '{}'",
            crate::baseobjspace::object_functionstr_type_name(a),
        )))
    }
}

const BOOL_INVERT_DEPRECATION_TEXT: &str = "Bitwise inversion '~' on bool is deprecated and will be removed in \
Python 3.16. This returns the bitwise inversion of the underlying int \
object and is usually not what you expect from negating a bool. \
Use the 'not' operator for boolean negation or ~int(x) if you really want \
the bitwise inversion of the underlying int.";

/// The wrapped form of [`BOOL_INVERT_DEPRECATION_TEXT`]: `~True` in a loop
/// issues the same message on every iteration, and the registry lookup that
/// deduplicates it hashes and compares the message each time.
///
/// `dont_look_inside` because the cell is a `static` of a host type, which
/// the front-end cannot lift: reaching it from a lifted body fails the
/// callee's own lift and, transitively, every caller's — [`invert`] then has
/// no jitcode, the tracer has to residualise the whole `~` opcode, and a
/// virtualizable forced inside that residual aborts the trace
/// (`ABORT_ESCAPE`), so a `~bool` in a hot loop stops the loop compiling at
/// all.  Hiding the static behind an opaque accessor keeps the body out of
/// the lift, exactly as `w_dict_new` does for its host `IndexMap`.
#[majit_macros::dont_look_inside]
pub(crate) fn bool_invert_deprecation_text() -> PyObjectRef {
    static CELL: crate::warn::PrebuiltText = crate::warn::PrebuiltText::new();
    CELL.get(BOOL_INVERT_DEPRECATION_TEXT)
}

/// Unary bitwise inversion.

pub fn invert(a: PyObjectRef) -> PyResult {
    unsafe {
        if let Some(result) = try_numeric_unaryop_override(a, "__invert__")? {
            return Ok(result);
        }
        if is_bool(a) {
            // CPython 3.14 `Objects/boolobject.c:bool_invert`.  The bundled
            // PyPy source inherits `W_IntObject.descr_invert`; 3.14 inserts
            // this warning-bearing bool slot before the integer inversion.
            crate::warn::warn_category_w(bool_invert_deprecation_text(), "DeprecationWarning", 2)?;
            return Ok(w_int_new(!int_value(a)));
        }
        if is_int(a) {
            return Ok(w_int_new(!int_value(a)));
        }
        if is_long(a) {
            return Ok(w_long_new(bigint_invert(w_long_get_value(a))));
        }
        if let Some(result) = try_instance_unaryop(a, "__invert__")? {
            return Ok(result);
        }
        Err(PyError::type_error(format!(
            "unsupported operand type for unary ~: '{}'",
            crate::baseobjspace::object_functionstr_type_name(a),
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_pointer_width = "64")]
    fn decoded_bigint_result(value: pyre_object::longobject::JitBigIntResult) -> *mut BigInt {
        value
    }

    #[cfg(target_pointer_width = "32")]
    fn decoded_bigint_result(value: pyre_object::longobject::JitBigIntResult) -> *mut BigInt {
        value as usize as *mut BigInt
    }

    /// Test projection for assertions that accept either compact or long
    /// integer results. Production consumers borrow W_LongObject payloads.
    unsafe fn as_bigint(obj: PyObjectRef) -> BigInt {
        if is_bool(obj) {
            BigInt::from(w_bool_get_value(obj) as i64)
        } else if is_int(obj) {
            BigInt::from(w_int_get_value(obj))
        } else {
            w_long_get_value(obj).clone()
        }
    }

    fn assert_compare_bool(a: PyObjectRef, b: PyObjectRef, op: CompareOp, expected: bool) {
        let result = compare(a, b, op).unwrap();
        assert_eq!(unsafe { w_bool_get_value(result) }, expected);
    }

    #[test]
    fn test_int_add() {
        let a = w_int_new(3);
        let b = w_int_new(4);
        let result = add(a, b).unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 7) };
    }

    #[test]
    fn test_int_compare() {
        let a = w_int_new(5);
        let b = w_int_new(10);
        let result = compare(a, b, CompareOp::Lt).unwrap();
        unsafe { assert!(w_bool_get_value(result)) };
    }

    #[test]
    fn test_specialised_tuple_equality_uses_same_class_raw_slots() {
        // The `_oo` case interns its strs, which reaches a str-keyed dict.
        crate::test_hooks::install_hash_hook();
        assert_compare_bool(
            w_tuple_new(vec![w_int_new(1), w_int_new(2)]),
            w_tuple_new(vec![w_int_new(1), w_int_new(2)]),
            CompareOp::Eq,
            true,
        );
        assert_compare_bool(
            w_tuple_new(vec![w_int_new(1), w_int_new(2)]),
            w_tuple_new(vec![w_int_new(1), w_int_new(3)]),
            CompareOp::Eq,
            false,
        );
        assert_compare_bool(
            pyre_object::makespecialisedtuple2(w_float_new(1.0), w_float_new(2.0)),
            pyre_object::makespecialisedtuple2(w_float_new(1.0), w_float_new(2.0)),
            CompareOp::Eq,
            true,
        );

        let nan = w_float_new(f64::NAN);
        let nan_pair = pyre_object::makespecialisedtuple2(nan, nan);
        assert_compare_bool(nan_pair, nan_pair, CompareOp::Eq, true);

        assert_compare_bool(
            w_tuple_new(vec![w_str_new("a"), w_str_new("b")]),
            w_tuple_new(vec![w_str_new("a"), w_str_new("b")]),
            CompareOp::Eq,
            true,
        );
        assert_compare_bool(
            w_tuple_new(vec![w_int_new(1), w_int_new(2)]),
            pyre_object::w_tuple_new_array_backed(vec![w_int_new(1), w_int_new(2)]),
            CompareOp::Eq,
            true,
        );
    }

    #[test]
    fn test_long_pos_preserves_exact_object_identity() {
        let value = BigInt::from(1).lshift(80).unwrap();
        let a = w_long_new(value);
        let result = pos(a).unwrap();
        assert_eq!(result, a);
    }

    #[test]
    fn test_zero_division() {
        let a = w_int_new(5);
        let b = w_int_new(0);
        assert!(floordiv(a, b).is_err());
    }

    #[test]
    fn test_int_floordiv_and_mod_bounce_min_overflow_to_rbigint() {
        // intobject.py uses `ovfcheck` for both operations.  On a signed
        // machine word MIN / -1 is their sole non-zero-divisor overflow.
        let a = w_int_new(i64::MIN);
        let b = w_int_new(-1);
        let q = floordiv(a, b).unwrap();
        let r = mod_(a, b).unwrap();
        unsafe {
            assert_eq!(as_bigint(q), BigInt::from(i64::MAX).int_add(1));
            assert_eq!(as_bigint(r), BigInt::zero());
        }
    }

    #[test]
    fn test_long_floordiv_and_mod_keep_python_sign_rules() {
        let magnitude = BigInt::one().lshift(80).unwrap().int_add(13);
        for (lhs, rhs) in [
            (magnitude.clone(), BigInt::fromint(10)),
            (magnitude.neg(), BigInt::fromint(10)),
            (magnitude.clone(), BigInt::fromint(-10)),
            (magnitude.neg(), BigInt::fromint(-10)),
        ] {
            let a = w_long_new(lhs.clone());
            let b = w_long_new(rhs.clone());
            let q = floordiv(a, b).unwrap();
            let r = mod_(a, b).unwrap();
            let q = unsafe { as_bigint(q) };
            let r = unsafe { as_bigint(r) };
            assert!(q.mul(&rhs).add(&r).eq(&lhs));
            assert!(r.get_sign() == 0 || r.get_sign() == rhs.get_sign());
        }
    }

    #[test]
    fn test_truthiness() {
        assert!(is_true(w_int_new(1)).unwrap());
        assert!(!is_true(w_int_new(0)).unwrap());
        assert!(!is_true(w_none()).unwrap());
        assert!(is_true(w_bool_from(true)).unwrap());
        assert!(!is_true(w_bool_from(false)).unwrap());
    }

    #[test]
    fn test_int_add_overflow() {
        let a = w_int_new(i64::MAX);
        let b = w_int_new(1);
        let result = add(a, b).unwrap();
        unsafe {
            assert!(is_long(result));
            assert_eq!(
                *w_long_get_value(result),
                BigInt::from(i64::MAX) + BigInt::from(1)
            );
        }
    }

    #[test]
    fn test_int_sub_overflow() {
        let a = w_int_new(i64::MIN);
        let b = w_int_new(1);
        let result = sub(a, b).unwrap();
        unsafe {
            assert!(is_long(result));
            assert_eq!(
                *w_long_get_value(result),
                BigInt::from(i64::MIN) - BigInt::from(1)
            );
        }
    }

    #[test]
    fn test_int_mul_overflow() {
        let a = w_int_new(i64::MAX);
        let b = w_int_new(2);
        let result = mul(a, b).unwrap();
        unsafe {
            assert!(is_long(result));
            assert_eq!(
                *w_long_get_value(result),
                BigInt::from(i64::MAX) * BigInt::from(2)
            );
        }
    }

    #[test]
    fn test_long_add() {
        let a = w_long_new(BigInt::from(i64::MAX) + BigInt::from(1));
        let b = w_int_new(100);
        let result = add(a, b).unwrap();
        unsafe {
            assert!(is_long(result));
            assert_eq!(
                *w_long_get_value(result),
                BigInt::from(i64::MAX) + BigInt::from(101)
            );
        }
    }

    #[test]
    fn test_long_add_keeps_long_when_fits() {
        // long + int whose sum fits back in i64 stays a W_LongObject: `newlong`
        // never demotes (withsmalllong=False), so a shrunk long keeps the long
        // representation.
        let a = w_long_new(BigInt::from(i64::MAX) + BigInt::from(1));
        let b = w_int_new(-1);
        let result = add(a, b).unwrap();
        unsafe {
            assert!(is_long(result));
            assert_eq!(*w_long_get_value(result), BigInt::from(i64::MAX));
        }
    }

    #[test]
    fn test_mixed_long_int_operations_match_rbigint_int_specializations() {
        // pypy/objspace/std/longobject.py:_make_generic_descr_binop routes
        // mixed W_LongObject/W_IntObject operations through rbigint.int_*.
        // Use a multi-digit value and both signs so this covers paths where
        // materializing the small operand as a second rbigint would select a
        // different internal algorithm.
        let magnitude = BigInt::one()
            .lshift(190)
            .unwrap()
            .int_add(0x0123_4567_89ab_cdef);
        for value in [magnitude.clone(), magnitude.neg()] {
            for machine in [-17_i64, -1, 0, 1, 37, i64::MIN, i64::MAX] {
                let expected_add = value.int_add(machine);
                let expected_mul = value.int_mul(machine);
                let expected_and = value.int_and_(machine);
                let expected_or = value.int_or_(machine);
                let expected_xor = value.int_xor(machine);
                let expected_sub = value.int_sub(machine);
                let expected_rsub = BigInt::fromint(machine).sub(&value);

                for result in [
                    add(w_long_new(value.clone()), w_int_new(machine)).unwrap(),
                    add(w_int_new(machine), w_long_new(value.clone())).unwrap(),
                ] {
                    assert!(unsafe { as_bigint(result) }.eq(&expected_add));
                }
                for result in [
                    mul(w_long_new(value.clone()), w_int_new(machine)).unwrap(),
                    mul(w_int_new(machine), w_long_new(value.clone())).unwrap(),
                ] {
                    assert!(unsafe { as_bigint(result) }.eq(&expected_mul));
                }
                for result in [
                    and_(w_long_new(value.clone()), w_int_new(machine)).unwrap(),
                    and_(w_int_new(machine), w_long_new(value.clone())).unwrap(),
                ] {
                    assert!(unsafe { as_bigint(result) }.eq(&expected_and));
                }
                for result in [
                    or_(w_long_new(value.clone()), w_int_new(machine)).unwrap(),
                    or_(w_int_new(machine), w_long_new(value.clone())).unwrap(),
                ] {
                    assert!(unsafe { as_bigint(result) }.eq(&expected_or));
                }
                for result in [
                    xor(w_long_new(value.clone()), w_int_new(machine)).unwrap(),
                    xor(w_int_new(machine), w_long_new(value.clone())).unwrap(),
                ] {
                    assert!(unsafe { as_bigint(result) }.eq(&expected_xor));
                }

                let result = sub(w_long_new(value.clone()), w_int_new(machine)).unwrap();
                assert!(unsafe { as_bigint(result) }.eq(&expected_sub));
                let reflected = sub(w_int_new(machine), w_long_new(value.clone())).unwrap();
                assert!(unsafe { as_bigint(reflected) }.eq(&expected_rsub));
            }
        }

        // W_BoolObject subclasses W_IntObject upstream, so bool operands use
        // the same machine-int specialization.
        let expected = magnitude.int_add(1);
        for result in [
            add(w_long_new(magnitude.clone()), w_bool_from(true)).unwrap(),
            add(w_bool_from(true), w_long_new(magnitude.clone())).unwrap(),
        ] {
            assert!(unsafe { as_bigint(result) }.eq(&expected));
        }
    }

    #[test]
    fn test_jit_w_long_floordiv_mod_raw() {
        // Both operands out of i64 range → long // long / long % long fast path
        // payload helpers return a bare `*mut BigInt` of the quotient/remainder.
        let x = BigInt::from(i64::MAX) * BigInt::from(1000) + BigInt::from(7);
        let y = BigInt::from(i64::MAX) + BigInt::from(3);
        let a = w_long_new(x.clone());
        let b = w_long_new(y.clone());
        unsafe {
            let d = jit_w_long_floordiv_raw(a as i64, b as i64) as *mut BigInt;
            assert_eq!(*d, x.floordiv(&y).expect("test divisor is nonzero"));
            let m = jit_w_long_mod_raw(a as i64, b as i64) as *mut BigInt;
            assert_eq!(*m, x.r#mod(&y).expect("test divisor is nonzero"));
        }
    }

    #[test]
    fn test_rbigint_residuals_preserve_upstream_reference_fast_paths() {
        let magnitude = BigInt::one().lshift(130).unwrap().int_add(17);
        let larger = magnitude.lshift(70).unwrap();
        let one = pyre_object::longobject::alloc_bigint_nursery(BigInt::one());
        let zero = pyre_object::longobject::alloc_bigint_nursery(BigInt::zero());
        let a = pyre_object::longobject::alloc_bigint_nursery(magnitude);
        let b = pyre_object::longobject::alloc_bigint_nursery(larger);

        assert_eq!(
            decoded_bigint_result(jit_bigint_add(a as i64, zero as i64)),
            a
        );
        assert_eq!(
            decoded_bigint_result(jit_bigint_sub(a as i64, zero as i64)),
            a
        );
        assert_eq!(decoded_bigint_result(jit_bigint_int_add(a as i64, 0)), a);
        assert_eq!(decoded_bigint_result(jit_bigint_int_sub(a as i64, 0)), a);
        assert_eq!(decoded_bigint_result(jit_bigint_int_mul(a as i64, 1)), a);
        assert_eq!(
            decoded_bigint_result(jit_bigint_int_div_floor(a as i64, 1)),
            a
        );
        assert_eq!(
            decoded_bigint_result(jit_bigint_pow_nomod(a as i64, one as i64)),
            a
        );
        assert_eq!(
            decoded_bigint_result(jit_bigint_int_pow_nomod(a as i64, 1)),
            a
        );
        assert_eq!(
            decoded_bigint_result(jit_bigint_lshift_count(a as i64, 0)),
            a
        );
        assert_eq!(decoded_bigint_result(jit_bigint_shl(a as i64, 0)), a);
        assert_eq!(decoded_bigint_result(jit_bigint_shr(a as i64, 0)), a);

        // `_divrem` returns its literal input remainder when its early
        // magnitude test succeeds. Floored modulo retains it for equal signs
        // and a non-one-digit divisor.
        assert_eq!(decoded_bigint_result(jit_bigint_rem(a as i64, b as i64)), a);
        assert_eq!(
            decoded_bigint_result(jit_bigint_mod_floor(a as i64, b as i64)),
            a
        );

        // `lshift` returns even a non-canonical zero handle unchanged.
        let fresh_zero = majit_rlib::rbigint::alloc_rbigint_clone_nursery_collecting(unsafe {
            (&*zero).clone()
        });
        assert_ne!(fresh_zero, zero);
        assert_eq!(
            decoded_bigint_result(jit_bigint_lshift_count(fresh_zero as i64, 37)),
            fresh_zero
        );
    }

    #[test]
    fn test_machine_int_host_seams_use_dedicated_rbigint_legs() {
        let value = BigInt::one().lshift(130).unwrap().int_add(17);
        let quotient = bigint_int_floordiv_nonzero(&value, 3);
        assert!(quotient.eq(&value.int_floordiv(3).unwrap()));

        let power = bigint_int_pow_nomod(&value, 3).expect("small exponent");
        assert!(power.eq(&value.int_pow(3, None).unwrap()));
    }

    #[test]
    fn test_bigint_truediv_exponent() {
        // Regression: the exponent assembly carried a spurious `+ 1` that
        // doubled every quotient (equal operands gave 2.0, not 1.0).
        let big = BigInt::from(10u64).int_pow(40, None).unwrap();
        assert_eq!(bigint_truediv(&big, &big).unwrap(), 1.0);
        let a = BigInt::from(10u64).int_pow(60, None).unwrap();
        let b = BigInt::from(2) * BigInt::from(10u64).int_pow(59, None).unwrap();
        assert_eq!(bigint_truediv(&a, &b).unwrap(), 5.0);
        assert_eq!(bigint_truediv(&a.neg(), &b).unwrap(), -5.0);
        assert_eq!(bigint_truediv(&a, &b.neg()).unwrap(), -5.0);
        assert!(bigint_truediv(&a, &BigInt::from(0)).is_err());
    }

    #[test]
    fn test_machine_int_truediv_uses_rbigint_rounding_past_binary64_mantissa() {
        let result = truediv_builtin(w_int_new(63_050_394_783_186_940), w_int_new(7)).unwrap();
        unsafe {
            assert_eq!(w_float_get_value(result), 9_007_199_254_740_991.0);
        }
    }

    #[test]
    fn test_bigint_truediv_sticky_rounding() {
        // a ≫ b (shift < 0): low bits of `a` that a right-shift would discard
        // must still steer round-half-to-even. b is odd and > 2^63 so the path
        // exercises the bigint divide, not i64.
        let b = BigInt::from(2u64).int_pow(64, None).unwrap() + BigInt::from(1); // 2^64 + 1, odd
        let two55 = 2.0_f64.powi(55);
        // a_exact/b == 2^55 + 4 exactly: a half-ULP tie between 2^55 and 2^55+8.
        // Round-half-to-even → 2^55 (its low mantissa bit is 0).
        let a_exact = (BigInt::from(2u64).int_pow(53, None).unwrap() + BigInt::from(1))
            * BigInt::from(4)
            * &b;
        assert_eq!(bigint_truediv(&a_exact, &b).unwrap(), two55);
        // +1 makes the true quotient exceed the tie → sticky → round up to 2^55+8.
        assert_eq!(
            bigint_truediv(&a_exact.int_add(1), &b).unwrap(),
            two55 + 8.0
        );
        // -1 drops it just below the tie → round down to 2^55.
        assert_eq!(bigint_truediv(&a_exact.int_sub(1), &b).unwrap(), two55);
    }

    #[test]
    fn test_bigint_truediv_subnormal() {
        // Subnormal-range results must match what `math.ldexp` produces; a lone
        // `2f64.powi` underflows the scale and loses them. Expected bit patterns
        // are CPython's.
        let p = |e: u32| BigInt::from(2u64).int_pow(e as i64, None).unwrap();
        // 1 / 2^1030 == 2^-1030 (exact subnormal)
        assert_eq!(
            bigint_truediv(&BigInt::from(1), &p(1030)).unwrap(),
            f64::from_bits(0x0000_1000_0000_0000)
        );
        // 7 / 2^1074 == 7 * 2^-1074 (seven smallest subnormals)
        assert_eq!(
            bigint_truediv(&BigInt::from(7), &p(1074)).unwrap(),
            f64::from_bits(0x0000_0000_0000_0007)
        );
        // (2^53+1) / (2^1075+7) rounds across the subnormal/normal boundary to 2^-1022
        assert_eq!(
            bigint_truediv(&p(53).int_add(1), &p(1075).int_add(7)).unwrap(),
            f64::from_bits(0x0010_0000_0000_0000)
        );
        // sign preserved
        assert_eq!(
            bigint_truediv(&BigInt::from(-1), &p(1030)).unwrap(),
            -f64::from_bits(0x0000_1000_0000_0000)
        );
    }

    #[test]
    fn test_jit_w_long_shift_truediv_raw() {
        let x = BigInt::from(i64::MAX) * BigInt::from(1000) + BigInt::from(7);
        let a = w_long_new(x.clone());
        let two = w_long_new(BigInt::from(2));
        let y = BigInt::from(i64::MAX) + BigInt::from(3);
        let b = w_long_new(y.clone());
        unsafe {
            let l = jit_w_long_lshift_raw(a as i64, two as i64) as *mut BigInt;
            assert_eq!(*l, bigint_lshift(&x, 2).unwrap());
            let r = jit_w_long_rshift_raw(a as i64, two as i64) as *mut BigInt;
            assert_eq!(*r, bigint_rshift(&x, 2));
            // true-divide returns the f64 quotient directly (CallPureF).
            let f = jit_w_long_truediv_raw(a as i64, b as i64);
            assert_eq!(f, bigint_truediv(&x, &y).unwrap());
        }
    }

    #[test]
    fn test_negate_min_int() {
        let a = w_int_new(i64::MIN);
        let result = neg(a).unwrap();
        unsafe {
            assert!(is_long(result));
            assert_eq!(*w_long_get_value(result), -BigInt::from(i64::MIN));
        }
    }

    #[test]
    fn test_long_neg_and_zero_count_shifts_preserve_rpython_payload_identity() {
        unsafe {
            // rbigint.neg always constructs a fresh handle, including zero.
            let zero = w_long_new(BigInt::zero());
            let negated = neg(zero).unwrap();
            assert_ne!(negated, zero);
            assert_ne!(w_long_get_raw_value(negated), w_long_get_raw_value(zero));
            assert!(w_long_get_value(negated).is_zero());

            let magnitude = BigInt::one().lshift(130).unwrap().int_add(17);
            let value = w_long_new(magnitude);
            let count_zero = w_int_new(0);

            // rbigint returns `self`; newlong creates a fresh W_LongObject
            // around that exact translated payload.
            let shifted_left = long_lshift(value, count_zero).unwrap();
            assert_ne!(shifted_left, value);
            assert_eq!(
                w_long_get_raw_value(shifted_left),
                w_long_get_raw_value(value)
            );
            let shifted_right = long_rshift(value, count_zero).unwrap();
            assert_ne!(shifted_right, value);
            assert_eq!(
                w_long_get_raw_value(shifted_right),
                w_long_get_raw_value(value)
            );

            let zero_shifted = long_lshift(zero, w_int_new(5)).unwrap();
            assert_ne!(zero_shifted, zero);
            assert_eq!(
                w_long_get_raw_value(zero_shifted),
                w_long_get_raw_value(zero)
            );

            // Overflowing count is the W_LongObject-level early return.
            let huge_count = w_long_new(BigInt::one().lshift(100).unwrap());
            assert_eq!(long_lshift(zero, huge_count).unwrap(), zero);
        }
    }

    #[test]
    fn test_long_arithmetic_rewraps_rbigint_operand_fast_paths() {
        unsafe {
            let magnitude = BigInt::one().lshift(130).unwrap().int_add(17);
            let value = w_long_new(magnitude.translated_alias());
            let zero = w_long_new(BigInt::zero());
            let payload = w_long_get_raw_value(value);

            for result in [
                long_add(value, w_int_new(0)).unwrap(),
                long_add(w_int_new(0), value).unwrap(),
                long_add(value, zero).unwrap(),
                long_add(zero, value).unwrap(),
                long_sub(value, w_int_new(0)).unwrap(),
                long_sub(value, zero).unwrap(),
                long_mul(value, w_int_new(1)).unwrap(),
                long_mul(w_bool_from(true), value).unwrap(),
                long_floordiv(value, w_int_new(1)).unwrap(),
                long_pow(value, w_int_new(1)).unwrap(),
            ] {
                assert_ne!(result, value);
                assert_eq!(w_long_get_raw_value(result), payload);
            }

            // `rbigint.int_floordiv(1)` returns `self`, but the long-divisor
            // path goes through `rbigint.divmod` -> `int_divmod`, whose
            // quotient is freshly constructed.
            let long_divisor_result = long_floordiv(value, w_long_new(BigInt::one())).unwrap();
            assert_ne!(w_long_get_raw_value(long_divisor_result), payload);
            assert!(w_long_get_value(long_divisor_result).eq(w_long_get_value(value)));

            // These upstream methods check zero before their alias shortcut
            // and return canonical NULLRBIGINT, not a fresh zero operand.
            let fresh_zero = neg(zero).unwrap();
            assert_ne!(w_long_get_raw_value(fresh_zero), w_long_get_raw_value(zero));
            for result in [
                long_add(fresh_zero, w_int_new(0)).unwrap(),
                long_mul(fresh_zero, w_int_new(1)).unwrap(),
                long_pow(fresh_zero, w_int_new(1)).unwrap(),
            ] {
                assert_eq!(w_long_get_raw_value(result), w_long_get_raw_value(zero));
            }

            let larger = w_long_new(magnitude.lshift(70).unwrap());
            let remainder = long_mod(value, larger).unwrap();
            assert_eq!(w_long_get_raw_value(remainder), payload);
            let pair = integer_divmod_pair(value, larger).unwrap();
            let pair_remainder = w_tuple_getitem(pair, 1).expect("remainder");
            assert_eq!(w_long_get_raw_value(pair_remainder), payload);

            let pair = integer_divmod_pair(value, w_int_new(1)).unwrap();
            let pair_quotient = w_tuple_getitem(pair, 0).expect("quotient");
            assert_ne!(w_long_get_raw_value(pair_quotient), payload);
            assert!(w_long_get_value(pair_quotient).eq(w_long_get_value(value)));
        }
    }

    #[test]
    fn test_invert_int() {
        let result = invert(w_int_new(6)).unwrap();
        unsafe {
            assert!(is_int(result));
            assert_eq!(w_int_get_value(result), !6);
        }
    }

    #[test]
    fn test_long_compare() {
        let a = w_long_new(BigInt::from(i64::MAX) + BigInt::from(1));
        let b = w_int_new(i64::MAX);
        let result = compare(a, b, CompareOp::Gt).unwrap();
        unsafe { assert!(w_bool_get_value(result)) };

        let magnitude = BigInt::one().lshift(190).unwrap();
        let a = w_long_new(magnitude.int_add(1));
        let b = w_long_new(magnitude.int_add(2));
        for (op, expected) in [
            (CompareOp::Lt, true),
            (CompareOp::Le, true),
            (CompareOp::Gt, false),
            (CompareOp::Ge, false),
            (CompareOp::Eq, false),
            (CompareOp::Ne, true),
        ] {
            let result = compare(a, b, op).unwrap();
            assert_eq!(unsafe { w_bool_get_value(result) }, expected);
        }
    }

    #[test]
    fn test_complex_long_comparison_is_exact() {
        let magnitude = BigInt::one().lshift(100).unwrap();
        let complex = w_complex_new(2_f64.powi(100), 0.0);
        let equal = compare(complex, w_long_new(magnitude.clone()), CompareOp::Eq).unwrap();
        assert!(unsafe { w_bool_get_value(equal) });

        let unequal = compare(complex, w_long_new(magnitude.int_add(1)), CompareOp::Eq).unwrap();
        assert!(!unsafe { w_bool_get_value(unequal) });
    }

    #[test]
    fn test_mixed_long_int_comparisons_match_rbigint_int_specializations() {
        let magnitude = BigInt::one().lshift(190).unwrap().int_add(37);
        for value in [magnitude.clone(), magnitude.neg()] {
            for machine in [-1_i64, 0, 1, i64::MIN, i64::MAX] {
                for op in [
                    CompareOp::Lt,
                    CompareOp::Le,
                    CompareOp::Gt,
                    CompareOp::Ge,
                    CompareOp::Eq,
                    CompareOp::Ne,
                ] {
                    let expected = match op {
                        CompareOp::Lt => value.int_lt(machine),
                        CompareOp::Le => value.int_le(machine),
                        CompareOp::Gt => value.int_gt(machine),
                        CompareOp::Ge => value.int_ge(machine),
                        CompareOp::Eq => value.int_eq(machine),
                        CompareOp::Ne => value.int_ne(machine),
                    };
                    let result =
                        compare(w_long_new(value.clone()), w_int_new(machine), op).unwrap();
                    assert_eq!(unsafe { w_bool_get_value(result) }, expected);

                    let expected_reversed = match op {
                        CompareOp::Lt => value.int_gt(machine),
                        CompareOp::Le => value.int_ge(machine),
                        CompareOp::Gt => value.int_lt(machine),
                        CompareOp::Ge => value.int_le(machine),
                        CompareOp::Eq => value.int_eq(machine),
                        CompareOp::Ne => value.int_ne(machine),
                    };
                    let result =
                        compare(w_int_new(machine), w_long_new(value.clone()), op).unwrap();
                    assert_eq!(unsafe { w_bool_get_value(result) }, expected_reversed);
                }
            }
        }

        for (op, expected) in [
            (CompareOp::Gt, true),
            (CompareOp::Ge, true),
            (CompareOp::Eq, false),
            (CompareOp::Ne, true),
        ] {
            let result = compare(w_long_new(magnitude.clone()), w_bool_from(true), op).unwrap();
            assert_eq!(unsafe { w_bool_get_value(result) }, expected);
        }
    }

    #[test]
    fn test_long_truthiness() {
        assert!(is_true(w_long_new(BigInt::from(i64::MAX) + BigInt::from(1))).unwrap());
        assert!(!is_true(w_long_new(BigInt::from(0))).unwrap());
    }

    #[test]
    fn test_numeric_divmod_computes_matching_pair_for_all_integer_shapes() {
        let big_positive = BigInt::one().lshift(190).unwrap().int_add(37);
        let big_negative = big_positive.neg();
        for (lhs, rhs) in [
            (big_positive.clone(), BigInt::from(-257)),
            (big_negative.clone(), BigInt::from(257)),
            (BigInt::from(-257), big_positive.clone()),
            (BigInt::from(i64::MIN), BigInt::from(-1)),
        ] {
            let lhs_obj = if lhs.toint().is_ok() {
                w_int_new(lhs.toint().unwrap())
            } else {
                w_long_new(lhs.clone())
            };
            let rhs_obj = if rhs.toint().is_ok() {
                w_int_new(rhs.toint().unwrap())
            } else {
                w_long_new(rhs.clone())
            };
            let expected = lhs.divmod(&rhs).unwrap();
            let pair = divmod(lhs_obj, rhs_obj).unwrap();
            let q = unsafe { w_tuple_getitem(pair, 0).expect("quotient") };
            let r = unsafe { w_tuple_getitem(pair, 1).expect("remainder") };
            assert!(unsafe { as_bigint(q) }.eq(&expected.0));
            assert!(unsafe { as_bigint(r) }.eq(&expected.1));
        }
    }

    #[test]
    fn test_int_pow() {
        let result = pow(w_int_new(2), w_int_new(10)).unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 1024) };
    }

    #[test]
    fn test_int_pow_overflow() {
        let result = pow(w_int_new(2), w_int_new(63)).unwrap();
        unsafe {
            // 2^63 overflows i64, should be long
            assert!(is_long(result));
            assert_eq!(
                *w_long_get_value(result),
                BigInt::from(2).int_pow(63, None).unwrap()
            );
        }
    }

    #[test]
    fn test_int_pow_negative_exponent() {
        let result = pow(w_int_new(2), w_int_new(-1)).unwrap();
        unsafe {
            assert!(is_float(result));
            assert_eq!(w_float_get_value(result), 0.5);
        }
    }

    #[test]
    fn test_long_pow_negative_exponent_rejects_float_overflow() {
        let huge = BigInt::one().lshift(2000).unwrap();
        let error = pow(w_long_new(huge), w_int_new(-1)).unwrap_err();
        assert_eq!(error.kind, PyErrorKind::OverflowError);
        assert_eq!(error.message_text(), "int too large to convert to float");

        let huge_negative_exponent = BigInt::one().lshift(2000).unwrap().neg();
        let error = pow(w_int_new(1), w_long_new(huge_negative_exponent)).unwrap_err();
        assert_eq!(error.kind, PyErrorKind::OverflowError);
        assert_eq!(error.message_text(), "int too large to convert to float");
    }

    #[test]
    fn test_int_lshift() {
        let result = lshift(w_int_new(1), w_int_new(10)).unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 1024) };
    }

    #[test]
    fn test_int_lshift_overflow() {
        let result = lshift(w_int_new(1), w_int_new(64)).unwrap();
        unsafe {
            assert!(is_long(result));
            assert_eq!(*w_long_get_value(result), BigInt::from(1) << 64);
        }
        let min = lshift(w_int_new(-1), w_int_new(63)).unwrap();
        unsafe { assert_eq!(w_int_get_value(min), i64::MIN) };
        let positive = lshift(w_int_new(1), w_int_new(63)).unwrap();
        unsafe {
            assert!(is_long(positive));
            assert_eq!(*w_long_get_value(positive), BigInt::from(1) << 63);
        }
        let error = lshift(w_int_new(1), w_int_new(i64::MAX)).unwrap_err();
        assert_eq!(error.kind, PyErrorKind::MemoryError);
    }

    #[test]
    fn test_int_rshift() {
        let result = rshift(w_int_new(1024), w_int_new(3)).unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 128) };
    }

    #[test]
    fn test_long_shift_mixed_operands_and_large_counts() {
        unsafe {
            let base = BigInt::one().lshift(70).unwrap();
            let shifted = lshift(w_long_new(base.clone()), w_long_new(BigInt::from(5))).unwrap();
            assert!(is_long(shifted));
            assert_eq!(*w_long_get_value(shifted), base.lshift(5).unwrap());

            let mixed = lshift(w_int_new(3), w_long_new(BigInt::from(65))).unwrap();
            assert!(is_long(mixed));
            assert_eq!(
                *w_long_get_value(mixed),
                BigInt::from(3).lshift(65).unwrap()
            );

            let negative = BigInt::from(-3).lshift(70).unwrap();
            let right = rshift(w_long_new(negative.clone()), w_int_new(69)).unwrap();
            assert!(is_long(right));
            assert_eq!(
                *w_long_get_value(right),
                negative.rshift(69, false).unwrap()
            );

            let enormous_count = BigInt::one().lshift(70).unwrap();
            let saturated =
                rshift(w_long_new(negative), w_long_new(enormous_count.clone())).unwrap();
            assert_eq!(w_int_get_value(saturated), -1);
            let zero = lshift(w_long_new(BigInt::zero()), w_long_new(enormous_count)).unwrap();
            assert!(is_long(zero));
            assert_eq!(w_long_get_value(zero).get_sign(), 0);
        }
    }

    #[test]
    fn test_negative_shift_count() {
        assert!(lshift(w_int_new(1), w_int_new(-1)).is_err());
        assert!(rshift(w_int_new(1), w_int_new(-1)).is_err());
        assert!(lshift(w_long_new(BigInt::from(1)), w_long_new(BigInt::from(-1))).is_err());
        assert!(rshift(w_long_new(BigInt::from(1)), w_long_new(BigInt::from(-1))).is_err());
    }

    #[test]
    fn test_long_repeat_count_bounds() {
        unsafe {
            let huge = BigInt::one().lshift(100).unwrap();
            let negative_error = repeat_count(w_long_new(huge.neg())).unwrap_err();
            assert_eq!(negative_error.kind, PyErrorKind::OverflowError);
            let error = repeat_count(w_long_new(huge)).unwrap_err();
            assert_eq!(error.kind, PyErrorKind::OverflowError);
        }
    }

    #[test]
    fn test_int_bitand() {
        let result = and_(w_int_new(0xFF), w_int_new(0x0F)).unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 0x0F) };
    }

    #[test]
    fn test_int_bitor() {
        let result = or_(w_int_new(0xF0), w_int_new(0x0F)).unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 0xFF) };
    }

    #[test]
    fn test_int_bitxor() {
        let result = xor(w_int_new(0xFF), w_int_new(0x0F)).unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 0xF0) };
    }

    #[test]
    fn test_long_bitand() {
        let a = w_long_new(BigInt::from(i64::MAX) + BigInt::from(1));
        let b = w_int_new(0xFF);
        let result = and_(a, b).unwrap();
        // long & int keeps a W_LongObject even when the result fits (newlong).
        unsafe {
            assert!(is_long(result));
            assert_eq!(*w_long_get_value(result), BigInt::from(0));
        }
    }

    #[test]
    fn test_invert_long() {
        let a = w_long_new(BigInt::from(i64::MAX) + BigInt::from(1));
        let result = invert(a).unwrap();
        unsafe {
            assert!(is_long(result));
            assert_eq!(
                *w_long_get_value(result),
                !(BigInt::from(i64::MAX) + BigInt::from(1))
            );
        }
    }
}
