//! Raw entry points for the builtins the JIT walker folds out of a residual
//! call, and the identity table that names them.
//!
//! A traced `abs(x)` / `hash(x)` / `min(a, b)` reaches the interpreter through
//! `bh_call_fn`, which roots its arguments, resolves the execution context and
//! binds the gateway signature before the builtin body runs at all.  That
//! preamble costs an order of magnitude more than the operation it guards, and
//! it is paid once per loop iteration.  PyPy has no equivalent step: its
//! builtins are RPython, so the tracer walks straight into the body and the
//! result box is virtual.
//!
//! Each helper here is the body of one builtin, restricted to the operand
//! shapes it can answer without running Python and without allocating, and
//! reporting every other direction through a decline channel:
//!
//! * an `i64`-valued helper declines with [`INT_FOLD_DECLINE`];
//! * an `f64`-valued helper declines with NaN, and the trace guards the result
//!   finite;
//! * a reference-valued helper declines with `PY_NULL`, and the trace guards
//!   the result non-null.
//!
//! A decline resumes in the interpreter at the same bytecode, which re-executes
//! the whole call — so a helper may decline for any reason at all, including a
//! result that happens to collide with its own sentinel.  What it may not do is
//! answer where the builtin would have raised, returned something else, or run
//! app-level code.  Nothing here allocates: the result box is emitted as
//! `wrapint` / `wrapfloat` / `newbool` in the trace instead, where the
//! optimizer can keep it virtual.  Every helper must also be a pure function
//! of its arguments: the same operands must always produce the same answer.
//! That is what lets the trace record a call as elidable, so the optimizer may
//! serve a loop's copy from an earlier identical one; a helper whose answer
//! drifted would be answered once and reused forever.  The int and float
//! channels are recorded that way; the reference channel satisfies the same
//! requirement but is still emitted as a plain call, for a reason that belongs
//! with the emission and is recorded at
//! `try_walker_specialize_builtin_fold2`.  That site also answers a pair of
//! exact machine ints without reaching this module at all, by guarding the
//! ordering of the two operands instead of calling anything.

use pyre_object::{PY_NULL, PyObjectRef};

// Every raw helper takes and returns `i64`, never `PyObjectRef`, and casts at
// its own boundary.  That is the residual-call ABI the backends assume: the
// wasm backend lowers an all-Int/Ref call to a direct `call_indirect` typed
// `(i64 x n) -> i64`, and a `PyObjectRef` parameter is an `i32` on wasm32, so
// a helper spelled with pointer arguments traps the moment the compiled trace
// calls it.

/// The `i64` an int-valued raw helper returns to decline its operand.
///
/// A builtin whose real answer is this value declines too and takes the side
/// exit; the interpreter then computes the same answer the slow way.
pub const INT_FOLD_DECLINE: i64 = i64::MIN;

/// How a row's raw helper is called and what its result means.
#[derive(Clone, Copy)]
pub enum BuiltinFoldRaw {
    /// One argument, machine-int result, [`INT_FOLD_DECLINE`] declines.
    Int1(extern "C" fn(i64) -> i64),
    /// One argument, float result, NaN declines.
    Float1(extern "C" fn(i64) -> f64),
    /// Two arguments, reference result, `PY_NULL` declines.
    Ref2(extern "C" fn(i64, i64) -> i64),
}

/// Addresses of the fold helpers whose recorded call descr describes their
/// real wasm ABI, for `majit_backend_wasm::set_faithful_residual_call_addrs`.
///
/// [`BuiltinFoldRaw::Float1`] is the only shape that needs vouching. Its descr
/// is `[Ref] -> Float` (`call_float_typed_with_effect` in the walker's fold
/// arm), which on wasm32 is `(i64) -> f64` -- a mixed signature the backend
/// will not lower on a descr's word alone. `Int1` and `Ref2` are uniformly
/// word-sized and already lower through the arity-keyed family.
///
/// Derived from the table rather than hand-listed, so a fold added later is
/// covered by construction.
pub fn float_fold_helper_addrs() -> Vec<i64> {
    BUILTIN_FOLDS
        .iter()
        .filter_map(|fold| match fold.raw {
            BuiltinFoldRaw::Float1(f) => Some(f as *const () as usize as i64),
            BuiltinFoldRaw::Int1(_) | BuiltinFoldRaw::Ref2(_) => None,
        })
        .collect()
}

impl BuiltinFoldRaw {
    /// Positional argument count this helper answers for.  A call with any
    /// other count is not this row's shape and keeps the residual.
    pub fn arity(&self) -> usize {
        match self {
            Self::Int1(_) | Self::Float1(_) => 1,
            Self::Ref2(_) => 2,
        }
    }
}

/// One folded builtin: the identity that recognizes it and the helper that
/// answers for it.
pub struct BuiltinFold {
    /// Names the row in `PYRE_FBW_SPEC_CENSUS` output.
    pub name: &'static str,
    /// The `BuiltinCodeFn` the module namespace registered under this name.
    /// A rebound global keeps the residual: the identity is the wrapped code,
    /// not the name it is reachable under.
    code_fn: crate::gateway::BuiltinCodeFn,
    pub raw: BuiltinFoldRaw,
}

/// `hash(x)` for the exact scalar types `try_hash_value` already answers
/// without a `__hash__` lookup.  Every one of them is final in place, so an
/// exact instance can only reach the digest `hash_value` computes.
///
/// A NaN `float` is the one arm excluded for its effects rather than its
/// answer: `hash_value` sends it to the identity hash, and a float's identity
/// is its bit pattern widened through `immutable_unique_id`, which wraps a
/// fresh `int`.  The trace emits this call as one that cannot collect, so it
/// carries no gcmap and spills no reference registers; allocating under it
/// would leave the collector blind.  Decline and let the interpreter allocate.
///
/// `hash_value` memoizes a string's digest through `w_str_set_hash`; that write
/// is compatible with elidability because it stores the digest the next call
/// would recompute, and `EffectInfo.__new__` drops `_write_descrs_*` for every
/// `EF_ELIDABLE_*` for exactly this shape.  Every other admitted arm reads a
/// content- or value-determined digest, and the one arm that is neither -- a
/// NaN `float`, which reaches `default_identity_hash_value` -- already
/// declines.
extern "C" fn jit_builtin_hash(obj: i64) -> i64 {
    let obj = obj as PyObjectRef;
    if obj.is_null() {
        return INT_FOLD_DECLINE;
    }
    unsafe {
        if pyre_object::is_exact_type(obj, &pyre_object::FLOAT_TYPE)
            && pyre_object::w_float_get_value(obj).is_nan()
        {
            return INT_FOLD_DECLINE;
        }
        for tp in [
            &pyre_object::STR_TYPE,
            &pyre_object::INT_TYPE,
            &pyre_object::BOOL_TYPE,
            &pyre_object::LONG_TYPE,
            &pyre_object::FLOAT_TYPE,
            &pyre_object::bytesobject::BYTES_TYPE,
        ] {
            if pyre_object::is_exact_type(obj, tp) {
                return crate::builtins::hash_value(obj);
            }
        }
    }
    INT_FOLD_DECLINE
}

/// `ord(x)` for a one-code-point string or a one-byte bytes-like.  Any other
/// length or type is the arm that raises, so it declines.
extern "C" fn jit_builtin_ord(obj: i64) -> i64 {
    let obj = obj as PyObjectRef;
    if obj.is_null() {
        return INT_FOLD_DECLINE;
    }
    unsafe {
        if pyre_object::is_exact_type(obj, &pyre_object::STR_TYPE) {
            if pyre_object::w_str_len(obj) != 1 {
                return INT_FOLD_DECLINE;
            }
            let Some(cp) = pyre_object::w_str_get_wtf8(obj).code_points().next() else {
                return INT_FOLD_DECLINE;
            };
            return cp.to_u32() as i64;
        }
        if pyre_object::bytesobject::is_bytes(obj) {
            let data = pyre_object::bytesobject::bytes_like_data(obj);
            if data.len() != 1 {
                return INT_FOLD_DECLINE;
            }
            return data[0] as i64;
        }
    }
    INT_FOLD_DECLINE
}

/// `abs(x)` for an exact machine int or bool.  `i64::MIN` is the value
/// `builtin_abs` promotes to a long, and it is also the decline sentinel, so
/// both directions leave through the same side exit.
extern "C" fn jit_builtin_abs_int(obj: i64) -> i64 {
    let obj = obj as PyObjectRef;
    if obj.is_null() {
        return INT_FOLD_DECLINE;
    }
    unsafe {
        if pyre_object::is_exact_type(obj, &pyre_object::BOOL_TYPE) {
            return pyre_object::w_bool_get_value(obj) as i64;
        }
        // `is_int` reads `ob_type`, which a subclass instance shares with the
        // builtin, so it alone would answer for one whose `__abs__` override
        // the fold has no operand-class guard to notice.  `is_exact_type`
        // reads `w_class`, which the subclass retags -- and it also excludes
        // `bool`, whose own arm sits above.  Neither test implies the other:
        // `is_exact_type` alone would admit a bigint, whose `*mut BigInt` sits
        // where `intval` does.
        if pyre_object::is_exact_type(obj, &pyre_object::INT_TYPE) && pyre_object::is_int(obj) {
            return pyre_object::w_int_get_value(obj)
                .checked_abs()
                .unwrap_or(INT_FOLD_DECLINE);
        }
    }
    INT_FOLD_DECLINE
}

/// `abs(x)` for an exact float.  A NaN operand declines through the same
/// channel a NaN result would, which is what the trace's finite guard reads.
extern "C" fn jit_builtin_abs_float(obj: i64) -> f64 {
    let obj = obj as PyObjectRef;
    if obj.is_null() {
        return f64::NAN;
    }
    unsafe {
        if pyre_object::is_exact_type(obj, &pyre_object::FLOAT_TYPE) {
            return pyre_object::w_float_get_value(obj).abs();
        }
    }
    f64::NAN
}

/// The comparison `builtin_min` / `builtin_max` reach for two positional
/// arguments, restricted to the operand pairs that compare without a
/// `__lt__`/`__gt__` dispatch.  Returns the winning *object*, so ties keep the
/// argument the iteration order keeps.
fn compare_pair(a: PyObjectRef, b: PyObjectRef, want_second_wins: bool) -> Option<bool> {
    unsafe {
        let exact = |obj, tp| pyre_object::is_exact_type(obj, tp);
        // `is_exact_type` answers on `w_class`, and `w_long_from_raw` wires a
        // bigint's `w_class` to `int`'s so that `type(x) is int` holds for one
        // — so the exact-`int` gate alone admits a `W_LongObject`, whose
        // `value: *mut BigInt` sits exactly where `W_IntObject` keeps
        // `intval`.  Reading one as the other would compare heap addresses
        // instead of numbers.  `is_int` reads `ob_type`, which still separates
        // the two layouts; this is the conjunct the dict's builtin-key test
        // uses for the same reason.
        let machine_int = |obj| exact(obj, &pyre_object::INT_TYPE) && pyre_object::is_int(obj);
        if machine_int(a) && machine_int(b) {
            let (a, b) = (
                pyre_object::w_int_get_value(a),
                pyre_object::w_int_get_value(b),
            );
            return Some(if want_second_wins { b > a } else { b < a });
        }
        if exact(a, &pyre_object::FLOAT_TYPE) && exact(b, &pyre_object::FLOAT_TYPE) {
            let (a, b) = (
                pyre_object::w_float_get_value(a),
                pyre_object::w_float_get_value(b),
            );
            // A NaN operand makes every comparison false, and which argument
            // that leaves standing depends on the scan order; decline instead
            // of encoding it here.
            if a.is_nan() || b.is_nan() {
                return None;
            }
            return Some(if want_second_wins { b > a } else { b < a });
        }
    }
    None
}

/// `min(a, b)` — `builtin_min`'s two-positional form: keep the first argument
/// unless the second compares strictly smaller.
///
/// The helper returns one of its own arguments after comparing two exact
/// machine ints or two non-NaN floats, so the answer is fixed by the pair, as
/// the table requires.  A NaN operand, whose winner would depend on the scan
/// order, already declines.
extern "C" fn jit_builtin_min2(a: i64, b: i64) -> i64 {
    let (a, b) = (a as PyObjectRef, b as PyObjectRef);
    if a.is_null() || b.is_null() {
        return PY_NULL as i64;
    }
    match compare_pair(a, b, false) {
        Some(true) => b as i64,
        Some(false) => a as i64,
        None => PY_NULL as i64,
    }
}

/// `max(a, b)` — the `builtin_max` twin of [`jit_builtin_min2`].
///
/// Like [`jit_builtin_min2`], this helper returns one of its own arguments
/// after an exact machine-int or non-NaN float comparison, so the answer is
/// fixed by the pair.  A NaN operand already declines instead of preserving a
/// scan-order-dependent winner.
extern "C" fn jit_builtin_max2(a: i64, b: i64) -> i64 {
    let (a, b) = (a as PyObjectRef, b as PyObjectRef);
    if a.is_null() || b.is_null() {
        return PY_NULL as i64;
    }
    match compare_pair(a, b, true) {
        Some(true) => b as i64,
        Some(false) => a as i64,
        None => PY_NULL as i64,
    }
}

/// Every folded builtin.  Adding a row adds a fold: the walker looks the
/// callable up here and emits the channel the row names.
static BUILTIN_FOLDS: &[BuiltinFold] = &[
    BuiltinFold {
        name: "hash",
        code_fn: crate::builtins::builtin_hash,
        raw: BuiltinFoldRaw::Int1(jit_builtin_hash),
    },
    BuiltinFold {
        name: "ord",
        code_fn: crate::builtins::builtin_ord,
        raw: BuiltinFoldRaw::Int1(jit_builtin_ord),
    },
    BuiltinFold {
        name: "abs",
        code_fn: crate::builtins::__majit_wrap_builtin_abs,
        raw: BuiltinFoldRaw::Int1(jit_builtin_abs_int),
    },
    BuiltinFold {
        name: "abs",
        code_fn: crate::builtins::__majit_wrap_builtin_abs,
        raw: BuiltinFoldRaw::Float1(jit_builtin_abs_float),
    },
    BuiltinFold {
        name: "min",
        code_fn: crate::builtins::builtin_min,
        raw: BuiltinFoldRaw::Ref2(jit_builtin_min2),
    },
    BuiltinFold {
        name: "max",
        code_fn: crate::builtins::builtin_max,
        raw: BuiltinFoldRaw::Ref2(jit_builtin_max2),
    },
];

/// The rows `callable` could be, in table order, for a call carrying `argc`
/// positional arguments.
///
/// One builtin may hold several rows — `abs` has an int one and a float one —
/// and the walker takes the first whose helper answers for the recorded
/// operands.
pub fn builtin_folds_for(
    callable: PyObjectRef,
    argc: usize,
) -> impl Iterator<Item = &'static BuiltinFold> {
    let code_fn = unsafe { builtin_code_fn_of(callable) };
    BUILTIN_FOLDS.iter().filter(move |fold| {
        fold.raw.arity() == argc
            && code_fn.is_some_and(|found| unsafe {
                crate::gateway::builtin_code_fn_eq(found, fold.code_fn)
            })
    })
}

/// The `BuiltinCodeFn` a callable's wrapped code holds, or `None` when the
/// callable is not a builtin-code function at all.
unsafe fn builtin_code_fn_of(callable: PyObjectRef) -> Option<crate::gateway::BuiltinCodeFn> {
    unsafe {
        if callable.is_null() || !crate::is_function(callable) {
            return None;
        }
        let code = crate::function_get_code(callable) as PyObjectRef;
        if code.is_null() || !crate::gateway::is_builtin_code(code) {
            return None;
        }
        Some(crate::gateway::builtin_code_get(code))
    }
}

/// Row count, for the walker's census to size its per-row tally.
pub const BUILTIN_FOLD_COUNT: usize = BUILTIN_FOLDS.len();
