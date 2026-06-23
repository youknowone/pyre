//! Operation execution for the blackhole interpreter.
//!
//! Mirrors RPython's `executor.py`: executes individual JIT IR operations
//! by dispatching on the opcode and computing the result.

use majit_ir::{OpCode, OpRef};

/// rpython/jit/metainterp/executor.py:524-528 `execute_varargs(cpu, metainterp, opnum, argboxes, descr)`.
///
/// ```python
/// def execute_varargs(cpu, metainterp, opnum, argboxes, descr):
///     # only for opnums with a variable arity (calls, typically)
///     check_descr(descr)
///     func = get_execute_function(opnum, -1, True)
///     return func(cpu, metainterp, argboxes, descr)
/// ```
///
/// For CALL_* opcodes, `func` ultimately calls `cpu.bh_call_*(funcaddr,
/// args)`.  Pyre routes every arm through `dispatch::call_int_function`
/// / `dispatch::call_void_function` using the concrete values carried
/// alongside each typed argbox.  The Float arm shares the i64-return
/// ABI: helper concrete pointers built by `#[jit_module]` pre-pack the
/// f64 result via `f64::to_bits` (majit-macros/src/lib.rs:194), and
/// callers recover the f64 with `f64::from_bits(resvalue as u64)`
/// (pyjitpl/mod.rs:8901-8902) — bit-identical to the BC_CALL_FLOAT
/// family in blackhole.rs:2349-2371 which uses the same convention.
/// `argboxes[0]` is the funcbox (carrying the function pointer in its
/// `i64` slot) and the remaining slots are the typed call arguments.
///
/// `metainterp` mirrors RPython's `self` parameter: helper-side
/// exceptions published on the `BH_LAST_EXC_VALUE` thread-local seam
/// (the convention `bh_call_fn_impl` and friends use) are transcribed
/// onto `metainterp.last_exc_value` before returning, the same way
/// RPython's cpu auto-publishes onto `cpu.last_exc_value`.  Pyre has
/// no separate `cpu` object; the metainterp owns the field directly.
///
/// Returns the concrete result value as an `i64` — for void-returning
/// calls returns `0` (caller ignores it); for float-returning calls
/// the i64 carries the f64 bits via `f64::to_bits` and the caller
/// must unpack with `f64::from_bits(resvalue as u64)`.  When the
/// helper raises (BH_LAST_EXC_VALUE seam fires), the post-call hook
/// transcribes onto `metainterp.last_exc_value` and overrides the
/// result with `0` — matching `executor.py:52-78`'s neutral-zero
/// behavior across INT, REF (NULL), FLOAT (longlong.ZEROF), and VOID.
// executor.py:188-190
//
//     def do_getfield_gc_i(cpu, _, structbox, fielddescr):
//         struct = structbox.getref_base()
//         return cpu.bh_getfield_gc_i(struct, fielddescr)
//
// `structbox.getref_base()` returns the GCREF the box carries; pyre's
// flat box analog is the concrete `i64` shadow paired with the symbolic
// OpRef (the (OpRef, i64) tuple returned by `read_ref_reg`). The
// caller has already projected `structbox.getref_base()` and passes it
// as `structbox` here so the function shape matches RPython 1:1.
pub fn do_getfield_gc_i(
    cpu: &dyn majit_backend::Backend,
    _metainterp: (),
    structbox: i64,
    fielddescr: &majit_translate::jitcode::BhDescr,
) -> i64 {
    let struct_ = structbox;
    cpu.bh_getfield_gc_i(struct_, fielddescr)
}

// executor.py:192-194
pub fn do_getfield_gc_r(
    cpu: &dyn majit_backend::Backend,
    _metainterp: (),
    structbox: i64,
    fielddescr: &majit_translate::jitcode::BhDescr,
) -> majit_ir::GcRef {
    let struct_ = structbox;
    cpu.bh_getfield_gc_r(struct_, fielddescr)
}

// executor.py:196-198
pub fn do_getfield_gc_f(
    cpu: &dyn majit_backend::Backend,
    _metainterp: (),
    structbox: i64,
    fielddescr: &majit_translate::jitcode::BhDescr,
) -> f64 {
    let struct_ = structbox;
    cpu.bh_getfield_gc_f(struct_, fielddescr)
}

// executor.py:206-212 do_getarrayitem_gc_{i,r,f}: project box → gcref +
// concrete index, dispatch to `cpu.bh_getarrayitem_gc_*`.  Pyre's flat
// box analog passes `(array, index)` as plain `i64` values projected
// from the symbolic OpRefs by the caller.
pub fn do_getarrayitem_gc_i(
    cpu: &dyn majit_backend::Backend,
    _metainterp: (),
    arraybox: i64,
    indexbox: i64,
    arraydescr: &majit_translate::jitcode::BhDescr,
) -> i64 {
    cpu.bh_getarrayitem_gc_i(arraybox, indexbox, arraydescr)
}

pub fn do_getarrayitem_gc_r(
    cpu: &dyn majit_backend::Backend,
    _metainterp: (),
    arraybox: i64,
    indexbox: i64,
    arraydescr: &majit_translate::jitcode::BhDescr,
) -> majit_ir::GcRef {
    cpu.bh_getarrayitem_gc_r(arraybox, indexbox, arraydescr)
}

pub fn do_getarrayitem_gc_f(
    cpu: &dyn majit_backend::Backend,
    _metainterp: (),
    arraybox: i64,
    indexbox: i64,
    arraydescr: &majit_translate::jitcode::BhDescr,
) -> f64 {
    cpu.bh_getarrayitem_gc_f(arraybox, indexbox, arraydescr)
}

// blackhole.py:1370 bhimpl_arraylen_gc(cpu, array, arraydescr): direct
// `cpu.bh_arraylen_gc(array, arraydescr)`.  RPython has no explicit
// `do_arraylen_gc` in executor.py; the dispatch goes through the
// blackhole fallback wrapper.  Pyre exposes it here in `executor.rs`
// for `TraceCtx::arraylen_sanity_load` to consume directly without
// importing the blackhole module.  Array is projected to `i64` from
// the BoxRef (`arraybox.getref_base()` analog).
pub fn do_arraylen_gc(
    cpu: &dyn majit_backend::Backend,
    _metainterp: (),
    arraybox: i64,
    arraydescr: &majit_translate::jitcode::BhDescr,
) -> i64 {
    cpu.bh_arraylen_gc(arraybox, arraydescr)
}

// executor.py:132 do_getarrayitem_raw_{i,f}: project arraybox → int
// (raw pointer), dispatch to `cpu.bh_getarrayitem_raw_*`.  Distinct
// from `do_getarrayitem_gc_*` (executor.py:117) which projects via
// `arraybox.getref_base()` — raw arrays carry their pointer as a
// plain integer.  Pyre passes the raw pointer as `i64`.
pub fn do_getarrayitem_raw_i(
    cpu: &dyn majit_backend::Backend,
    _metainterp: (),
    arraybox: i64,
    indexbox: i64,
    arraydescr: &majit_translate::jitcode::BhDescr,
) -> i64 {
    cpu.bh_getarrayitem_raw_i(arraybox, indexbox, arraydescr)
}

pub fn do_getarrayitem_raw_f(
    cpu: &dyn majit_backend::Backend,
    _metainterp: (),
    arraybox: i64,
    indexbox: i64,
    arraydescr: &majit_translate::jitcode::BhDescr,
) -> f64 {
    cpu.bh_getarrayitem_raw_f(arraybox, indexbox, arraydescr)
}

// executor.py:200 do_getfield_raw_{i,r,f}: project structbox → int
// (raw pointer via `structbox.getint()`), dispatch to
// `cpu.bh_getfield_raw_*`.  Distinct from `do_getfield_gc_*`
// (executor.py:188) which projects via `structbox.getref_base()` —
// raw structs carry their pointer as a plain integer.  Pyre's caller
// projects the symbolic OpRef carrier to `i64` directly.
pub fn do_getfield_raw_i(
    cpu: &dyn majit_backend::Backend,
    _metainterp: (),
    structbox: i64,
    fielddescr: &majit_translate::jitcode::BhDescr,
) -> i64 {
    cpu.bh_getfield_raw_i(structbox, fielddescr)
}

pub fn do_getfield_raw_r(
    cpu: &dyn majit_backend::Backend,
    _metainterp: (),
    structbox: i64,
    fielddescr: &majit_translate::jitcode::BhDescr,
) -> majit_ir::GcRef {
    cpu.bh_getfield_raw_r(structbox, fielddescr)
}

pub fn do_getfield_raw_f(
    cpu: &dyn majit_backend::Backend,
    _metainterp: (),
    structbox: i64,
    fielddescr: &majit_translate::jitcode::BhDescr,
) -> f64 {
    cpu.bh_getfield_raw_f(structbox, fielddescr)
}

pub fn execute_varargs<M: Clone>(
    metainterp: &mut crate::pyjitpl::MetaInterp<M>,
    opnum: OpCode,
    argboxes: &[(crate::jitcode::JitArgKind, OpRef, i64)],
    descr: &dyn majit_ir::descr::CallDescr,
) -> i64 {
    debug_assert!(opnum.is_call(), "execute_varargs requires a call opcode");
    // RPython's `cpu` parameter is the seam through which helper-side
    // exceptions reach `metainterp.last_exc_value` (`cpu.bh_call_*`
    // writes onto `cpu.last_exception` and the metainterp's
    // post-execute hook copies it).  Pyre has no separate cpu; we use
    // the `BH_LAST_EXC_VALUE` thread-local that production helpers
    // (`bh_call_fn_impl` etc.) publish on, mirroring the same convention
    // every blackhole.rs CALL_* arm uses (blackhole.rs:2270-2392).
    // Clear before dispatch so a stale value from a prior call cannot
    // bleed into this one.
    crate::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(0));
    // Inner closure carries every existing return path so the outer
    // wrapper can run BH_LAST_EXC_VALUE transcription regardless of
    // which arm fires.
    let result = (|| -> i64 {
        // COND_CALL / COND_CALL_VALUE_* layout (pyjitpl.py:2128-2151):
        //   argboxes[0] = condbox / valuebox
        //   argboxes[1] = funcbox
        //   argboxes[2..] = call args
        // RPython's executor handles cond-call dispatch via a per-opcode
        // execute function (`do_cond_call_*`); pyre inlines the same
        // semantics here.
        //
        // Two distinct shapes (blackhole.py:1257-1276):
        //   bhimpl_conditional_call_ir_v(condition, func, ...):
        //       if condition: cpu.bh_call_v(func, ...)        # void
        //   bhimpl_conditional_call_value_ir_{i,r}(value, func, ...):
        //       if value == 0: value = cpu.bh_call_*(func, ...)
        //       return value
        if matches!(opnum, OpCode::CondCallN) {
            debug_assert!(
                argboxes.len() >= 2,
                "COND_CALL_N requires [condbox, funcbox, *args]",
            );
            let cond = argboxes[0].2;
            if cond == 0 {
                // condition false → skip the call.
                return 0;
            }
            let func_ptr = argboxes[1].2 as *const ();
            let concrete_args: Vec<i64> = argboxes[2..].iter().map(|(_, _, c)| *c).collect();
            crate::pyjitpl::call_void_function(func_ptr, &concrete_args);
            return 0;
        }
        if matches!(opnum, OpCode::CondCallValueI | OpCode::CondCallValueR) {
            debug_assert!(
                argboxes.len() >= 2,
                "COND_CALL_VALUE_* requires [valuebox, funcbox, *args]",
            );
            let value = argboxes[0].2;
            if value != 0 {
                // blackhole.py:1267 / 1274: nonzero `value` short-circuits
                // and returns the existing value without calling.
                return value;
            }
            // value == 0 → call and return the call's result.
            let func_ptr = argboxes[1].2 as *const ();
            let concrete_args: Vec<i64> = argboxes[2..].iter().map(|(_, _, c)| *c).collect();
            return crate::pyjitpl::call_int_function(func_ptr, &concrete_args);
        }
        debug_assert!(
            !argboxes.is_empty(),
            "execute_varargs: argboxes must include funcbox at slot 0",
        );
        let func_ptr = argboxes[0].2 as *const ();
        let concrete_args: Vec<i64> = argboxes[1..].iter().map(|(_, _, c)| *c).collect();
        match descr.result_type() {
            // RPython dispatches Int and Ref through the same backend
            // primitive cpu.bh_call_i (returns i64); pyre's
            // call_int_function does the same — Ref is bit-identical to Int
            // at the ABI level.
            majit_ir::Type::Int | majit_ir::Type::Ref => {
                crate::pyjitpl::call_int_function(func_ptr, &concrete_args)
            }
            majit_ir::Type::Void => {
                crate::pyjitpl::call_void_function(func_ptr, &concrete_args);
                0
            }
            majit_ir::Type::Float => {
                // pyjitpl.py:2119 — CALL_F dispatches through cpu.bh_call_f.
                // Caller-contract: every path that reaches this arm carries
                // `funcbox.2` as a hand-written or `#[jit_module]`-generated
                // function pointer with i64-return ABI:
                //   * `do_recursive_call` (mod.rs:11148-11152) sets funcbox.2
                //     to `targetjitdriver_sd.portal_runner_adr`.  Pyre's
                //     portal entry is `bh_portal_runner(all_i, all_r, all_f)
                //     -> i64` (pyre-jit/src/call_jit.rs:467); it never
                //     declares an f64 return.
                //   * `#[jit_module]` (majit-macros/src/lib.rs:267) emits a
                //     Float helper's `concrete_ptr` as `extern "C" fn(...)
                //     -> i64` with the f64 result pre-packed via
                //     `f64::to_bits`; the f64-ABI `trace_ptr` is consumed
                //     only by pyre-jit-trace's `TraceCtx::call_may_force_*`
                //     family, which has its own seam and never reaches
                //     this arm.
                // Therefore route through `call_int_function` and let the
                // caller recover the f64 via `f64::from_bits` when needed
                // — bit-identical to blackhole.rs:2349-2371 BC_CALL_FLOAT
                // family which makes the same ABI choice for the same
                // reason (`registers_f` is an i64-carrier mirroring
                // RPython's `longlong.ZEROF` packing).
                crate::pyjitpl::call_int_function(func_ptr, &concrete_args)
            }
        }
    })();
    // Mirror RPython's executor.py:52-78 post-call exception flow:
    // each `cpu.bh_call_*` arm wraps the call in `try: ... except
    // Exception as e: metainterp.execute_raised(e); result = ZERO`.
    // Pyre observes the same condition through the BH_LAST_EXC_VALUE
    // thread-local seam.  Two pieces have to fire together:
    //   1. `metainterp.execute_raised(bh_exc, constant=False)` — sets
    //      `last_exc_value` AND clears `class_of_last_exc_is_const`,
    //      so a stale `True` from a prior GUARD_EXCEPTION cannot make
    //      `handle_possible_exception` treat the new exception's
    //      class as constant (pyjitpl.py:2745-2755).
    //   2. Override the returned value with the type's neutral zero —
    //      `make_result_of_lastop` (pyjitpl/mod.rs:8893) snapshots the
    //      concrete result *before* `handle_possible_exception` runs,
    //      so leaving the helper's return value in place can pin a
    //      garbage value into the resume snapshot.  `i64 == 0` covers
    //      INT=0, REF=NULL, and FLOAT=longlong.ZEROF (0.0_f64.to_bits()
    //      as i64 == 0); VOID callers ignore the slot.
    let bh_exc = crate::blackhole::BH_LAST_EXC_VALUE.with(|c| {
        let v = c.get();
        c.set(0);
        v
    });
    if bh_exc != 0 {
        metainterp.execute_raised(bh_exc, false);
        return 0;
    }
    result
}

/// executor.py:555 `execute_nonspec_const` for binary integer opcodes.
///
/// Returns the folded `i64` result when the operation is recognized and
/// the result is well-defined; returns `None` to abort folding when:
///   * the opcode is not a recognized binary int op
///   * an OVF arithmetic op (IntAddOvf/SubOvf/MulOvf) overflows —
///     RPython's `do_int_add_ovf` then hits
///     `assert metainterp is not None` (executor.py:287) which
///     AssertionErrors in the `constant_fold` path (metainterp=None);
///     pyre prefers the softer `None` skip so the op stays in the
///     trace and the runtime guard fires
///   * a shift count is outside `0..64` (mirrors
///     `blackhole.py:258 check_shift_count`)
///   * IntFloorDiv / IntMod with a zero divisor
///
/// Non-OVF IntAdd/IntSub/IntMul match `bhimpl_int_add/_sub/_mul`
/// (`blackhole.py:459-468`) which compute `intmask(a + b)` — i.e.
/// wrapping i64 arithmetic. Earlier `checked_*` use here would have
/// aborted the fold on a representable wrapping result.
///
/// Mirrors the `do_int_*` entries at executor.py:279-309 (OVF) +
/// `EXECUTE_BY_NUM_ARGS` binary-int rows (the unrolled dispatch table
/// generated at executor.py:495-498).
pub fn execute_binary_int_const(opcode: OpCode, a: i64, b: i64) -> Option<i64> {
    let result = match opcode {
        OpCode::IntAdd => a.wrapping_add(b),
        OpCode::IntSub => a.wrapping_sub(b),
        OpCode::IntMul => a.wrapping_mul(b),
        OpCode::IntAddOvf => a.checked_add(b)?,
        OpCode::IntSubOvf => a.checked_sub(b)?,
        OpCode::IntMulOvf => a.checked_mul(b)?,
        OpCode::IntAnd => a & b,
        OpCode::IntOr => a | b,
        OpCode::IntXor => a ^ b,
        OpCode::IntLshift if b >= 0 && b < 64 => a << b,
        OpCode::IntRshift if b >= 0 && b < 64 => a >> b,
        OpCode::UintRshift if b >= 0 && b < 64 => (a as u64 >> b as u64) as i64,
        OpCode::IntLt => (a < b) as i64,
        OpCode::IntLe => (a <= b) as i64,
        OpCode::IntGt => (a > b) as i64,
        OpCode::IntGe => (a >= b) as i64,
        OpCode::IntEq => (a == b) as i64,
        OpCode::IntNe => (a != b) as i64,
        OpCode::UintLt => ((a as u64) < (b as u64)) as i64,
        OpCode::UintLe => ((a as u64) <= (b as u64)) as i64,
        OpCode::UintGe => ((a as u64) >= (b as u64)) as i64,
        OpCode::UintGt => ((a as u64) > (b as u64)) as i64,
        OpCode::IntFloorDiv if b != 0 => {
            let (q, r) = (a / b, a % b);
            if (r != 0) && ((r ^ b) < 0) { q - 1 } else { q }
        }
        OpCode::IntMod if b != 0 => {
            let r = a % b;
            if (r != 0) && ((r ^ b) < 0) { r + b } else { r }
        }
        OpCode::IntSignext if b >= 1 && b <= 8 => {
            // blackhole.py:568 bhimpl_int_signext → support.py:30 int_signext.
            crate::support::int_signext(a, b)
        }
        OpCode::UintMulHigh => {
            // blackhole.py bhimpl_uint_mul_high — high 64 of (a as u64) * (b as u64).
            (((a as u64) as u128 * (b as u64) as u128) >> 64) as i64
        }
        _ => return None,
    };
    Some(result)
}

/// executor.py:495-498 ptr-compare row of EXECUTE_BY_NUM_ARGS.
/// Mirrors blackhole.py bhimpl_ptr_eq/_ne and instance_ptr_eq/_ne —
/// straight pointer identity once both args are constant references.
pub fn execute_ptr_compare_const(opcode: OpCode, a: usize, b: usize) -> Option<i64> {
    let result = match opcode {
        OpCode::PtrEq | OpCode::InstancePtrEq => a == b,
        OpCode::PtrNe | OpCode::InstancePtrNe => a != b,
        _ => return None,
    };
    Some(result as i64)
}

/// executor.py:495-498 unary-int row of EXECUTE_BY_NUM_ARGS, the
/// 1-arg variant. Mirrors blackhole.py:528-566 bhimpl_int_neg /
/// _invert / _is_zero / _is_true / _force_ge_zero.
///
/// Returns `None` for unrecognized opcodes so the caller can fall
/// through to other dispatch paths (executor.py:559's `assert False`
/// shape is reserved for non-matching opnums via the unrolled match).
pub fn execute_unary_int_const(opcode: OpCode, a: i64) -> Option<i64> {
    let result = match opcode {
        OpCode::IntNeg => a.wrapping_neg(),
        OpCode::IntInvert => !a,
        OpCode::IntIsZero => (a == 0) as i64,
        OpCode::IntIsTrue => (a != 0) as i64,
        OpCode::IntForceGeZero => a.max(0),
        _ => return None,
    };
    Some(result)
}

/// executor.py:495-498 unary-float row mirrors blackhole.py float
/// unops: bhimpl_float_neg / _abs.
pub fn execute_unary_float_const(opcode: OpCode, a: f64) -> Option<f64> {
    let result = match opcode {
        OpCode::FloatNeg => -a,
        OpCode::FloatAbs => a.abs(),
        _ => return None,
    };
    Some(result)
}

/// executor.py:495-498 binary-float row. Float arithmetic + comparisons
/// (comparisons return bool wrapped as 0/1 in the caller). Mirrors
/// blackhole.py bhimpl_float_add/_sub/_mul/_truediv (`:697-718`).
/// FLOAT_TRUEDIV with `b == 0.0` is NOT folded — see upstream
/// `test_optimizebasic.test_float_division_by_multiplication` which
/// preserves `float_truediv(f, 0.0)` in the optimized loop rather than
/// freezing the IEEE inf/nan constant. The runtime executor still
/// performs `a / b` per `blackhole.py:717` (translated C semantics);
/// only trace-time folding is suppressed.
pub fn execute_binary_float_const(opcode: OpCode, a: f64, b: f64) -> Option<f64> {
    let result = match opcode {
        OpCode::FloatAdd => a + b,
        OpCode::FloatSub => a - b,
        OpCode::FloatMul => a * b,
        OpCode::FloatTrueDiv if b != 0.0 => a / b,
        _ => return None,
    };
    Some(result)
}

/// executor.py:495-498 float→bool row. Mirrors blackhole.py
/// bhimpl_float_lt/_le/_eq/_ne/_gt/_ge.
pub fn execute_float_compare_const(opcode: OpCode, a: f64, b: f64) -> Option<i64> {
    let result = match opcode {
        OpCode::FloatLt => a < b,
        OpCode::FloatLe => a <= b,
        OpCode::FloatEq => a == b,
        OpCode::FloatNe => a != b,
        OpCode::FloatGt => a > b,
        OpCode::FloatGe => a >= b,
        _ => return None,
    };
    Some(result as i64)
}

/// `executor.py:555 execute_nonspec_const` delegates to `_execute_arglist`,
/// which raises `NotImplementedError` at `:610` when no helper is registered
/// for the opnum. RPython's `optimizer.py:810 constant_fold` does not catch
/// that exception; it propagates to the caller. Pyre encodes the same
/// dispatch distinction at the type level:
///   * `Err(NotImplemented)` — no helper claimed the opnum (terminal
///     fall-through below), mirroring the upstream raise.
///   * `Ok(None)` — a helper claimed the opnum but declined to fold
///     (null gcref, unsupported field size, etc.); pyre keeps these
///     as `Ok(None)` so the caller can still see "helper ran, fold
///     skipped" distinctly from "no helper".
///   * `Ok(Some(value))` — successful fold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NotImplemented;

/// executor.py:555 `execute_nonspec_const` free function — the
/// generic opnum dispatch invoked by `optimizer.py:810 constant_fold`
/// once every arg has been resolved to a `Const*` via
/// `get_constant_box`. Mirrors the RPython structure:
///
/// ```python
/// def execute_nonspec_const(cpu, metainterp, opnum, argboxes,
///                           descr=None, type='i'):
///     for num in unrolled_range:
///         if num == opnum:
///             return wrap_constant(_execute_arglist(cpu, metainterp, num,
///                                                    argboxes, descr))
///     assert False
/// ```
///
/// `_execute_arglist` (executor.py:563-610) selects
/// `EXECUTE_BY_NUM_ARGS[arity, withdescr][opnum]` and raises
/// `NotImplementedError` (`:610`) only when no function is registered
/// for the opnum. Pyre returns [`Err(NotImplemented)`](NotImplemented)
/// for that case and `Ok(None)` for helper-internal "decline to fold"
/// outcomes (e.g. null gcref, unsupported field size).
///
/// `_type` is accepted for signature parity with RPython's `type`
/// parameter; it is not consulted because the Value variant in
/// `argboxes` already determines the result type via the helper that
/// fires.
pub fn execute_nonspec_const(
    cpu: &dyn crate::cpu::Cpu,
    opnum: OpCode,
    argboxes: &[majit_ir::Value],
    descr: Option<&majit_ir::descr::DescrRef>,
    _type: majit_ir::Type,
) -> Result<Option<majit_ir::Value>, NotImplemented> {
    use majit_ir::Value;
    let arity = argboxes.len();

    // ── arity == 1 row of EXECUTE_BY_NUM_ARGS ──
    if arity == 1 {
        let a = argboxes[0];
        // executor.py:314-321 `do_same_as_i/r/f` — identity fold for
        // SAME_AS_I/R/F (`bhimpl_int_same_as` etc., `blackhole.py:455`).
        match opnum {
            OpCode::SameAsI | OpCode::SameAsR | OpCode::SameAsF => return Ok(Some(a)),
            _ => {}
        }
        if let Value::Int(i) = a {
            if let Some(folded) = execute_unary_int_const(opnum, i) {
                return Ok(Some(Value::Int(folded)));
            }
        }
        if let Value::Float(f) = a {
            if let Some(folded) = execute_unary_float_const(opnum, f) {
                return Ok(Some(Value::Float(folded)));
            }
        }
        if let Some(folded) = execute_cast_const(opnum, a) {
            return Ok(Some(folded));
        }
        // GETFIELD_GC_PURE_I/R/F — withdescr arity-1.
        // `optimizer.py:829-832 protect_speculative_operation` has
        // validated the gcref is non-null and of a valid type for
        // `fielddescr.parent_descr` (`llmodel.py:555-567`); the
        // dereference below cannot fault.  Unsupported field_size is
        // `llmodel.py:478 read_int_at_mem`'s NotImplementedError and
        // would propagate to crash upstream — pyre matches via
        // fail-loud `unreachable!()`.
        if let (Value::Ref(struct_ref), Some(d)) = (a, descr) {
            if let Some(fd) = d.as_field_descr() {
                return Ok(Some(match opnum {
                    OpCode::GetfieldGcPureI => match fd.field_size() {
                        1 | 2 | 4 | 8 => Value::Int(cpu.bh_getfield_gc_i(struct_ref.0, fd)),
                        sz => unreachable!(
                            "GETFIELD_GC_PURE_I: unsupported field_size {} \
                             (llmodel.py:478 raises NotImplementedError)",
                            sz
                        ),
                    },
                    OpCode::GetfieldGcPureR => Value::Ref(cpu.bh_getfield_gc_r(struct_ref.0, fd)),
                    OpCode::GetfieldGcPureF => Value::Float(cpu.bh_getfield_gc_f(struct_ref.0, fd)),
                    _ => return Err(NotImplemented),
                }));
            }
        }
        // ARRAYLEN_GC — withdescr arity-1.
        // `executor.py:do_arraylen_gc` → `cpu.bh_arraylen_gc(array, ad)`.
        // `protect_speculative_array` validated the gcref + tid; the
        // arraydescr is expected to carry a `len_descr` (registered
        // by the backend's array metadata), so a missing one is a
        // bug — fail-loud per `llmodel.py:585 assert isinstance(...)`.
        if let (Value::Ref(array), Some(d)) = (a, descr) {
            if let Some(ad) = d.as_array_descr() {
                if opnum == OpCode::ArraylenGc {
                    let len = cpu.bh_arraylen_gc(array, ad).expect(
                        "ARRAYLEN_GC: arraydescr missing len_descr (llmodel.py:585 asserts)",
                    );
                    return Ok(Some(Value::Int(len)));
                }
            }
        }
        // STRLEN / UNICODELEN — by the time the fold is entered,
        // `protect_speculative_string / _unicode` has already validated
        // `str_descr / unicode_descr` is registered (under
        // `supports_guard_gc_type == true`; the false case is gated
        // out at `OptContext::protect_speculative_operation`).
        if let Value::Ref(s) = a {
            match opnum {
                OpCode::Strlen => {
                    let len = cpu
                        .bh_strlen(s)
                        .expect("STRLEN: str_descr unregistered after protect_speculative_string");
                    return Ok(Some(Value::Int(len)));
                }
                OpCode::Unicodelen => {
                    let len = cpu.bh_unicodelen(s).expect(
                        "UNICODELEN: unicode_descr unregistered after protect_speculative_unicode",
                    );
                    return Ok(Some(Value::Int(len)));
                }
                _ => {}
            }
        }
    }

    // ── arity == 2 row of EXECUTE_BY_NUM_ARGS ──
    if arity == 2 {
        if let (Value::Int(a), Value::Int(b)) = (argboxes[0], argboxes[1]) {
            if let Some(folded) = execute_binary_int_const(opnum, a, b) {
                return Ok(Some(Value::Int(folded)));
            }
        }
        if let (Value::Float(a), Value::Float(b)) = (argboxes[0], argboxes[1]) {
            if let Some(folded) = execute_binary_float_const(opnum, a, b) {
                return Ok(Some(Value::Float(folded)));
            }
            if let Some(folded) = execute_float_compare_const(opnum, a, b) {
                return Ok(Some(Value::Int(folded)));
            }
        }
        if let (Value::Ref(a), Value::Ref(b)) = (argboxes[0], argboxes[1]) {
            if let Some(folded) = execute_ptr_compare_const(opnum, a.0, b.0) {
                return Ok(Some(Value::Int(folded)));
            }
        }
        // GETARRAYITEM_GC_PURE_I/R/F — withdescr arity-2 (array, index).
        // `executor.py:do_getarrayitem_gc_pure_*` →
        //   `cpu.bh_getarrayitem_gc_*(array, index, ad)`.
        // `protect_speculative_array` + the array-bounds check at
        // `optimizer.py:865-867` validated the gcref/index pre-fold;
        // unsupported `item_size` matches `llmodel.py:478`'s
        // NotImplementedError, fail-loud via `unreachable!()`.
        if let (Value::Ref(array), Value::Int(index), Some(d)) = (argboxes[0], argboxes[1], descr) {
            if let Some(ad) = d.as_array_descr() {
                return Ok(Some(match opnum {
                    OpCode::GetarrayitemGcPureI => {
                        let v = cpu.bh_getarrayitem_gc_i(array, index, ad).expect(
                            "GETARRAYITEM_GC_PURE_I: unsupported item_size \
                             (llmodel.py:478 raises NotImplementedError)",
                        );
                        Value::Int(v)
                    }
                    OpCode::GetarrayitemGcPureR => {
                        Value::Ref(cpu.bh_getarrayitem_gc_r(array, index, ad))
                    }
                    OpCode::GetarrayitemGcPureF => {
                        Value::Float(cpu.bh_getarrayitem_gc_f(array, index, ad))
                    }
                    _ => return Err(NotImplemented),
                }));
            }
        }
        // STRGETITEM / UNICODEGETITEM — protect_speculative_string /
        // _unicode has validated str_descr / unicode_descr is
        // registered (under `supports_guard_gc_type == true`).
        if let (Value::Ref(s), Value::Int(index)) = (argboxes[0], argboxes[1]) {
            match opnum {
                OpCode::Strgetitem => {
                    let v = cpu.bh_strgetitem(s, index).expect(
                        "STRGETITEM: str_descr unregistered after protect_speculative_string",
                    );
                    return Ok(Some(Value::Int(v)));
                }
                OpCode::Unicodegetitem => {
                    let v = cpu.bh_unicodegetitem(s, index).expect(
                        "UNICODEGETITEM: unicode_descr unregistered after protect_speculative_unicode",
                    );
                    return Ok(Some(Value::Int(v)));
                }
                _ => {}
            }
        }
    }

    // ── arity == 3 row of EXECUTE_BY_NUM_ARGS ──
    if arity == 3 {
        // executor.py `do_int_between` -> blackhole.py:560
        // `bhimpl_int_between(a, b, c): return a <= b < c`.
        if let (Value::Int(a), Value::Int(b), Value::Int(c)) =
            (argboxes[0], argboxes[1], argboxes[2])
        {
            if opnum == OpCode::IntBetween {
                return Ok(Some(Value::Int((a <= b && b < c) as i64)));
            }
        }
    }

    // `executor.py:610 _execute_arglist` raises `NotImplementedError`
    // when `EXECUTE_BY_NUM_ARGS[arity, withdescr][opnum]` is None.
    // RPython's `optimizer.py:810 constant_fold` lets that propagate; Pyre
    // encodes the same missing-helper signal via [`NotImplemented`].
    Err(NotImplemented)
}

/// executor.py cross-type cast fold:
///   CAST_FLOAT_TO_INT / CAST_INT_TO_FLOAT — true numeric conversion
///   CAST_FLOAT_TO_SINGLEFLOAT / CAST_SINGLEFLOAT_TO_FLOAT — f64↔f32 bits
///   CONVERT_FLOAT_BYTES_TO_LONGLONG / CONVERT_LONGLONG_BYTES_TO_FLOAT —
///       reinterpret-bits pass-through (Value carries f64 bits already
///       as i64 in pyre's TraceValues; the cast just relabels)
///   CAST_PTR_TO_INT / CAST_INT_TO_PTR — pointer reinterpret
/// Mirrors blackhole.py bhimpl_cast_*.
pub fn execute_cast_const(opcode: OpCode, arg: majit_ir::Value) -> Option<majit_ir::Value> {
    use majit_ir::{GcRef, Value};
    match (opcode, arg) {
        (OpCode::CastFloatToInt, Value::Float(f)) => {
            // blackhole.py:800-808 `bhimpl_cast_float_to_int = int(int(a))`
            // lowers to `lltype.cast_float_to_int` (C `(long)f`) post-
            // translation. Skip fold on non-finite or out-of-range
            // floats: the runtime path's `as i64` saturates (lenient
            // IEEE policy, see runtime arm at executor.rs:329) but
            // trace-time fold would freeze that saturation as a
            // constant — emit the cast op instead so the runtime
            // computes consistently. The safe i64 window is
            // `i64::MIN ..= i64::MAX`; `i64::MAX + 1` rounds to the
            // same f64 as `i64::MAX` (precision loss), so use the
            // strictly-less-than upper bound `9.223372036854776e18`.
            if !f.is_finite() {
                return None;
            }
            if f < (i64::MIN as f64) || f >= 9.223372036854776e18_f64 {
                return None;
            }
            Some(Value::Int(f as i64))
        }
        (OpCode::CastIntToFloat, Value::Int(i)) => Some(Value::Float(i as f64)),
        (OpCode::CastFloatToSinglefloat, Value::Float(f)) => {
            Some(Value::Int((f as f32).to_bits() as i64))
        }
        (OpCode::CastSinglefloatToFloat, Value::Int(i)) => {
            Some(Value::Float(f32::from_bits(i as u32) as f64))
        }
        (OpCode::ConvertFloatBytesToLonglong, Value::Float(f)) => {
            Some(Value::Int(f.to_bits() as i64))
        }
        (OpCode::ConvertLonglongBytesToFloat, Value::Int(i)) => {
            Some(Value::Float(f64::from_bits(i as u64)))
        }
        // `assembler.py:1528-1529 genop_cast_ptr_to_int =
        // _genop_same_as` / `genop_cast_int_to_ptr = _genop_same_as`.
        // PyPy treats both casts as raw identity at every level:
        // backend, executor, and test_lltype.py:693-701 /
        // runner_test.py:1957-1966 expect
        // `cast_int_to_ptr(21) → cast_ptr_to_int == 21`.
        (OpCode::CastPtrToInt, Value::Ref(r)) => Some(Value::Int(r.0 as i64)),
        (OpCode::CastIntToPtr, Value::Int(i)) => Some(Value::Ref(GcRef(i as usize))),
        _ => None,
    }
}

/// Narrow [`execute_varargs`] carve-out usable without `&mut MetaInterp`.
///
/// RPython `executor.execute_varargs(cpu, ..., exc=False)`
/// (`executor.py:75-78`) skips the `metainterp.execute_raised` coordination
/// when the helper provably cannot raise — `pyjitpl.py:_record_helper_pure`
/// (`pyjitpl.py:1346-1400`) reaches this path for every `EF_ELIDABLE_CANNOT_RAISE`
/// callee. Pyre's walker (`pyre-jit-trace::jitcode_dispatch::
/// dispatch_residual_call_*`) cannot thread `&mut MetaInterp` through the
/// trace recorder seam, so this helper exposes the `exc=False` shape via
/// direct `call_int_function` / `call_void_function` dispatch.
///
/// **Caller contract** (debug-asserted): `descr.get_extra_info()` must report
/// both `check_is_elidable()` true AND `check_can_raise(false)` false.  Any
/// other EI risks landing in `BH_LAST_EXC_VALUE` with no metainterp around
/// to transcribe it, which would silently swallow the exception.
///
/// `args` follows `_build_allboxes` (`pyjitpl.py:1960-1993`) layout
/// **excluding** the funcbox: the funcbox concrete int is `func_ptr` and the
/// remaining concrete operand values pass straight through to the host ABI
/// dispatcher (`pyjitpl::call_int_function` / `call_void_function`).  Up to
/// `MAX_HOST_CALL_ARITY` (16) operand slots.
pub fn execute_pure_call(
    descr: &dyn majit_ir::descr::CallDescr,
    func_ptr: i64,
    args: &[i64],
) -> i64 {
    debug_assert!(
        descr.get_extra_info().check_is_elidable()
            && !descr.get_extra_info().check_can_raise(false),
        "execute_pure_call requires EF_ELIDABLE_CANNOT_RAISE EI"
    );
    let func_ptr = func_ptr as *const ();
    match descr.result_type() {
        // RPython dispatches Int and Ref through the same backend primitive
        // `cpu.bh_call_i` (returns i64); pyre's `call_int_function` does
        // the same — Ref is bit-identical to Int at the ABI level.
        majit_ir::Type::Int | majit_ir::Type::Ref => {
            crate::pyjitpl::call_int_function(func_ptr, args)
        }
        majit_ir::Type::Void => {
            crate::pyjitpl::call_void_function(func_ptr, args);
            0
        }
        // See `execute_varargs`'s Float arm for the i64-bits ABI rationale:
        // `#[jit_module]` Float helpers expose `concrete_ptr` as
        // `extern "C" fn(...) -> i64` with the f64 pre-packed via
        // `f64::to_bits`; routing through `call_int_function` is bit-identical.
        majit_ir::Type::Float => crate::pyjitpl::call_int_function(func_ptr, args),
    }
}

/// `executor.execute_varargs` parity for the walker layer, simplified
/// for callers that do not hold a `MetaInterp` (record-time concrete
/// execution of residual_calls).
///
/// PyPy upstream (`rpython/jit/metainterp/pyjitpl.py:1995-2126`
/// `do_residual_call`) calls `executor.execute_varargs(opnum,
/// argboxes, descr, exc=can_raise, pure=is_elidable)` for every
/// residual_call regardless of EI branch — concrete execution always
/// runs at trace-record time, only the recorded opcode kind and the
/// post-call guard emission differ.  See `select_residual_call_opcode`
/// (`pyre-jit-trace/src/jitcode_dispatch.rs`) for the per-EI-branch
/// inventory.
///
/// The walker has no `MetaInterp` to thread; instead it owns a
/// `WalkContext.last_exc_value` shadow.  This function therefore
/// returns the BH_LAST_EXC_VALUE seam value via `Result::Err` so the
/// caller can write it into the walker's exc shadow + emit
/// `GUARD_NO_EXCEPTION`, mirroring upstream's
/// `metainterp.execute_raised(bh_exc, constant=False)` +
/// `handle_possible_exception` flow.
///
/// Differences from `execute_varargs`:
///   * No `MetaInterp` argument; caller handles exception state.
///   * No `cond_call` / `cond_call_value` dispatch — residual_call
///     opcodes are not cond-call variants.
///   * Returns `Result<i64, i64>` instead of side-effecting
///     `metainterp.execute_raised`.
///
/// Differences from `execute_pure_call`:
///   * No `is_elidable + cannot_raise` debug_assert.
///   * Clears BH_LAST_EXC_VALUE before dispatch (`execute_pure_call`
///     does not because elidable_cannot_raise cannot raise).
pub fn execute_residual_call(
    descr: &dyn majit_ir::descr::CallDescr,
    func_ptr: i64,
    args: &[i64],
) -> Result<i64, i64> {
    crate::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(0));
    let func_ptr = func_ptr as *const ();
    let result = match descr.result_type() {
        majit_ir::Type::Int | majit_ir::Type::Ref => {
            crate::pyjitpl::call_int_function(func_ptr, args)
        }
        majit_ir::Type::Void => {
            crate::pyjitpl::call_void_function(func_ptr, args);
            0
        }
        // See `execute_varargs`'s Float arm for the i64-bits ABI
        // rationale: `#[jit_module]` Float helpers expose
        // `concrete_ptr` as `extern "C" fn(...) -> i64` with the f64
        // pre-packed via `f64::to_bits`.
        majit_ir::Type::Float => crate::pyjitpl::call_int_function(func_ptr, args),
    };
    let bh_exc = crate::blackhole::BH_LAST_EXC_VALUE.with(|c| {
        let v = c.get();
        c.set(0);
        v
    });
    if bh_exc != 0 { Err(bh_exc) } else { Ok(result) }
}

#[cfg(test)]
mod execute_residual_call_tests {
    use super::*;
    use majit_ir::descr::SimpleCallDescr;
    use majit_ir::{EffectInfo, ExtraEffect, Type};

    extern "C" fn add2_i64(a: i64, b: i64) -> i64 {
        a.wrapping_add(b)
    }

    // A may-force helper that raises: publishes a non-zero exception
    // pointer on `BH_LAST_EXC_VALUE` (the blackhole CALL_* convention)
    // and returns the 0 result sentinel, exactly as a raising
    // `bh_call_*` arm does.
    extern "C" fn raises_stopiteration() -> i64 {
        crate::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(0xDEAD_BEEF));
        0
    }

    fn make_may_force_descr(arg_types: Vec<Type>, result_type: Type) -> SimpleCallDescr {
        let mut effect = EffectInfo::default();
        // EF_CAN_RAISE / forces — the non-pure classification.
        effect.extraeffect = ExtraEffect::CanRaise;
        SimpleCallDescr::new(0, arg_types, result_type, false, 8, effect)
    }

    #[test]
    fn non_raising_call_returns_ok_result() {
        let descr = make_may_force_descr(vec![Type::Int, Type::Int], Type::Int);
        let r = execute_residual_call(&descr, add2_i64 as *const () as i64, &[40, 2]);
        assert_eq!(
            r,
            Ok(42),
            "a non-raising add2_i64(40, 2) must return Ok(42)"
        );
    }

    #[test]
    fn raising_call_reports_published_exception_pointer() {
        let descr = make_may_force_descr(vec![], Type::Int);
        let r = execute_residual_call(&descr, raises_stopiteration as *const () as i64, &[]);
        assert_eq!(
            r,
            Err(0xDEAD_BEEF),
            "the helper's BH_LAST_EXC_VALUE publication must surface as Err"
        );
    }

    #[test]
    fn clears_published_exception_after_consuming_it() {
        // The Err return consumes the publication; the TLS slot must not
        // keep the pointer for the next (unrelated) call to observe.
        let descr = make_may_force_descr(vec![], Type::Int);
        let r = execute_residual_call(&descr, raises_stopiteration as *const () as i64, &[]);
        assert_eq!(r, Err(0xDEAD_BEEF));
        let stale = crate::blackhole::BH_LAST_EXC_VALUE.with(|c| c.get());
        assert_eq!(
            stale, 0,
            "BH_LAST_EXC_VALUE must be cleared after the exception is consumed"
        );
    }

    #[test]
    fn clears_stale_exception_before_dispatch() {
        // A prior call's exception must not bleed into a clean call.
        crate::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(0x1234));
        let descr = make_may_force_descr(vec![Type::Int, Type::Int], Type::Int);
        let r = execute_residual_call(&descr, add2_i64 as *const () as i64, &[1, 2]);
        assert_eq!(
            r,
            Ok(3),
            "stale BH_LAST_EXC_VALUE must be cleared before dispatch"
        );
    }
}

#[cfg(test)]
mod execute_pure_call_tests {
    use super::*;
    use majit_ir::descr::SimpleCallDescr;
    use majit_ir::{EffectInfo, ExtraEffect, Type};

    extern "C" fn double_i64(x: i64) -> i64 {
        x.wrapping_mul(2)
    }

    extern "C" fn add3_i64(a: i64, b: i64, c: i64) -> i64 {
        a.wrapping_add(b).wrapping_add(c)
    }

    extern "C" fn pack_float_to_bits(x: i64) -> i64 {
        let f = f64::from_bits(x as u64) * 2.0;
        f.to_bits() as i64
    }

    extern "C" fn void_no_op(_x: i64) {}

    fn make_descr(arg_types: Vec<Type>, result_type: Type) -> SimpleCallDescr {
        let mut effect = EffectInfo::default();
        effect.extraeffect = ExtraEffect::ElidableCannotRaise;
        SimpleCallDescr::new(0, arg_types, result_type, false, 8, effect)
    }

    #[test]
    fn executes_single_int_arg_and_returns_doubled_result() {
        let descr = make_descr(vec![Type::Int], Type::Int);
        let result = execute_pure_call(&descr, double_i64 as *const () as i64, &[21]);
        assert_eq!(result, 42, "double_i64(21) must return 42");
    }

    #[test]
    fn executes_three_int_args_routing_through_call_int_function() {
        let descr = make_descr(vec![Type::Int, Type::Int, Type::Int], Type::Int);
        let result = execute_pure_call(&descr, add3_i64 as *const () as i64, &[100, 20, 3]);
        assert_eq!(result, 123, "add3_i64(100, 20, 3) must return 123");
    }

    #[test]
    fn float_result_routes_through_call_int_function_with_bits_packing() {
        let descr = make_descr(vec![Type::Float], Type::Float);
        let input_bits = 3.5_f64.to_bits() as i64;
        let result = execute_pure_call(
            &descr,
            pack_float_to_bits as *const () as i64,
            &[input_bits],
        );
        let result_f = f64::from_bits(result as u64);
        assert_eq!(result_f, 7.0, "3.5 * 2.0 must equal 7.0");
    }

    #[test]
    fn void_return_routes_through_call_void_function_and_returns_zero_sentinel() {
        let descr = make_descr(vec![Type::Int], Type::Void);
        let result = execute_pure_call(&descr, void_no_op as *const () as i64, &[99]);
        assert_eq!(result, 0, "void execute_pure_call returns the 0 sentinel");
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "execute_pure_call requires EF_ELIDABLE_CANNOT_RAISE EI")]
    fn non_elidable_ei_panics_debug_assertion() {
        let mut effect = EffectInfo::default();
        effect.extraeffect = ExtraEffect::CannotRaise;
        let descr = SimpleCallDescr::new(0, vec![Type::Int], Type::Int, false, 8, effect);
        let _ = execute_pure_call(&descr, double_i64 as *const () as i64, &[1]);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "execute_pure_call requires EF_ELIDABLE_CANNOT_RAISE EI")]
    fn elidable_can_raise_panics_debug_assertion() {
        let mut effect = EffectInfo::default();
        effect.extraeffect = ExtraEffect::ElidableCanRaise;
        let descr = SimpleCallDescr::new(0, vec![Type::Int], Type::Int, false, 8, effect);
        let _ = execute_pure_call(&descr, double_i64 as *const () as i64, &[1]);
    }
}
