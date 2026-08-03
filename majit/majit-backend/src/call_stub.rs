//! C-ABI call stub dispatch shared between backends.
//!
//! `bh_call_i_dispatch` mirrors `rpython/jit/backend/llsupport/llmodel.py:816 call_stub_i`:
//! it materializes a typed `extern "C" fn` from a raw funcptr and forwards
//! arguments in the calldescr declaration order preserved by `arg_classes`.
//! `bh_call_f_dispatch` and `bh_call_v_dispatch` are the float-returning and
//! void-returning parallels for `llmodel.py:825 bh_call_f` and
//! `llmodel.py:834 bh_call_v`.
//!
//! Upstream `descr.py:574` builds the generated call expression by walking
//! `self.arg_classes`, and `descr.py:604-605` builds `FuncType(ARGS, RESULT)`
//! from that same ordered class list. Matching that shape is required by the
//! Microsoft x64 positional-slot convention and is also correct under SysV and
//! AAPCS.

/// `descr.py:556-570 TYPE()` collapsed to the two C-ABI register classes the
/// dispatch table can express: `'i'`, `'r'` and `'L'` (`lltype.Signed`,
/// `llmemory.GCREF`, `lltype.SignedLongLong`) all pass in an integer register;
/// `'f'` (`lltype.Float`) passes in a floating-point register.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ArgClass {
    Int,
    Float,
}

/// Class-sequence table shared by `bh_call_i_dispatch`, `bh_call_f_dispatch`,
/// and `bh_call_v_dispatch`. `$ret` plugs into both the function-pointer
/// signature and the dispatch function's return type.
///
/// `descr.py:574` / `descr.py:604-605 create_call_stub` parity: the signature
/// is built in `arg_classes` declaration order, so `fn(f64, i64)` is dispatched
/// as `extern "C" fn(f64, i64)` rather than a class-blind
/// `extern "C" fn(i64, f64)`. That preserves SysV/AAPCS register-file order
/// and the Microsoft x64 positional argument slots alike.
///
/// Coverage: every ordered sequence up to 5 arguments, plus the all-`Int`
/// sequences on to `MAX_HOST_CALL_ARITY`. The float-carrying bound is mirrored
/// by `majit_translate::codewriter::jitcode::MAX_FLOAT_CARRYING_CALL_ARITY`,
/// which flags such a signature where the calldescr is built instead of at the
/// deopt that first runs it; widening the arms here means raising it there in
/// the same change (`majit-translate` cannot call into `majit-backend`, so the
/// bound is stated on both sides rather than shared).
macro_rules! dispatch_classes_body {
    ($func:ident, $classes:ident, $args:ident, $ret:ty) => {{
        type I = i64;
        type F = f64;
        assert_eq!(
            $classes.len(),
            $args.len(),
            "bh_call dispatch: class sequence and positional arg list length differ"
        );
        match $classes {
            [] => {
                let f: unsafe extern "C" fn() -> $ret = std::mem::transmute($func);
                f()
            }
            [ArgClass::Int] => {
                let f: unsafe extern "C" fn(I) -> $ret = std::mem::transmute($func);
                f($args[0])
            }
            [ArgClass::Float] => {
                let f: unsafe extern "C" fn(F) -> $ret = std::mem::transmute($func);
                f(f64::from_bits($args[0] as u64))
            }
            [ArgClass::Int, ArgClass::Int] => {
                let f: unsafe extern "C" fn(I, I) -> $ret = std::mem::transmute($func);
                f($args[0], $args[1])
            }
            [ArgClass::Int, ArgClass::Float] => {
                let f: unsafe extern "C" fn(I, F) -> $ret = std::mem::transmute($func);
                f($args[0], f64::from_bits($args[1] as u64))
            }
            [ArgClass::Float, ArgClass::Int] => {
                let f: unsafe extern "C" fn(F, I) -> $ret = std::mem::transmute($func);
                f(f64::from_bits($args[0] as u64), $args[1])
            }
            [ArgClass::Float, ArgClass::Float] => {
                let f: unsafe extern "C" fn(F, F) -> $ret = std::mem::transmute($func);
                f(
                    f64::from_bits($args[0] as u64),
                    f64::from_bits($args[1] as u64),
                )
            }
            [ArgClass::Int, ArgClass::Int, ArgClass::Int] => {
                let f: unsafe extern "C" fn(I, I, I) -> $ret = std::mem::transmute($func);
                f($args[0], $args[1], $args[2])
            }
            [ArgClass::Int, ArgClass::Int, ArgClass::Float] => {
                let f: unsafe extern "C" fn(I, I, F) -> $ret = std::mem::transmute($func);
                f($args[0], $args[1], f64::from_bits($args[2] as u64))
            }
            [ArgClass::Int, ArgClass::Float, ArgClass::Int] => {
                let f: unsafe extern "C" fn(I, F, I) -> $ret = std::mem::transmute($func);
                f($args[0], f64::from_bits($args[1] as u64), $args[2])
            }
            [ArgClass::Int, ArgClass::Float, ArgClass::Float] => {
                let f: unsafe extern "C" fn(I, F, F) -> $ret = std::mem::transmute($func);
                f(
                    $args[0],
                    f64::from_bits($args[1] as u64),
                    f64::from_bits($args[2] as u64),
                )
            }
            [ArgClass::Float, ArgClass::Int, ArgClass::Int] => {
                let f: unsafe extern "C" fn(F, I, I) -> $ret = std::mem::transmute($func);
                f(f64::from_bits($args[0] as u64), $args[1], $args[2])
            }
            [ArgClass::Float, ArgClass::Int, ArgClass::Float] => {
                let f: unsafe extern "C" fn(F, I, F) -> $ret = std::mem::transmute($func);
                f(
                    f64::from_bits($args[0] as u64),
                    $args[1],
                    f64::from_bits($args[2] as u64),
                )
            }
            [ArgClass::Float, ArgClass::Float, ArgClass::Int] => {
                let f: unsafe extern "C" fn(F, F, I) -> $ret = std::mem::transmute($func);
                f(
                    f64::from_bits($args[0] as u64),
                    f64::from_bits($args[1] as u64),
                    $args[2],
                )
            }
            [ArgClass::Float, ArgClass::Float, ArgClass::Float] => {
                let f: unsafe extern "C" fn(F, F, F) -> $ret = std::mem::transmute($func);
                f(
                    f64::from_bits($args[0] as u64),
                    f64::from_bits($args[1] as u64),
                    f64::from_bits($args[2] as u64),
                )
            }
            [ArgClass::Int, ArgClass::Int, ArgClass::Int, ArgClass::Int] => {
                let f: unsafe extern "C" fn(I, I, I, I) -> $ret = std::mem::transmute($func);
                f($args[0], $args[1], $args[2], $args[3])
            }
            [ArgClass::Int, ArgClass::Int, ArgClass::Int, ArgClass::Float] => {
                let f: unsafe extern "C" fn(I, I, I, F) -> $ret = std::mem::transmute($func);
                f(
                    $args[0],
                    $args[1],
                    $args[2],
                    f64::from_bits($args[3] as u64),
                )
            }
            [ArgClass::Int, ArgClass::Int, ArgClass::Float, ArgClass::Int] => {
                let f: unsafe extern "C" fn(I, I, F, I) -> $ret = std::mem::transmute($func);
                f(
                    $args[0],
                    $args[1],
                    f64::from_bits($args[2] as u64),
                    $args[3],
                )
            }
            [
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Float,
                ArgClass::Float,
            ] => {
                let f: unsafe extern "C" fn(I, I, F, F) -> $ret = std::mem::transmute($func);
                f(
                    $args[0],
                    $args[1],
                    f64::from_bits($args[2] as u64),
                    f64::from_bits($args[3] as u64),
                )
            }
            [ArgClass::Int, ArgClass::Float, ArgClass::Int, ArgClass::Int] => {
                let f: unsafe extern "C" fn(I, F, I, I) -> $ret = std::mem::transmute($func);
                f(
                    $args[0],
                    f64::from_bits($args[1] as u64),
                    $args[2],
                    $args[3],
                )
            }
            [
                ArgClass::Int,
                ArgClass::Float,
                ArgClass::Int,
                ArgClass::Float,
            ] => {
                let f: unsafe extern "C" fn(I, F, I, F) -> $ret = std::mem::transmute($func);
                f(
                    $args[0],
                    f64::from_bits($args[1] as u64),
                    $args[2],
                    f64::from_bits($args[3] as u64),
                )
            }
            [
                ArgClass::Int,
                ArgClass::Float,
                ArgClass::Float,
                ArgClass::Int,
            ] => {
                let f: unsafe extern "C" fn(I, F, F, I) -> $ret = std::mem::transmute($func);
                f(
                    $args[0],
                    f64::from_bits($args[1] as u64),
                    f64::from_bits($args[2] as u64),
                    $args[3],
                )
            }
            [
                ArgClass::Int,
                ArgClass::Float,
                ArgClass::Float,
                ArgClass::Float,
            ] => {
                let f: unsafe extern "C" fn(I, F, F, F) -> $ret = std::mem::transmute($func);
                f(
                    $args[0],
                    f64::from_bits($args[1] as u64),
                    f64::from_bits($args[2] as u64),
                    f64::from_bits($args[3] as u64),
                )
            }
            [ArgClass::Float, ArgClass::Int, ArgClass::Int, ArgClass::Int] => {
                let f: unsafe extern "C" fn(F, I, I, I) -> $ret = std::mem::transmute($func);
                f(
                    f64::from_bits($args[0] as u64),
                    $args[1],
                    $args[2],
                    $args[3],
                )
            }
            [
                ArgClass::Float,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Float,
            ] => {
                let f: unsafe extern "C" fn(F, I, I, F) -> $ret = std::mem::transmute($func);
                f(
                    f64::from_bits($args[0] as u64),
                    $args[1],
                    $args[2],
                    f64::from_bits($args[3] as u64),
                )
            }
            [
                ArgClass::Float,
                ArgClass::Int,
                ArgClass::Float,
                ArgClass::Int,
            ] => {
                let f: unsafe extern "C" fn(F, I, F, I) -> $ret = std::mem::transmute($func);
                f(
                    f64::from_bits($args[0] as u64),
                    $args[1],
                    f64::from_bits($args[2] as u64),
                    $args[3],
                )
            }
            [
                ArgClass::Float,
                ArgClass::Int,
                ArgClass::Float,
                ArgClass::Float,
            ] => {
                let f: unsafe extern "C" fn(F, I, F, F) -> $ret = std::mem::transmute($func);
                f(
                    f64::from_bits($args[0] as u64),
                    $args[1],
                    f64::from_bits($args[2] as u64),
                    f64::from_bits($args[3] as u64),
                )
            }
            [
                ArgClass::Float,
                ArgClass::Float,
                ArgClass::Int,
                ArgClass::Int,
            ] => {
                let f: unsafe extern "C" fn(F, F, I, I) -> $ret = std::mem::transmute($func);
                f(
                    f64::from_bits($args[0] as u64),
                    f64::from_bits($args[1] as u64),
                    $args[2],
                    $args[3],
                )
            }
            [
                ArgClass::Float,
                ArgClass::Float,
                ArgClass::Int,
                ArgClass::Float,
            ] => {
                let f: unsafe extern "C" fn(F, F, I, F) -> $ret = std::mem::transmute($func);
                f(
                    f64::from_bits($args[0] as u64),
                    f64::from_bits($args[1] as u64),
                    $args[2],
                    f64::from_bits($args[3] as u64),
                )
            }
            [
                ArgClass::Float,
                ArgClass::Float,
                ArgClass::Float,
                ArgClass::Int,
            ] => {
                let f: unsafe extern "C" fn(F, F, F, I) -> $ret = std::mem::transmute($func);
                f(
                    f64::from_bits($args[0] as u64),
                    f64::from_bits($args[1] as u64),
                    f64::from_bits($args[2] as u64),
                    $args[3],
                )
            }
            [
                ArgClass::Float,
                ArgClass::Float,
                ArgClass::Float,
                ArgClass::Float,
            ] => {
                let f: unsafe extern "C" fn(F, F, F, F) -> $ret = std::mem::transmute($func);
                f(
                    f64::from_bits($args[0] as u64),
                    f64::from_bits($args[1] as u64),
                    f64::from_bits($args[2] as u64),
                    f64::from_bits($args[3] as u64),
                )
            }
            [
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
            ] => {
                let f: unsafe extern "C" fn(I, I, I, I, I) -> $ret = std::mem::transmute($func);
                f($args[0], $args[1], $args[2], $args[3], $args[4])
            }
            [
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Float,
            ] => {
                let f: unsafe extern "C" fn(I, I, I, I, F) -> $ret = std::mem::transmute($func);
                f(
                    $args[0],
                    $args[1],
                    $args[2],
                    $args[3],
                    f64::from_bits($args[4] as u64),
                )
            }
            [
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Float,
                ArgClass::Int,
            ] => {
                let f: unsafe extern "C" fn(I, I, I, F, I) -> $ret = std::mem::transmute($func);
                f(
                    $args[0],
                    $args[1],
                    $args[2],
                    f64::from_bits($args[3] as u64),
                    $args[4],
                )
            }
            [
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Float,
                ArgClass::Float,
            ] => {
                let f: unsafe extern "C" fn(I, I, I, F, F) -> $ret = std::mem::transmute($func);
                f(
                    $args[0],
                    $args[1],
                    $args[2],
                    f64::from_bits($args[3] as u64),
                    f64::from_bits($args[4] as u64),
                )
            }
            [
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Float,
                ArgClass::Int,
                ArgClass::Int,
            ] => {
                let f: unsafe extern "C" fn(I, I, F, I, I) -> $ret = std::mem::transmute($func);
                f(
                    $args[0],
                    $args[1],
                    f64::from_bits($args[2] as u64),
                    $args[3],
                    $args[4],
                )
            }
            [
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Float,
                ArgClass::Int,
                ArgClass::Float,
            ] => {
                let f: unsafe extern "C" fn(I, I, F, I, F) -> $ret = std::mem::transmute($func);
                f(
                    $args[0],
                    $args[1],
                    f64::from_bits($args[2] as u64),
                    $args[3],
                    f64::from_bits($args[4] as u64),
                )
            }
            [
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Float,
                ArgClass::Float,
                ArgClass::Int,
            ] => {
                let f: unsafe extern "C" fn(I, I, F, F, I) -> $ret = std::mem::transmute($func);
                f(
                    $args[0],
                    $args[1],
                    f64::from_bits($args[2] as u64),
                    f64::from_bits($args[3] as u64),
                    $args[4],
                )
            }
            [
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Float,
                ArgClass::Float,
                ArgClass::Float,
            ] => {
                let f: unsafe extern "C" fn(I, I, F, F, F) -> $ret = std::mem::transmute($func);
                f(
                    $args[0],
                    $args[1],
                    f64::from_bits($args[2] as u64),
                    f64::from_bits($args[3] as u64),
                    f64::from_bits($args[4] as u64),
                )
            }
            [
                ArgClass::Int,
                ArgClass::Float,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
            ] => {
                let f: unsafe extern "C" fn(I, F, I, I, I) -> $ret = std::mem::transmute($func);
                f(
                    $args[0],
                    f64::from_bits($args[1] as u64),
                    $args[2],
                    $args[3],
                    $args[4],
                )
            }
            [
                ArgClass::Int,
                ArgClass::Float,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Float,
            ] => {
                let f: unsafe extern "C" fn(I, F, I, I, F) -> $ret = std::mem::transmute($func);
                f(
                    $args[0],
                    f64::from_bits($args[1] as u64),
                    $args[2],
                    $args[3],
                    f64::from_bits($args[4] as u64),
                )
            }
            [
                ArgClass::Int,
                ArgClass::Float,
                ArgClass::Int,
                ArgClass::Float,
                ArgClass::Int,
            ] => {
                let f: unsafe extern "C" fn(I, F, I, F, I) -> $ret = std::mem::transmute($func);
                f(
                    $args[0],
                    f64::from_bits($args[1] as u64),
                    $args[2],
                    f64::from_bits($args[3] as u64),
                    $args[4],
                )
            }
            [
                ArgClass::Int,
                ArgClass::Float,
                ArgClass::Int,
                ArgClass::Float,
                ArgClass::Float,
            ] => {
                let f: unsafe extern "C" fn(I, F, I, F, F) -> $ret = std::mem::transmute($func);
                f(
                    $args[0],
                    f64::from_bits($args[1] as u64),
                    $args[2],
                    f64::from_bits($args[3] as u64),
                    f64::from_bits($args[4] as u64),
                )
            }
            [
                ArgClass::Int,
                ArgClass::Float,
                ArgClass::Float,
                ArgClass::Int,
                ArgClass::Int,
            ] => {
                let f: unsafe extern "C" fn(I, F, F, I, I) -> $ret = std::mem::transmute($func);
                f(
                    $args[0],
                    f64::from_bits($args[1] as u64),
                    f64::from_bits($args[2] as u64),
                    $args[3],
                    $args[4],
                )
            }
            [
                ArgClass::Int,
                ArgClass::Float,
                ArgClass::Float,
                ArgClass::Int,
                ArgClass::Float,
            ] => {
                let f: unsafe extern "C" fn(I, F, F, I, F) -> $ret = std::mem::transmute($func);
                f(
                    $args[0],
                    f64::from_bits($args[1] as u64),
                    f64::from_bits($args[2] as u64),
                    $args[3],
                    f64::from_bits($args[4] as u64),
                )
            }
            [
                ArgClass::Int,
                ArgClass::Float,
                ArgClass::Float,
                ArgClass::Float,
                ArgClass::Int,
            ] => {
                let f: unsafe extern "C" fn(I, F, F, F, I) -> $ret = std::mem::transmute($func);
                f(
                    $args[0],
                    f64::from_bits($args[1] as u64),
                    f64::from_bits($args[2] as u64),
                    f64::from_bits($args[3] as u64),
                    $args[4],
                )
            }
            [
                ArgClass::Int,
                ArgClass::Float,
                ArgClass::Float,
                ArgClass::Float,
                ArgClass::Float,
            ] => {
                let f: unsafe extern "C" fn(I, F, F, F, F) -> $ret = std::mem::transmute($func);
                f(
                    $args[0],
                    f64::from_bits($args[1] as u64),
                    f64::from_bits($args[2] as u64),
                    f64::from_bits($args[3] as u64),
                    f64::from_bits($args[4] as u64),
                )
            }
            [
                ArgClass::Float,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
            ] => {
                let f: unsafe extern "C" fn(F, I, I, I, I) -> $ret = std::mem::transmute($func);
                f(
                    f64::from_bits($args[0] as u64),
                    $args[1],
                    $args[2],
                    $args[3],
                    $args[4],
                )
            }
            [
                ArgClass::Float,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Float,
            ] => {
                let f: unsafe extern "C" fn(F, I, I, I, F) -> $ret = std::mem::transmute($func);
                f(
                    f64::from_bits($args[0] as u64),
                    $args[1],
                    $args[2],
                    $args[3],
                    f64::from_bits($args[4] as u64),
                )
            }
            [
                ArgClass::Float,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Float,
                ArgClass::Int,
            ] => {
                let f: unsafe extern "C" fn(F, I, I, F, I) -> $ret = std::mem::transmute($func);
                f(
                    f64::from_bits($args[0] as u64),
                    $args[1],
                    $args[2],
                    f64::from_bits($args[3] as u64),
                    $args[4],
                )
            }
            [
                ArgClass::Float,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Float,
                ArgClass::Float,
            ] => {
                let f: unsafe extern "C" fn(F, I, I, F, F) -> $ret = std::mem::transmute($func);
                f(
                    f64::from_bits($args[0] as u64),
                    $args[1],
                    $args[2],
                    f64::from_bits($args[3] as u64),
                    f64::from_bits($args[4] as u64),
                )
            }
            [
                ArgClass::Float,
                ArgClass::Int,
                ArgClass::Float,
                ArgClass::Int,
                ArgClass::Int,
            ] => {
                let f: unsafe extern "C" fn(F, I, F, I, I) -> $ret = std::mem::transmute($func);
                f(
                    f64::from_bits($args[0] as u64),
                    $args[1],
                    f64::from_bits($args[2] as u64),
                    $args[3],
                    $args[4],
                )
            }
            [
                ArgClass::Float,
                ArgClass::Int,
                ArgClass::Float,
                ArgClass::Int,
                ArgClass::Float,
            ] => {
                let f: unsafe extern "C" fn(F, I, F, I, F) -> $ret = std::mem::transmute($func);
                f(
                    f64::from_bits($args[0] as u64),
                    $args[1],
                    f64::from_bits($args[2] as u64),
                    $args[3],
                    f64::from_bits($args[4] as u64),
                )
            }
            [
                ArgClass::Float,
                ArgClass::Int,
                ArgClass::Float,
                ArgClass::Float,
                ArgClass::Int,
            ] => {
                let f: unsafe extern "C" fn(F, I, F, F, I) -> $ret = std::mem::transmute($func);
                f(
                    f64::from_bits($args[0] as u64),
                    $args[1],
                    f64::from_bits($args[2] as u64),
                    f64::from_bits($args[3] as u64),
                    $args[4],
                )
            }
            [
                ArgClass::Float,
                ArgClass::Int,
                ArgClass::Float,
                ArgClass::Float,
                ArgClass::Float,
            ] => {
                let f: unsafe extern "C" fn(F, I, F, F, F) -> $ret = std::mem::transmute($func);
                f(
                    f64::from_bits($args[0] as u64),
                    $args[1],
                    f64::from_bits($args[2] as u64),
                    f64::from_bits($args[3] as u64),
                    f64::from_bits($args[4] as u64),
                )
            }
            [
                ArgClass::Float,
                ArgClass::Float,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
            ] => {
                let f: unsafe extern "C" fn(F, F, I, I, I) -> $ret = std::mem::transmute($func);
                f(
                    f64::from_bits($args[0] as u64),
                    f64::from_bits($args[1] as u64),
                    $args[2],
                    $args[3],
                    $args[4],
                )
            }
            [
                ArgClass::Float,
                ArgClass::Float,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Float,
            ] => {
                let f: unsafe extern "C" fn(F, F, I, I, F) -> $ret = std::mem::transmute($func);
                f(
                    f64::from_bits($args[0] as u64),
                    f64::from_bits($args[1] as u64),
                    $args[2],
                    $args[3],
                    f64::from_bits($args[4] as u64),
                )
            }
            [
                ArgClass::Float,
                ArgClass::Float,
                ArgClass::Int,
                ArgClass::Float,
                ArgClass::Int,
            ] => {
                let f: unsafe extern "C" fn(F, F, I, F, I) -> $ret = std::mem::transmute($func);
                f(
                    f64::from_bits($args[0] as u64),
                    f64::from_bits($args[1] as u64),
                    $args[2],
                    f64::from_bits($args[3] as u64),
                    $args[4],
                )
            }
            [
                ArgClass::Float,
                ArgClass::Float,
                ArgClass::Int,
                ArgClass::Float,
                ArgClass::Float,
            ] => {
                let f: unsafe extern "C" fn(F, F, I, F, F) -> $ret = std::mem::transmute($func);
                f(
                    f64::from_bits($args[0] as u64),
                    f64::from_bits($args[1] as u64),
                    $args[2],
                    f64::from_bits($args[3] as u64),
                    f64::from_bits($args[4] as u64),
                )
            }
            [
                ArgClass::Float,
                ArgClass::Float,
                ArgClass::Float,
                ArgClass::Int,
                ArgClass::Int,
            ] => {
                let f: unsafe extern "C" fn(F, F, F, I, I) -> $ret = std::mem::transmute($func);
                f(
                    f64::from_bits($args[0] as u64),
                    f64::from_bits($args[1] as u64),
                    f64::from_bits($args[2] as u64),
                    $args[3],
                    $args[4],
                )
            }
            [
                ArgClass::Float,
                ArgClass::Float,
                ArgClass::Float,
                ArgClass::Int,
                ArgClass::Float,
            ] => {
                let f: unsafe extern "C" fn(F, F, F, I, F) -> $ret = std::mem::transmute($func);
                f(
                    f64::from_bits($args[0] as u64),
                    f64::from_bits($args[1] as u64),
                    f64::from_bits($args[2] as u64),
                    $args[3],
                    f64::from_bits($args[4] as u64),
                )
            }
            [
                ArgClass::Float,
                ArgClass::Float,
                ArgClass::Float,
                ArgClass::Float,
                ArgClass::Int,
            ] => {
                let f: unsafe extern "C" fn(F, F, F, F, I) -> $ret = std::mem::transmute($func);
                f(
                    f64::from_bits($args[0] as u64),
                    f64::from_bits($args[1] as u64),
                    f64::from_bits($args[2] as u64),
                    f64::from_bits($args[3] as u64),
                    $args[4],
                )
            }
            [
                ArgClass::Float,
                ArgClass::Float,
                ArgClass::Float,
                ArgClass::Float,
                ArgClass::Float,
            ] => {
                let f: unsafe extern "C" fn(F, F, F, F, F) -> $ret = std::mem::transmute($func);
                f(
                    f64::from_bits($args[0] as u64),
                    f64::from_bits($args[1] as u64),
                    f64::from_bits($args[2] as u64),
                    f64::from_bits($args[3] as u64),
                    f64::from_bits($args[4] as u64),
                )
            }
            [
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
            ] => {
                let f: unsafe extern "C" fn(I, I, I, I, I, I) -> $ret = std::mem::transmute($func);
                f($args[0], $args[1], $args[2], $args[3], $args[4], $args[5])
            }
            [
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
            ] => {
                let f: unsafe extern "C" fn(I, I, I, I, I, I, I) -> $ret =
                    std::mem::transmute($func);
                f(
                    $args[0], $args[1], $args[2], $args[3], $args[4], $args[5], $args[6],
                )
            }
            [
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
            ] => {
                let f: unsafe extern "C" fn(I, I, I, I, I, I, I, I) -> $ret =
                    std::mem::transmute($func);
                f(
                    $args[0], $args[1], $args[2], $args[3], $args[4], $args[5], $args[6], $args[7],
                )
            }
            [
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
            ] => {
                let f: unsafe extern "C" fn(I, I, I, I, I, I, I, I, I) -> $ret =
                    std::mem::transmute($func);
                f(
                    $args[0], $args[1], $args[2], $args[3], $args[4], $args[5], $args[6], $args[7],
                    $args[8],
                )
            }
            [
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
            ] => {
                let f: unsafe extern "C" fn(I, I, I, I, I, I, I, I, I, I) -> $ret =
                    std::mem::transmute($func);
                f(
                    $args[0], $args[1], $args[2], $args[3], $args[4], $args[5], $args[6], $args[7],
                    $args[8], $args[9],
                )
            }
            [
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
            ] => {
                let f: unsafe extern "C" fn(I, I, I, I, I, I, I, I, I, I, I) -> $ret =
                    std::mem::transmute($func);
                f(
                    $args[0], $args[1], $args[2], $args[3], $args[4], $args[5], $args[6], $args[7],
                    $args[8], $args[9], $args[10],
                )
            }
            [
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
            ] => {
                let f: unsafe extern "C" fn(I, I, I, I, I, I, I, I, I, I, I, I) -> $ret =
                    std::mem::transmute($func);
                f(
                    $args[0], $args[1], $args[2], $args[3], $args[4], $args[5], $args[6], $args[7],
                    $args[8], $args[9], $args[10], $args[11],
                )
            }
            [
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
            ] => {
                let f: unsafe extern "C" fn(I, I, I, I, I, I, I, I, I, I, I, I, I) -> $ret =
                    std::mem::transmute($func);
                f(
                    $args[0], $args[1], $args[2], $args[3], $args[4], $args[5], $args[6], $args[7],
                    $args[8], $args[9], $args[10], $args[11], $args[12],
                )
            }
            [
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
            ] => {
                let f: unsafe extern "C" fn(I, I, I, I, I, I, I, I, I, I, I, I, I, I) -> $ret =
                    std::mem::transmute($func);
                f(
                    $args[0], $args[1], $args[2], $args[3], $args[4], $args[5], $args[6], $args[7],
                    $args[8], $args[9], $args[10], $args[11], $args[12], $args[13],
                )
            }
            [
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
            ] => {
                let f: unsafe extern "C" fn(I, I, I, I, I, I, I, I, I, I, I, I, I, I, I) -> $ret =
                    std::mem::transmute($func);
                f(
                    $args[0], $args[1], $args[2], $args[3], $args[4], $args[5], $args[6], $args[7],
                    $args[8], $args[9], $args[10], $args[11], $args[12], $args[13], $args[14],
                )
            }
            [
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
                ArgClass::Int,
            ] => {
                let f: unsafe extern "C" fn(
                    I,
                    I,
                    I,
                    I,
                    I,
                    I,
                    I,
                    I,
                    I,
                    I,
                    I,
                    I,
                    I,
                    I,
                    I,
                    I,
                ) -> $ret = std::mem::transmute($func);
                f(
                    $args[0], $args[1], $args[2], $args[3], $args[4], $args[5], $args[6], $args[7],
                    $args[8], $args[9], $args[10], $args[11], $args[12], $args[13], $args[14],
                    $args[15],
                )
            }
            classes => {
                // TODO: upstream
                // `rpython/jit/backend/llsupport/descr.py:574` /
                // `descr.py:604-605 create_call_stub` generates a
                // per-calldescr stub at translation time, so every class
                // sequence has a matching extern "C" signature. Rust has no
                // translation-time codegen equivalent here, so the dispatch
                // is a hand-rolled class-sequence table. Convergence path:
                // wire libffi (or an ABI adapter) so any sequence is
                // dispatchable; until then, callees outside the table panic
                // instead of silently corrupting registers.
                panic!(
                    "bh_call dispatch: unsupported arg class sequence {classes:?}; \
                     needs libffi for general dispatch"
                );
            }
        }
    }};
}

/// llmodel.py:816 call_stub_i: ABI-correct dispatch in calldescr declaration
/// order.
///
/// Safety: `func` must be a valid function pointer matching `classes`, i.e. an
/// `extern "C" fn(...) -> i64` whose parameter list is the same ordered
/// Int/Float sequence and whose float slots are carried in `args` as
/// `f64::to_bits`.
pub unsafe fn bh_call_i_dispatch(func: usize, classes: &[ArgClass], args: &[i64]) -> i64 {
    unsafe { dispatch_classes_body!(func, classes, args, i64) }
}

/// llmodel.py:834 bh_call_v: void-typed parallel of `bh_call_i_dispatch`.
///
/// Safety: `func` must be a valid function pointer matching `classes`.
/// `descr.py:590-605 create_call_stub` builds a real void-returning stub for
/// `RESULT == lltype.Void`; calling such a function through an `i64`-returning
/// transmute reads garbage from rax/x0, so the canonical
/// `BC_RESIDUAL_CALL_*_V` blackhole/trace path must use this dispatcher.
pub unsafe fn bh_call_v_dispatch(func: usize, classes: &[ArgClass], args: &[i64]) {
    unsafe { dispatch_classes_body!(func, classes, args, ()) }
}

/// llmodel.py:825 bh_call_f: f64-typed parallel of `bh_call_i_dispatch`.
///
/// Safety: `func` must be a valid function pointer matching `classes`.
/// `descr.py:584-605 create_call_stub` generates a real f64-returning stub for
/// `RESULT == lltype.Float`; the C ABI returns f64 in xmm0 / d0 rather than
/// rax / x0, so an `i64`-typed transmute would read uninitialized integer-bank
/// state.
pub unsafe fn bh_call_f_dispatch(func: usize, classes: &[ArgClass], args: &[i64]) -> f64 {
    unsafe { dispatch_classes_body!(func, classes, args, f64) }
}

/// Build the C-ABI class sequence and positional argument list from
/// `args_i` / `args_r` / `args_f`, following `calldescr.arg_classes` order.
///
/// `arg_classes` is the per-argument class string from
/// `majit_translate::jitcode::BhCallDescr`. RPython
/// `rpython/jit/backend/llsupport/descr.py:545-574 create_call_stub`'s
/// `process(c)` walks the class string in declaration order and pulls the next
/// item out of the corresponding storage bank.
///
/// | class | storage bank | C ABI type              | dispatch class |
/// |-------|--------------|-------------------------|----------------|
/// | `i`   | `args_i`     | `lltype.Signed`         | Int            |
/// | `r`   | `args_r`     | `llmemory.GCREF`        | Int            |
/// | `f`   | `args_f`     | `lltype.Float`          | Float          |
/// | `L`   | `args_f`     | `lltype.SignedLongLong` | Int            |
/// | `S`   | `args_i`     | `lltype.SingleFloat`    | Float (f32)    |
///
/// Note the asymmetry: `L` is stored in the float bank (PyPy `process('L')`
/// rewrites `c = 'f'` for the storage lookup) yet passed in an integer
/// register, while `S` is stored in the int bank (PyPy
/// `int2singlefloat(args_i[..])`) yet passed in a float register as a
/// 32-bit value.
///
/// `S` currently panics: pyre's dispatch table only emits `extern "C" fn(.., f64, ..)`
/// arms, so an `f32` ABI cannot be transmuted accurately (a 64-bit movsd
/// vs. a 32-bit movss to the same xmm/d register file). Pyre's
/// `type_to_argclass` (`majit-translate/src/codewriter/call.rs:190-197`)
/// never produces `S`, so the panic is unreachable from in-tree callers
/// today; reaching it requires a foreign-supplied calldescr (e.g. a
/// build-time bincode embed loaded from RPython).
///
/// Mirrors `rpython/jit/backend/llsupport/descr.py:616-620 verify_types`:
/// the per-class counts in `arg_classes` must match the corresponding list
/// length, and any unknown class is a codegen bug.
pub fn collect_call_args(
    arg_classes: &str,
    args_i: Option<&[i64]>,
    args_r: Option<&[i64]>,
    args_f: Option<&[i64]>,
) -> (Vec<ArgClass>, Vec<i64>) {
    // descr.py:616-620 verify_types parity: assert per-class counts.
    let count_i: usize = arg_classes
        .chars()
        .filter(|c| matches!(c, 'i' | 'S'))
        .count();
    let count_r: usize = arg_classes.chars().filter(|c| *c == 'r').count();
    let count_f: usize = arg_classes
        .chars()
        .filter(|c| matches!(c, 'f' | 'L'))
        .count();
    let len_i = args_i.map_or(0, <[i64]>::len);
    let len_r = args_r.map_or(0, <[i64]>::len);
    let len_f = args_f.map_or(0, <[i64]>::len);
    assert_eq!(
        count_i, len_i,
        "BhCallDescr.verify_types: arg_classes={arg_classes:?} has {count_i} int slots, args_i has {len_i}"
    );
    assert_eq!(
        count_r, len_r,
        "BhCallDescr.verify_types: arg_classes={arg_classes:?} has {count_r} ref slots, args_r has {len_r}"
    );
    assert_eq!(
        count_f, len_f,
        "BhCallDescr.verify_types: arg_classes={arg_classes:?} has {count_f} float slots, args_f has {len_f}"
    );

    let mut classes: Vec<ArgClass> = Vec::with_capacity(arg_classes.len());
    let mut args: Vec<i64> = Vec::with_capacity(arg_classes.len());
    let mut ii = 0usize;
    let mut ri = 0usize;
    let mut fi = 0usize;
    for c in arg_classes.chars() {
        match c {
            'i' => {
                classes.push(ArgClass::Int);
                args.push(args_i.expect("BhCallDescr.collect_call_args: args_i missing")[ii]);
                ii += 1;
            }
            'r' => {
                classes.push(ArgClass::Int);
                args.push(args_r.expect("BhCallDescr.collect_call_args: args_r missing")[ri]);
                ri += 1;
            }
            'f' => {
                classes.push(ArgClass::Float);
                args.push(args_f.expect("BhCallDescr.collect_call_args: args_f missing")[fi]);
                fi += 1;
            }
            'L' => {
                // descr.py:546-548 process('L'): storage bank = `args_f`
                // (PyPy rewrites `c = 'f'` for the lookup); FUNC parameter
                // type = `lltype.SignedLongLong` -> C `long long` ->
                // 8-byte int dispatched in an integer register.
                classes.push(ArgClass::Int);
                args.push(args_f.expect("BhCallDescr.collect_call_args: args_f missing")[fi]);
                fi += 1;
            }
            'S' => {
                // descr.py:551-552 process('S'): storage bank = `args_i`
                // (PyPy reads via `int2singlefloat(args_i[..])`); FUNC
                // parameter type = `lltype.SingleFloat` -> C `float` ->
                // 32-bit float dispatched in an xmm/d register. pyre's
                // dispatch table emits only `extern "C" fn(.., f64, ..)`
                // arms, so transmuting f32 through f64 would mismatch the
                // C ABI (movss vs. movsd to the same register file).
                let _ = (ii, args_i);
                panic!(
                    "BhCallDescr.collect_call_args: 'S' (SingleFloat) ABI \
                     requires f32-aware dispatch; pyre's dispatch table \
                     only supports f64. arg_classes={arg_classes:?}"
                );
            }
            other => panic!(
                "BhCallDescr.collect_call_args: unsupported arg class {other:?} \
                 in arg_classes={arg_classes:?}"
            ),
        }
    }
    (classes, args)
}

/// Bucket `args_i` / `args_r` / `args_f` into a single positional list in
/// `arg_classes` order, floats carried as their raw 64-bit pattern.
///
/// Unlike [`collect_call_args`] this does NOT split the arguments into the
/// SysV/AAPCS integer + float register files — it preserves the original
/// declaration order so a positional ABI (the wasm `call_indirect`, where
/// arguments are stack-positional rather than register-file partitioned)
/// receives them as the callee declares them. Used by the residual-host-call
/// trampoline path (see [`residual_host_call`]).
pub fn collect_call_args_positional(
    arg_classes: &str,
    args_i: Option<&[i64]>,
    args_r: Option<&[i64]>,
    args_f: Option<&[i64]>,
) -> Vec<i64> {
    let mut out: Vec<i64> = Vec::with_capacity(arg_classes.len());
    let mut ii = 0usize;
    let mut ri = 0usize;
    let mut fi = 0usize;
    for c in arg_classes.chars() {
        match c {
            'i' => {
                out.push(args_i.expect("collect_call_args_positional: args_i missing")[ii]);
                ii += 1;
            }
            'r' => {
                out.push(args_r.expect("collect_call_args_positional: args_r missing")[ri]);
                ri += 1;
            }
            // `f` (Float) and `L` (SignedLongLong) both live in the `args_f`
            // storage bank; their raw 64-bit slot value is forwarded verbatim
            // — the host trampoline coerces it to f64/i64 from the callee's
            // reflected parameter type.
            'f' | 'L' => {
                out.push(args_f.expect("collect_call_args_positional: args_f missing")[fi]);
                fi += 1;
            }
            other => panic!(
                "collect_call_args_positional: unsupported arg class {other:?} \
                 in arg_classes={arg_classes:?}"
            ),
        }
    }
    out
}

/// Dispatch a residual call described by `arg_classes`, picking the signature
/// strategy the active backend supports.
///
/// `bh_call_*_dispatch` transmutes the funcptr to an `extern "C" fn` built
/// from `arg_classes` declaration order, matching `descr.py:574` /
/// `descr.py:604-605 create_call_stub`. wasm32 still cannot use that direct
/// transmute path: `call_indirect` type-checks the callee's declared type on
/// every call, and a pointer parameter is `i32` where the native table uses
/// `i64`, so a mistyped guess traps with `indirect call type mismatch`.
///
/// Where a host trampoline is installed (`set_residual_host_call`, wasm32) the
/// call must therefore go through it with the positional argument list.
///
/// # Safety
/// On the transmute path, `func` must match the ABI [`collect_call_args`]
/// derives from `arg_classes` — see [`bh_call_i_dispatch`].
pub unsafe fn bh_call_i_by_classes(
    func: usize,
    arg_classes: &str,
    args_i: Option<&[i64]>,
    args_r: Option<&[i64]>,
    args_f: Option<&[i64]>,
) -> i64 {
    if let Some(hook) = residual_host_call() {
        let args = collect_call_args_positional(arg_classes, args_i, args_r, args_f);
        return hook(func, &args);
    }
    let (classes, args) = collect_call_args(arg_classes, args_i, args_r, args_f);
    unsafe { bh_call_i_dispatch(func, &classes, &args) }
}

/// f64-returning parallel of [`bh_call_i_by_classes`].
///
/// # Safety
/// See [`bh_call_i_by_classes`].
pub unsafe fn bh_call_f_by_classes(
    func: usize,
    arg_classes: &str,
    args_i: Option<&[i64]>,
    args_r: Option<&[i64]>,
    args_f: Option<&[i64]>,
) -> f64 {
    if let Some(hook) = residual_host_call() {
        let args = collect_call_args_positional(arg_classes, args_i, args_r, args_f);
        // The trampoline returns an f64 callee result as its raw bits.
        return f64::from_bits(hook(func, &args) as u64);
    }
    let (classes, args) = collect_call_args(arg_classes, args_i, args_r, args_f);
    unsafe { bh_call_f_dispatch(func, &classes, &args) }
}

/// Result-discarding parallel of [`bh_call_i_by_classes`].
///
/// # Safety
/// See [`bh_call_i_by_classes`].
pub unsafe fn bh_call_v_by_classes(
    func: usize,
    arg_classes: &str,
    args_i: Option<&[i64]>,
    args_r: Option<&[i64]>,
    args_f: Option<&[i64]>,
) {
    if let Some(hook) = residual_host_call() {
        let args = collect_call_args_positional(arg_classes, args_i, args_r, args_f);
        let _ = hook(func, &args);
        return;
    }
    let (classes, args) = collect_call_args(arg_classes, args_i, args_r, args_f);
    unsafe { bh_call_v_dispatch(func, &classes, &args) }
}

/// A host-provided trampoline that performs a residual call by reflecting the
/// callee's real signature, rather than transmuting the raw funcptr to a
/// statically-guessed `extern "C" fn`.
///

/// `func_ptr` is the raw callee address (a table index on wasm32); `args` is
/// the positional argument list (floats as raw bits). The return value is the
/// callee result as a 64-bit pattern (Void callees return 0; Ref returns the
/// pointer; Float returns `f64::to_bits`).
///
/// Installed only where in-module static-signature dispatch is impossible:
/// the wasm32 backend, whose `call_indirect` type-checks every call and so
/// cannot reuse the uniform-`i64` transmute that the SysV/AAPCS C ABI tolerates
/// on native backends. `None` (the default on dynasm/cranelift) keeps the
/// direct transmute path.
pub type ResidualHostCallFn = fn(func_ptr: usize, args: &[i64]) -> i64;

thread_local! {
    static RESIDUAL_HOST_CALL: std::cell::Cell<Option<ResidualHostCallFn>> =
        const { std::cell::Cell::new(None) };
}

/// Install the active residual-call host trampoline. Pass `None` to clear.
pub fn set_residual_host_call(hook: Option<ResidualHostCallFn>) {
    RESIDUAL_HOST_CALL.with(|c| c.set(hook));
}

/// The active residual-call host trampoline, or `None` for direct transmute.
pub fn residual_host_call() -> Option<ResidualHostCallFn> {
    RESIDUAL_HOST_CALL.with(|c| c.get())
}

#[cfg(test)]
mod tests {
    use super::*;

    extern "C" fn f2(a: f64, b: *const i64) -> f64 {
        a + unsafe { *b } as f64
    }

    extern "C" fn int_float_int(a: i64, b: f64, c: i64) -> i64 {
        a + b as i64 * 10 + c * 100
    }

    extern "C" fn float_float_int(a: f64, b: f64, c: i64) -> f64 {
        a + b * 10.0 + c as f64 * 100.0
    }

    /// Rust port of
    /// `rpython/jit/backend/llsupport/test/test_descr.py::test_call_stubs_2`.
    #[test]
    fn call_stub_f_interleaved_float_ref_preserves_declaration_order() {
        let b = [1_i64];
        let result = unsafe {
            bh_call_f_dispatch(
                f2 as *const () as usize,
                &[ArgClass::Float, ArgClass::Int],
                &[3.5_f64.to_bits() as i64, b.as_ptr() as i64],
            )
        };
        assert_eq!(result, 4.5);
    }

    #[test]
    fn call_stub_i_interleaved_int_float_int_preserves_declaration_order() {
        let result = unsafe {
            bh_call_i_dispatch(
                int_float_int as *const () as usize,
                &[ArgClass::Int, ArgClass::Float, ArgClass::Int],
                &[1, 2.0_f64.to_bits() as i64, 3],
            )
        };
        assert_eq!(result, 321);
    }

    #[test]
    fn call_stub_f_interleaved_float_float_int_preserves_declaration_order() {
        let result = unsafe {
            bh_call_f_dispatch(
                float_float_int as *const () as usize,
                &[ArgClass::Float, ArgClass::Float, ArgClass::Int],
                &[1.0_f64.to_bits() as i64, 2.0_f64.to_bits() as i64, 3],
            )
        };
        assert_eq!(result, 321.0);
    }

    extern "C" fn four_ints_then_float(a: i64, b: i64, c: i64, d: i64, e: f64) -> i64 {
        a + b * 10 + c * 100 + d * 1000 + e as i64 * 10000
    }

    /// The widest float-carrying sequence the table covers, and the bound
    /// `majit_translate::codewriter::jitcode::MAX_FLOAT_CARRYING_CALL_ARITY`
    /// states on the descr-build side.
    #[test]
    fn call_stub_i_dispatches_a_float_in_the_last_covered_slot() {
        let result = unsafe {
            bh_call_i_dispatch(
                four_ints_then_float as *const () as usize,
                &[
                    ArgClass::Int,
                    ArgClass::Int,
                    ArgClass::Int,
                    ArgClass::Int,
                    ArgClass::Float,
                ],
                &[1, 2, 3, 4, 5.0_f64.to_bits() as i64],
            )
        };
        assert_eq!(result, 54321);
    }

    /// One argument past that bound the table has no arm, so the call is
    /// refused instead of being placed against the wrong signature.
    #[test]
    #[should_panic(expected = "unsupported arg class sequence")]
    fn call_stub_i_refuses_a_float_past_the_covered_width() {
        unsafe {
            bh_call_i_dispatch(
                four_ints_then_float as *const () as usize,
                &[
                    ArgClass::Int,
                    ArgClass::Int,
                    ArgClass::Int,
                    ArgClass::Int,
                    ArgClass::Int,
                    ArgClass::Float,
                ],
                &[1, 2, 3, 4, 5, 6.0_f64.to_bits() as i64],
            );
        }
    }
}
