//! C-ABI call stub dispatch shared between backends.
//!
//! `bh_call_i_dispatch` mirrors `rpython/jit/backend/llsupport/llmodel.py bh_call_i call_stub_i`:
//! it materializes a typed `extern "C" fn` from a raw funcptr and forwards
//! arguments in the calldescr declaration order preserved by `arg_classes`.
//! `bh_call_f_dispatch` and `bh_call_v_dispatch` are the float-returning and
//! void-returning parallels for `llmodel.py bh_call_f` and
//! `llmodel.py bh_call_v`.
//!
//! Upstream `descr.py:574` builds the generated call expression by walking
//! `self.arg_classes`, and `descr.py` builds `FuncType(ARGS, RESULT)`
//! from that same ordered class list. Matching that shape is required by the
//! Microsoft x64 positional-slot convention and is also correct under SysV and
//! AAPCS.

use majit_jitcode::codewriter::insns::MAX_HOST_CALL_ARITY;
use majit_jitcode::jitcode::{BhCallDescr, BhCallStub};

/// `descr.py TYPE()` collapsed to the two C-ABI register classes the
/// dispatch table can express: `'i'`, `'r'` and `'L'` (`lltype.Signed`,
/// `llmemory.GCREF`, `lltype.SignedLongLong`) all pass in an integer register;
/// `'f'` (`lltype.Float`) passes in a floating-point register.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ArgClass {
    Int,
    /// GCREF. Staged wrappers pass it as an `i64` word, including on wasm32.
    /// A callee whose table type is a real `i32` pointer is host-reflected.
    Ref,
    Float,
}

/// Class-sequence table shared by `bh_call_i_dispatch`, `bh_call_f_dispatch`,
/// and `bh_call_v_dispatch`. Each row is one monomorphic stub module; the
/// same list is the match in `lookup_stub_*` and the arms `dispatch_classes_body!`
/// used to inline. `$ret` plugs into both the function-pointer signature
/// and the dispatch function's return type.
///
/// `descr.py` / `descr.py CallDescr.create_call_stub` parity: the signature
/// is built in `arg_classes` declaration order, so `fn(f64, i64)` is dispatched
/// as `extern "C" fn(f64, i64)` rather than a class-blind
/// `extern "C" fn(i64, f64)`. That preserves SysV/AAPCS register-file order
/// and the Microsoft x64 positional argument slots alike.
///
/// Coverage: every ordered sequence up to 5 arguments, plus the all-`Int`
/// sequences on to `MAX_HOST_CALL_ARITY`. The float-carrying bound is mirrored
/// by `majit_jitcode::codewriter::jitcode::MAX_FLOAT_CARRYING_CALL_ARITY`,
/// which flags such a signature where the calldescr is built instead of at the
/// deopt that first runs it; widening the arms here means raising it there in
/// the same change (`majit-translate` cannot call into `majit-backend`, so the
/// bound is stated on both sides rather than shared).
#[cfg(not(target_arch = "wasm32"))]
macro_rules! invoke_ty {
    (Int) => {
        i64
    };
    (Ref) => {
        i64
    };
    (Float) => {
        f64
    };
}

#[cfg(target_arch = "wasm32")]
macro_rules! invoke_ty {
    (Int) => {
        i64
    };
    (Ref) => {
        i64
    };
    (Float) => {
        f64
    };
}

#[cfg(not(target_arch = "wasm32"))]
macro_rules! invoke_arg {
    (Int, $a:ident, $i:tt) => {
        $a[$i]
    };
    (Ref, $a:ident, $i:tt) => {
        $a[$i]
    };
    (Float, $a:ident, $i:tt) => {
        f64::from_bits($a[$i] as u64)
    };
}

#[cfg(target_arch = "wasm32")]
macro_rules! invoke_arg {
    (Int, $a:ident, $i:tt) => {
        $a[$i]
    };
    (Ref, $a:ident, $i:tt) => {
        $a[$i]
    };
    (Float, $a:ident, $i:tt) => {
        f64::from_bits($a[$i] as u64)
    };
}

macro_rules! invoke_with_idx {
    ($f:ident, $a:ident) => {{
        let _ = $a;
        $f()
    }};
    ($f:ident, $a:ident, $c0:ident) => {
        $f(invoke_arg!($c0, $a, 0))
    };
    ($f:ident, $a:ident, $c0:ident, $c1:ident) => {
        $f(invoke_arg!($c0, $a, 0), invoke_arg!($c1, $a, 1))
    };
    ($f:ident, $a:ident, $c0:ident, $c1:ident, $c2:ident) => {
        $f(
            invoke_arg!($c0, $a, 0),
            invoke_arg!($c1, $a, 1),
            invoke_arg!($c2, $a, 2),
        )
    };
    ($f:ident, $a:ident, $c0:ident, $c1:ident, $c2:ident, $c3:ident) => {
        $f(
            invoke_arg!($c0, $a, 0),
            invoke_arg!($c1, $a, 1),
            invoke_arg!($c2, $a, 2),
            invoke_arg!($c3, $a, 3),
        )
    };
    ($f:ident, $a:ident, $c0:ident, $c1:ident, $c2:ident, $c3:ident, $c4:ident) => {
        $f(
            invoke_arg!($c0, $a, 0),
            invoke_arg!($c1, $a, 1),
            invoke_arg!($c2, $a, 2),
            invoke_arg!($c3, $a, 3),
            invoke_arg!($c4, $a, 4),
        )
    };
    ($f:ident, $a:ident, $c0:ident, $c1:ident, $c2:ident, $c3:ident, $c4:ident, $c5:ident) => {
        $f(
            invoke_arg!($c0, $a, 0),
            invoke_arg!($c1, $a, 1),
            invoke_arg!($c2, $a, 2),
            invoke_arg!($c3, $a, 3),
            invoke_arg!($c4, $a, 4),
            invoke_arg!($c5, $a, 5),
        )
    };
    ($f:ident, $a:ident, $c0:ident, $c1:ident, $c2:ident, $c3:ident, $c4:ident, $c5:ident, $c6:ident) => {
        $f(
            invoke_arg!($c0, $a, 0),
            invoke_arg!($c1, $a, 1),
            invoke_arg!($c2, $a, 2),
            invoke_arg!($c3, $a, 3),
            invoke_arg!($c4, $a, 4),
            invoke_arg!($c5, $a, 5),
            invoke_arg!($c6, $a, 6),
        )
    };
    ($f:ident, $a:ident, $c0:ident, $c1:ident, $c2:ident, $c3:ident, $c4:ident, $c5:ident, $c6:ident, $c7:ident) => {
        $f(
            invoke_arg!($c0, $a, 0),
            invoke_arg!($c1, $a, 1),
            invoke_arg!($c2, $a, 2),
            invoke_arg!($c3, $a, 3),
            invoke_arg!($c4, $a, 4),
            invoke_arg!($c5, $a, 5),
            invoke_arg!($c6, $a, 6),
            invoke_arg!($c7, $a, 7),
        )
    };
    ($f:ident, $a:ident, $c0:ident, $c1:ident, $c2:ident, $c3:ident, $c4:ident, $c5:ident, $c6:ident, $c7:ident, $c8:ident) => {
        $f(
            invoke_arg!($c0, $a, 0),
            invoke_arg!($c1, $a, 1),
            invoke_arg!($c2, $a, 2),
            invoke_arg!($c3, $a, 3),
            invoke_arg!($c4, $a, 4),
            invoke_arg!($c5, $a, 5),
            invoke_arg!($c6, $a, 6),
            invoke_arg!($c7, $a, 7),
            invoke_arg!($c8, $a, 8),
        )
    };
    ($f:ident, $a:ident, $c0:ident, $c1:ident, $c2:ident, $c3:ident, $c4:ident, $c5:ident, $c6:ident, $c7:ident, $c8:ident, $c9:ident) => {
        $f(
            invoke_arg!($c0, $a, 0),
            invoke_arg!($c1, $a, 1),
            invoke_arg!($c2, $a, 2),
            invoke_arg!($c3, $a, 3),
            invoke_arg!($c4, $a, 4),
            invoke_arg!($c5, $a, 5),
            invoke_arg!($c6, $a, 6),
            invoke_arg!($c7, $a, 7),
            invoke_arg!($c8, $a, 8),
            invoke_arg!($c9, $a, 9),
        )
    };
    ($f:ident, $a:ident, $c0:ident, $c1:ident, $c2:ident, $c3:ident, $c4:ident, $c5:ident, $c6:ident, $c7:ident, $c8:ident, $c9:ident, $c10:ident) => {
        $f(
            invoke_arg!($c0, $a, 0),
            invoke_arg!($c1, $a, 1),
            invoke_arg!($c2, $a, 2),
            invoke_arg!($c3, $a, 3),
            invoke_arg!($c4, $a, 4),
            invoke_arg!($c5, $a, 5),
            invoke_arg!($c6, $a, 6),
            invoke_arg!($c7, $a, 7),
            invoke_arg!($c8, $a, 8),
            invoke_arg!($c9, $a, 9),
            invoke_arg!($c10, $a, 10),
        )
    };
    ($f:ident, $a:ident, $c0:ident, $c1:ident, $c2:ident, $c3:ident, $c4:ident, $c5:ident, $c6:ident, $c7:ident, $c8:ident, $c9:ident, $c10:ident, $c11:ident) => {
        $f(
            invoke_arg!($c0, $a, 0),
            invoke_arg!($c1, $a, 1),
            invoke_arg!($c2, $a, 2),
            invoke_arg!($c3, $a, 3),
            invoke_arg!($c4, $a, 4),
            invoke_arg!($c5, $a, 5),
            invoke_arg!($c6, $a, 6),
            invoke_arg!($c7, $a, 7),
            invoke_arg!($c8, $a, 8),
            invoke_arg!($c9, $a, 9),
            invoke_arg!($c10, $a, 10),
            invoke_arg!($c11, $a, 11),
        )
    };
    ($f:ident, $a:ident, $c0:ident, $c1:ident, $c2:ident, $c3:ident, $c4:ident, $c5:ident, $c6:ident, $c7:ident, $c8:ident, $c9:ident, $c10:ident, $c11:ident, $c12:ident) => {
        $f(
            invoke_arg!($c0, $a, 0),
            invoke_arg!($c1, $a, 1),
            invoke_arg!($c2, $a, 2),
            invoke_arg!($c3, $a, 3),
            invoke_arg!($c4, $a, 4),
            invoke_arg!($c5, $a, 5),
            invoke_arg!($c6, $a, 6),
            invoke_arg!($c7, $a, 7),
            invoke_arg!($c8, $a, 8),
            invoke_arg!($c9, $a, 9),
            invoke_arg!($c10, $a, 10),
            invoke_arg!($c11, $a, 11),
            invoke_arg!($c12, $a, 12),
        )
    };
    ($f:ident, $a:ident, $c0:ident, $c1:ident, $c2:ident, $c3:ident, $c4:ident, $c5:ident, $c6:ident, $c7:ident, $c8:ident, $c9:ident, $c10:ident, $c11:ident, $c12:ident, $c13:ident) => {
        $f(
            invoke_arg!($c0, $a, 0),
            invoke_arg!($c1, $a, 1),
            invoke_arg!($c2, $a, 2),
            invoke_arg!($c3, $a, 3),
            invoke_arg!($c4, $a, 4),
            invoke_arg!($c5, $a, 5),
            invoke_arg!($c6, $a, 6),
            invoke_arg!($c7, $a, 7),
            invoke_arg!($c8, $a, 8),
            invoke_arg!($c9, $a, 9),
            invoke_arg!($c10, $a, 10),
            invoke_arg!($c11, $a, 11),
            invoke_arg!($c12, $a, 12),
            invoke_arg!($c13, $a, 13),
        )
    };
    ($f:ident, $a:ident, $c0:ident, $c1:ident, $c2:ident, $c3:ident, $c4:ident, $c5:ident, $c6:ident, $c7:ident, $c8:ident, $c9:ident, $c10:ident, $c11:ident, $c12:ident, $c13:ident, $c14:ident) => {
        $f(
            invoke_arg!($c0, $a, 0),
            invoke_arg!($c1, $a, 1),
            invoke_arg!($c2, $a, 2),
            invoke_arg!($c3, $a, 3),
            invoke_arg!($c4, $a, 4),
            invoke_arg!($c5, $a, 5),
            invoke_arg!($c6, $a, 6),
            invoke_arg!($c7, $a, 7),
            invoke_arg!($c8, $a, 8),
            invoke_arg!($c9, $a, 9),
            invoke_arg!($c10, $a, 10),
            invoke_arg!($c11, $a, 11),
            invoke_arg!($c12, $a, 12),
            invoke_arg!($c13, $a, 13),
            invoke_arg!($c14, $a, 14),
        )
    };
    ($f:ident, $a:ident, $c0:ident, $c1:ident, $c2:ident, $c3:ident, $c4:ident, $c5:ident, $c6:ident, $c7:ident, $c8:ident, $c9:ident, $c10:ident, $c11:ident, $c12:ident, $c13:ident, $c14:ident, $c15:ident) => {
        $f(
            invoke_arg!($c0, $a, 0),
            invoke_arg!($c1, $a, 1),
            invoke_arg!($c2, $a, 2),
            invoke_arg!($c3, $a, 3),
            invoke_arg!($c4, $a, 4),
            invoke_arg!($c5, $a, 5),
            invoke_arg!($c6, $a, 6),
            invoke_arg!($c7, $a, 7),
            invoke_arg!($c8, $a, 8),
            invoke_arg!($c9, $a, 9),
            invoke_arg!($c10, $a, 10),
            invoke_arg!($c11, $a, 11),
            invoke_arg!($c12, $a, 12),
            invoke_arg!($c13, $a, 13),
            invoke_arg!($c14, $a, 14),
            invoke_arg!($c15, $a, 15),
        )
    };
}

macro_rules! invoke_stub {
    ($func:ident, $args:ident, $ret:ty $(, $class:ident)*) => {{
        let f: unsafe extern "C" fn($(invoke_ty!($class)),*) -> $ret =
            std::mem::transmute($func);
        invoke_with_idx!(f, $args $(, $class)*)
    }};
}

/// One module per ABI sequence: `call_i` / `call_f` / `call_v` are the
/// monomorphic stubs `descr.py CallDescr.create_call_stub` would emit.
/// `dispatch_classes_body!` is this table (via `lookup_stub_*`).
macro_rules! call_sig_table {
    ($apply:ident) => {
        $apply! {
        Z {  }
        I { Int }
        F { Float }
        II { Int, Int }
        IF { Int, Float }
        FI { Float, Int }
        FF { Float, Float }
        III { Int, Int, Int }
        IIF { Int, Int, Float }
        IFI { Int, Float, Int }
        IFF { Int, Float, Float }
        FII { Float, Int, Int }
        FIF { Float, Int, Float }
        FFI { Float, Float, Int }
        FFF { Float, Float, Float }
        IIII { Int, Int, Int, Int }
        IIIF { Int, Int, Int, Float }
        IIFI { Int, Int, Float, Int }
        IIFF { Int, Int, Float, Float }
        IFII { Int, Float, Int, Int }
        IFIF { Int, Float, Int, Float }
        IFFI { Int, Float, Float, Int }
        IFFF { Int, Float, Float, Float }
        FIII { Float, Int, Int, Int }
        FIIF { Float, Int, Int, Float }
        FIFI { Float, Int, Float, Int }
        FIFF { Float, Int, Float, Float }
        FFII { Float, Float, Int, Int }
        FFIF { Float, Float, Int, Float }
        FFFI { Float, Float, Float, Int }
        FFFF { Float, Float, Float, Float }
        IIIII { Int, Int, Int, Int, Int }
        IIIIF { Int, Int, Int, Int, Float }
        IIIFI { Int, Int, Int, Float, Int }
        IIIFF { Int, Int, Int, Float, Float }
        IIFII { Int, Int, Float, Int, Int }
        IIFIF { Int, Int, Float, Int, Float }
        IIFFI { Int, Int, Float, Float, Int }
        IIFFF { Int, Int, Float, Float, Float }
        IFIII { Int, Float, Int, Int, Int }
        IFIIF { Int, Float, Int, Int, Float }
        IFIFI { Int, Float, Int, Float, Int }
        IFIFF { Int, Float, Int, Float, Float }
        IFFII { Int, Float, Float, Int, Int }
        IFFIF { Int, Float, Float, Int, Float }
        IFFFI { Int, Float, Float, Float, Int }
        IFFFF { Int, Float, Float, Float, Float }
        FIIII { Float, Int, Int, Int, Int }
        FIIIF { Float, Int, Int, Int, Float }
        FIIFI { Float, Int, Int, Float, Int }
        FIIFF { Float, Int, Int, Float, Float }
        FIFII { Float, Int, Float, Int, Int }
        FIFIF { Float, Int, Float, Int, Float }
        FIFFI { Float, Int, Float, Float, Int }
        FIFFF { Float, Int, Float, Float, Float }
        FFIII { Float, Float, Int, Int, Int }
        FFIIF { Float, Float, Int, Int, Float }
        FFIFI { Float, Float, Int, Float, Int }
        FFIFF { Float, Float, Int, Float, Float }
        FFFII { Float, Float, Float, Int, Int }
        FFFIF { Float, Float, Float, Int, Float }
        FFFFI { Float, Float, Float, Float, Int }
        FFFFF { Float, Float, Float, Float, Float }
        R { Ref }
        RI { Ref, Int }
        IR { Int, Ref }
        RR { Ref, Ref }
        FR { Float, Ref }
        RF { Ref, Float }
        RII { Ref, Int, Int }
        IRI { Int, Ref, Int }
        RRI { Ref, Ref, Int }
        FRI { Float, Ref, Int }
        RFI { Ref, Float, Int }
        IIR { Int, Int, Ref }
        RIR { Ref, Int, Ref }
        FIR { Float, Int, Ref }
        IRR { Int, Ref, Ref }
        RRR { Ref, Ref, Ref }
        FRR { Float, Ref, Ref }
        IFR { Int, Float, Ref }
        RFR { Ref, Float, Ref }
        FFR { Float, Float, Ref }
        RIF { Ref, Int, Float }
        IRF { Int, Ref, Float }
        RRF { Ref, Ref, Float }
        FRF { Float, Ref, Float }
        RFF { Ref, Float, Float }
        RIII { Ref, Int, Int, Int }
        IRII { Int, Ref, Int, Int }
        RRII { Ref, Ref, Int, Int }
        FRII { Float, Ref, Int, Int }
        RFII { Ref, Float, Int, Int }
        IIRI { Int, Int, Ref, Int }
        RIRI { Ref, Int, Ref, Int }
        FIRI { Float, Int, Ref, Int }
        IRRI { Int, Ref, Ref, Int }
        RRRI { Ref, Ref, Ref, Int }
        FRRI { Float, Ref, Ref, Int }
        IFRI { Int, Float, Ref, Int }
        RFRI { Ref, Float, Ref, Int }
        FFRI { Float, Float, Ref, Int }
        RIFI { Ref, Int, Float, Int }
        IRFI { Int, Ref, Float, Int }
        RRFI { Ref, Ref, Float, Int }
        FRFI { Float, Ref, Float, Int }
        RFFI { Ref, Float, Float, Int }
        IIIR { Int, Int, Int, Ref }
        RIIR { Ref, Int, Int, Ref }
        FIIR { Float, Int, Int, Ref }
        IRIR { Int, Ref, Int, Ref }
        RRIR { Ref, Ref, Int, Ref }
        FRIR { Float, Ref, Int, Ref }
        IFIR { Int, Float, Int, Ref }
        RFIR { Ref, Float, Int, Ref }
        FFIR { Float, Float, Int, Ref }
        IIRR { Int, Int, Ref, Ref }
        RIRR { Ref, Int, Ref, Ref }
        FIRR { Float, Int, Ref, Ref }
        IRRR { Int, Ref, Ref, Ref }
        RRRR { Ref, Ref, Ref, Ref }
        FRRR { Float, Ref, Ref, Ref }
        IFRR { Int, Float, Ref, Ref }
        RFRR { Ref, Float, Ref, Ref }
        FFRR { Float, Float, Ref, Ref }
        IIFR { Int, Int, Float, Ref }
        RIFR { Ref, Int, Float, Ref }
        FIFR { Float, Int, Float, Ref }
        IRFR { Int, Ref, Float, Ref }
        RRFR { Ref, Ref, Float, Ref }
        FRFR { Float, Ref, Float, Ref }
        IFFR { Int, Float, Float, Ref }
        RFFR { Ref, Float, Float, Ref }
        FFFR { Float, Float, Float, Ref }
        RIIF { Ref, Int, Int, Float }
        IRIF { Int, Ref, Int, Float }
        RRIF { Ref, Ref, Int, Float }
        FRIF { Float, Ref, Int, Float }
        RFIF { Ref, Float, Int, Float }
        IIRF { Int, Int, Ref, Float }
        RIRF { Ref, Int, Ref, Float }
        FIRF { Float, Int, Ref, Float }
        IRRF { Int, Ref, Ref, Float }
        RRRF { Ref, Ref, Ref, Float }
        FRRF { Float, Ref, Ref, Float }
        IFRF { Int, Float, Ref, Float }
        RFRF { Ref, Float, Ref, Float }
        FFRF { Float, Float, Ref, Float }
        RIFF { Ref, Int, Float, Float }
        IRFF { Int, Ref, Float, Float }
        RRFF { Ref, Ref, Float, Float }
        FRFF { Float, Ref, Float, Float }
        RFFF { Ref, Float, Float, Float }
        RIIII { Ref, Int, Int, Int, Int }
        IRIII { Int, Ref, Int, Int, Int }
        RRIII { Ref, Ref, Int, Int, Int }
        FRIII { Float, Ref, Int, Int, Int }
        RFIII { Ref, Float, Int, Int, Int }
        IIRII { Int, Int, Ref, Int, Int }
        RIRII { Ref, Int, Ref, Int, Int }
        FIRII { Float, Int, Ref, Int, Int }
        IRRII { Int, Ref, Ref, Int, Int }
        RRRII { Ref, Ref, Ref, Int, Int }
        FRRII { Float, Ref, Ref, Int, Int }
        IFRII { Int, Float, Ref, Int, Int }
        RFRII { Ref, Float, Ref, Int, Int }
        FFRII { Float, Float, Ref, Int, Int }
        RIFII { Ref, Int, Float, Int, Int }
        IRFII { Int, Ref, Float, Int, Int }
        RRFII { Ref, Ref, Float, Int, Int }
        FRFII { Float, Ref, Float, Int, Int }
        RFFII { Ref, Float, Float, Int, Int }
        IIIRI { Int, Int, Int, Ref, Int }
        RIIRI { Ref, Int, Int, Ref, Int }
        FIIRI { Float, Int, Int, Ref, Int }
        IRIRI { Int, Ref, Int, Ref, Int }
        RRIRI { Ref, Ref, Int, Ref, Int }
        FRIRI { Float, Ref, Int, Ref, Int }
        IFIRI { Int, Float, Int, Ref, Int }
        RFIRI { Ref, Float, Int, Ref, Int }
        FFIRI { Float, Float, Int, Ref, Int }
        IIRRI { Int, Int, Ref, Ref, Int }
        RIRRI { Ref, Int, Ref, Ref, Int }
        FIRRI { Float, Int, Ref, Ref, Int }
        IRRRI { Int, Ref, Ref, Ref, Int }
        RRRRI { Ref, Ref, Ref, Ref, Int }
        FRRRI { Float, Ref, Ref, Ref, Int }
        IFRRI { Int, Float, Ref, Ref, Int }
        RFRRI { Ref, Float, Ref, Ref, Int }
        FFRRI { Float, Float, Ref, Ref, Int }
        IIFRI { Int, Int, Float, Ref, Int }
        RIFRI { Ref, Int, Float, Ref, Int }
        FIFRI { Float, Int, Float, Ref, Int }
        IRFRI { Int, Ref, Float, Ref, Int }
        RRFRI { Ref, Ref, Float, Ref, Int }
        FRFRI { Float, Ref, Float, Ref, Int }
        IFFRI { Int, Float, Float, Ref, Int }
        RFFRI { Ref, Float, Float, Ref, Int }
        FFFRI { Float, Float, Float, Ref, Int }
        RIIFI { Ref, Int, Int, Float, Int }
        IRIFI { Int, Ref, Int, Float, Int }
        RRIFI { Ref, Ref, Int, Float, Int }
        FRIFI { Float, Ref, Int, Float, Int }
        RFIFI { Ref, Float, Int, Float, Int }
        IIRFI { Int, Int, Ref, Float, Int }
        RIRFI { Ref, Int, Ref, Float, Int }
        FIRFI { Float, Int, Ref, Float, Int }
        IRRFI { Int, Ref, Ref, Float, Int }
        RRRFI { Ref, Ref, Ref, Float, Int }
        FRRFI { Float, Ref, Ref, Float, Int }
        IFRFI { Int, Float, Ref, Float, Int }
        RFRFI { Ref, Float, Ref, Float, Int }
        FFRFI { Float, Float, Ref, Float, Int }
        RIFFI { Ref, Int, Float, Float, Int }
        IRFFI { Int, Ref, Float, Float, Int }
        RRFFI { Ref, Ref, Float, Float, Int }
        FRFFI { Float, Ref, Float, Float, Int }
        RFFFI { Ref, Float, Float, Float, Int }
        IIIIR { Int, Int, Int, Int, Ref }
        RIIIR { Ref, Int, Int, Int, Ref }
        FIIIR { Float, Int, Int, Int, Ref }
        IRIIR { Int, Ref, Int, Int, Ref }
        RRIIR { Ref, Ref, Int, Int, Ref }
        FRIIR { Float, Ref, Int, Int, Ref }
        IFIIR { Int, Float, Int, Int, Ref }
        RFIIR { Ref, Float, Int, Int, Ref }
        FFIIR { Float, Float, Int, Int, Ref }
        IIRIR { Int, Int, Ref, Int, Ref }
        RIRIR { Ref, Int, Ref, Int, Ref }
        FIRIR { Float, Int, Ref, Int, Ref }
        IRRIR { Int, Ref, Ref, Int, Ref }
        RRRIR { Ref, Ref, Ref, Int, Ref }
        FRRIR { Float, Ref, Ref, Int, Ref }
        IFRIR { Int, Float, Ref, Int, Ref }
        RFRIR { Ref, Float, Ref, Int, Ref }
        FFRIR { Float, Float, Ref, Int, Ref }
        IIFIR { Int, Int, Float, Int, Ref }
        RIFIR { Ref, Int, Float, Int, Ref }
        FIFIR { Float, Int, Float, Int, Ref }
        IRFIR { Int, Ref, Float, Int, Ref }
        RRFIR { Ref, Ref, Float, Int, Ref }
        FRFIR { Float, Ref, Float, Int, Ref }
        IFFIR { Int, Float, Float, Int, Ref }
        RFFIR { Ref, Float, Float, Int, Ref }
        FFFIR { Float, Float, Float, Int, Ref }
        IIIRR { Int, Int, Int, Ref, Ref }
        RIIRR { Ref, Int, Int, Ref, Ref }
        FIIRR { Float, Int, Int, Ref, Ref }
        IRIRR { Int, Ref, Int, Ref, Ref }
        RRIRR { Ref, Ref, Int, Ref, Ref }
        FRIRR { Float, Ref, Int, Ref, Ref }
        IFIRR { Int, Float, Int, Ref, Ref }
        RFIRR { Ref, Float, Int, Ref, Ref }
        FFIRR { Float, Float, Int, Ref, Ref }
        IIRRR { Int, Int, Ref, Ref, Ref }
        RIRRR { Ref, Int, Ref, Ref, Ref }
        FIRRR { Float, Int, Ref, Ref, Ref }
        IRRRR { Int, Ref, Ref, Ref, Ref }
        RRRRR { Ref, Ref, Ref, Ref, Ref }
        FRRRR { Float, Ref, Ref, Ref, Ref }
        IFRRR { Int, Float, Ref, Ref, Ref }
        RFRRR { Ref, Float, Ref, Ref, Ref }
        FFRRR { Float, Float, Ref, Ref, Ref }
        IIFRR { Int, Int, Float, Ref, Ref }
        RIFRR { Ref, Int, Float, Ref, Ref }
        FIFRR { Float, Int, Float, Ref, Ref }
        IRFRR { Int, Ref, Float, Ref, Ref }
        RRFRR { Ref, Ref, Float, Ref, Ref }
        FRFRR { Float, Ref, Float, Ref, Ref }
        IFFRR { Int, Float, Float, Ref, Ref }
        RFFRR { Ref, Float, Float, Ref, Ref }
        FFFRR { Float, Float, Float, Ref, Ref }
        IIIFR { Int, Int, Int, Float, Ref }
        RIIFR { Ref, Int, Int, Float, Ref }
        FIIFR { Float, Int, Int, Float, Ref }
        IRIFR { Int, Ref, Int, Float, Ref }
        RRIFR { Ref, Ref, Int, Float, Ref }
        FRIFR { Float, Ref, Int, Float, Ref }
        IFIFR { Int, Float, Int, Float, Ref }
        RFIFR { Ref, Float, Int, Float, Ref }
        FFIFR { Float, Float, Int, Float, Ref }
        IIRFR { Int, Int, Ref, Float, Ref }
        RIRFR { Ref, Int, Ref, Float, Ref }
        FIRFR { Float, Int, Ref, Float, Ref }
        IRRFR { Int, Ref, Ref, Float, Ref }
        RRRFR { Ref, Ref, Ref, Float, Ref }
        FRRFR { Float, Ref, Ref, Float, Ref }
        IFRFR { Int, Float, Ref, Float, Ref }
        RFRFR { Ref, Float, Ref, Float, Ref }
        FFRFR { Float, Float, Ref, Float, Ref }
        IIFFR { Int, Int, Float, Float, Ref }
        RIFFR { Ref, Int, Float, Float, Ref }
        FIFFR { Float, Int, Float, Float, Ref }
        IRFFR { Int, Ref, Float, Float, Ref }
        RRFFR { Ref, Ref, Float, Float, Ref }
        FRFFR { Float, Ref, Float, Float, Ref }
        IFFFR { Int, Float, Float, Float, Ref }
        RFFFR { Ref, Float, Float, Float, Ref }
        FFFFR { Float, Float, Float, Float, Ref }
        RIIIF { Ref, Int, Int, Int, Float }
        IRIIF { Int, Ref, Int, Int, Float }
        RRIIF { Ref, Ref, Int, Int, Float }
        FRIIF { Float, Ref, Int, Int, Float }
        RFIIF { Ref, Float, Int, Int, Float }
        IIRIF { Int, Int, Ref, Int, Float }
        RIRIF { Ref, Int, Ref, Int, Float }
        FIRIF { Float, Int, Ref, Int, Float }
        IRRIF { Int, Ref, Ref, Int, Float }
        RRRIF { Ref, Ref, Ref, Int, Float }
        FRRIF { Float, Ref, Ref, Int, Float }
        IFRIF { Int, Float, Ref, Int, Float }
        RFRIF { Ref, Float, Ref, Int, Float }
        FFRIF { Float, Float, Ref, Int, Float }
        RIFIF { Ref, Int, Float, Int, Float }
        IRFIF { Int, Ref, Float, Int, Float }
        RRFIF { Ref, Ref, Float, Int, Float }
        FRFIF { Float, Ref, Float, Int, Float }
        RFFIF { Ref, Float, Float, Int, Float }
        IIIRF { Int, Int, Int, Ref, Float }
        RIIRF { Ref, Int, Int, Ref, Float }
        FIIRF { Float, Int, Int, Ref, Float }
        IRIRF { Int, Ref, Int, Ref, Float }
        RRIRF { Ref, Ref, Int, Ref, Float }
        FRIRF { Float, Ref, Int, Ref, Float }
        IFIRF { Int, Float, Int, Ref, Float }
        RFIRF { Ref, Float, Int, Ref, Float }
        FFIRF { Float, Float, Int, Ref, Float }
        IIRRF { Int, Int, Ref, Ref, Float }
        RIRRF { Ref, Int, Ref, Ref, Float }
        FIRRF { Float, Int, Ref, Ref, Float }
        IRRRF { Int, Ref, Ref, Ref, Float }
        RRRRF { Ref, Ref, Ref, Ref, Float }
        FRRRF { Float, Ref, Ref, Ref, Float }
        IFRRF { Int, Float, Ref, Ref, Float }
        RFRRF { Ref, Float, Ref, Ref, Float }
        FFRRF { Float, Float, Ref, Ref, Float }
        IIFRF { Int, Int, Float, Ref, Float }
        RIFRF { Ref, Int, Float, Ref, Float }
        FIFRF { Float, Int, Float, Ref, Float }
        IRFRF { Int, Ref, Float, Ref, Float }
        RRFRF { Ref, Ref, Float, Ref, Float }
        FRFRF { Float, Ref, Float, Ref, Float }
        IFFRF { Int, Float, Float, Ref, Float }
        RFFRF { Ref, Float, Float, Ref, Float }
        FFFRF { Float, Float, Float, Ref, Float }
        RIIFF { Ref, Int, Int, Float, Float }
        IRIFF { Int, Ref, Int, Float, Float }
        RRIFF { Ref, Ref, Int, Float, Float }
        FRIFF { Float, Ref, Int, Float, Float }
        RFIFF { Ref, Float, Int, Float, Float }
        IIRFF { Int, Int, Ref, Float, Float }
        RIRFF { Ref, Int, Ref, Float, Float }
        FIRFF { Float, Int, Ref, Float, Float }
        IRRFF { Int, Ref, Ref, Float, Float }
        RRRFF { Ref, Ref, Ref, Float, Float }
        FRRFF { Float, Ref, Ref, Float, Float }
        IFRFF { Int, Float, Ref, Float, Float }
        RFRFF { Ref, Float, Ref, Float, Float }
        FFRFF { Float, Float, Ref, Float, Float }
        RIFFF { Ref, Int, Float, Float, Float }
        IRFFF { Int, Ref, Float, Float, Float }
        RRFFF { Ref, Ref, Float, Float, Float }
        FRFFF { Float, Ref, Float, Float, Float }
        RFFFF { Ref, Float, Float, Float, Float }
        RIIIII { Ref, Int, Int, Int, Int, Int }
        IRIIII { Int, Ref, Int, Int, Int, Int }
        RRIIII { Ref, Ref, Int, Int, Int, Int }
        IIRIII { Int, Int, Ref, Int, Int, Int }
        RIRIII { Ref, Int, Ref, Int, Int, Int }
        IRRIII { Int, Ref, Ref, Int, Int, Int }
        RRRIII { Ref, Ref, Ref, Int, Int, Int }
        IIIRII { Int, Int, Int, Ref, Int, Int }
        RIIRII { Ref, Int, Int, Ref, Int, Int }
        IRIRII { Int, Ref, Int, Ref, Int, Int }
        RRIRII { Ref, Ref, Int, Ref, Int, Int }
        IIRRII { Int, Int, Ref, Ref, Int, Int }
        RIRRII { Ref, Int, Ref, Ref, Int, Int }
        IRRRII { Int, Ref, Ref, Ref, Int, Int }
        RRRRII { Ref, Ref, Ref, Ref, Int, Int }
        IIIIRI { Int, Int, Int, Int, Ref, Int }
        RIIIRI { Ref, Int, Int, Int, Ref, Int }
        IRIIRI { Int, Ref, Int, Int, Ref, Int }
        RRIIRI { Ref, Ref, Int, Int, Ref, Int }
        IIRIRI { Int, Int, Ref, Int, Ref, Int }
        RIRIRI { Ref, Int, Ref, Int, Ref, Int }
        IRRIRI { Int, Ref, Ref, Int, Ref, Int }
        RRRIRI { Ref, Ref, Ref, Int, Ref, Int }
        IIIRRI { Int, Int, Int, Ref, Ref, Int }
        RIIRRI { Ref, Int, Int, Ref, Ref, Int }
        IRIRRI { Int, Ref, Int, Ref, Ref, Int }
        RRIRRI { Ref, Ref, Int, Ref, Ref, Int }
        IIRRRI { Int, Int, Ref, Ref, Ref, Int }
        RIRRRI { Ref, Int, Ref, Ref, Ref, Int }
        IRRRRI { Int, Ref, Ref, Ref, Ref, Int }
        RRRRRI { Ref, Ref, Ref, Ref, Ref, Int }
        IIIIIR { Int, Int, Int, Int, Int, Ref }
        RIIIIR { Ref, Int, Int, Int, Int, Ref }
        IRIIIR { Int, Ref, Int, Int, Int, Ref }
        RRIIIR { Ref, Ref, Int, Int, Int, Ref }
        IIRIIR { Int, Int, Ref, Int, Int, Ref }
        RIRIIR { Ref, Int, Ref, Int, Int, Ref }
        IRRIIR { Int, Ref, Ref, Int, Int, Ref }
        RRRIIR { Ref, Ref, Ref, Int, Int, Ref }
        IIIRIR { Int, Int, Int, Ref, Int, Ref }
        RIIRIR { Ref, Int, Int, Ref, Int, Ref }
        IRIRIR { Int, Ref, Int, Ref, Int, Ref }
        RRIRIR { Ref, Ref, Int, Ref, Int, Ref }
        IIRRIR { Int, Int, Ref, Ref, Int, Ref }
        RIRRIR { Ref, Int, Ref, Ref, Int, Ref }
        IRRRIR { Int, Ref, Ref, Ref, Int, Ref }
        RRRRIR { Ref, Ref, Ref, Ref, Int, Ref }
        IIIIRR { Int, Int, Int, Int, Ref, Ref }
        RIIIRR { Ref, Int, Int, Int, Ref, Ref }
        IRIIRR { Int, Ref, Int, Int, Ref, Ref }
        RRIIRR { Ref, Ref, Int, Int, Ref, Ref }
        IIRIRR { Int, Int, Ref, Int, Ref, Ref }
        RIRIRR { Ref, Int, Ref, Int, Ref, Ref }
        IRRIRR { Int, Ref, Ref, Int, Ref, Ref }
        RRRIRR { Ref, Ref, Ref, Int, Ref, Ref }
        IIIRRR { Int, Int, Int, Ref, Ref, Ref }
        RIIRRR { Ref, Int, Int, Ref, Ref, Ref }
        IRIRRR { Int, Ref, Int, Ref, Ref, Ref }
        RRIRRR { Ref, Ref, Int, Ref, Ref, Ref }
        IIRRRR { Int, Int, Ref, Ref, Ref, Ref }
        RIRRRR { Ref, Int, Ref, Ref, Ref, Ref }
        IRRRRR { Int, Ref, Ref, Ref, Ref, Ref }
        RRRRRR { Ref, Ref, Ref, Ref, Ref, Ref }
        RIIIIII { Ref, Int, Int, Int, Int, Int, Int }
        IRIIIII { Int, Ref, Int, Int, Int, Int, Int }
        RRIIIII { Ref, Ref, Int, Int, Int, Int, Int }
        IIRIIII { Int, Int, Ref, Int, Int, Int, Int }
        RIRIIII { Ref, Int, Ref, Int, Int, Int, Int }
        IRRIIII { Int, Ref, Ref, Int, Int, Int, Int }
        RRRIIII { Ref, Ref, Ref, Int, Int, Int, Int }
        IIIRIII { Int, Int, Int, Ref, Int, Int, Int }
        RIIRIII { Ref, Int, Int, Ref, Int, Int, Int }
        IRIRIII { Int, Ref, Int, Ref, Int, Int, Int }
        RRIRIII { Ref, Ref, Int, Ref, Int, Int, Int }
        IIRRIII { Int, Int, Ref, Ref, Int, Int, Int }
        RIRRIII { Ref, Int, Ref, Ref, Int, Int, Int }
        IRRRIII { Int, Ref, Ref, Ref, Int, Int, Int }
        RRRRIII { Ref, Ref, Ref, Ref, Int, Int, Int }
        IIIIRII { Int, Int, Int, Int, Ref, Int, Int }
        RIIIRII { Ref, Int, Int, Int, Ref, Int, Int }
        IRIIRII { Int, Ref, Int, Int, Ref, Int, Int }
        RRIIRII { Ref, Ref, Int, Int, Ref, Int, Int }
        IIRIRII { Int, Int, Ref, Int, Ref, Int, Int }
        RIRIRII { Ref, Int, Ref, Int, Ref, Int, Int }
        IRRIRII { Int, Ref, Ref, Int, Ref, Int, Int }
        RRRIRII { Ref, Ref, Ref, Int, Ref, Int, Int }
        IIIRRII { Int, Int, Int, Ref, Ref, Int, Int }
        RIIRRII { Ref, Int, Int, Ref, Ref, Int, Int }
        IRIRRII { Int, Ref, Int, Ref, Ref, Int, Int }
        RRIRRII { Ref, Ref, Int, Ref, Ref, Int, Int }
        IIRRRII { Int, Int, Ref, Ref, Ref, Int, Int }
        RIRRRII { Ref, Int, Ref, Ref, Ref, Int, Int }
        IRRRRII { Int, Ref, Ref, Ref, Ref, Int, Int }
        RRRRRII { Ref, Ref, Ref, Ref, Ref, Int, Int }
        IIIIIRI { Int, Int, Int, Int, Int, Ref, Int }
        RIIIIRI { Ref, Int, Int, Int, Int, Ref, Int }
        IRIIIRI { Int, Ref, Int, Int, Int, Ref, Int }
        RRIIIRI { Ref, Ref, Int, Int, Int, Ref, Int }
        IIRIIRI { Int, Int, Ref, Int, Int, Ref, Int }
        RIRIIRI { Ref, Int, Ref, Int, Int, Ref, Int }
        IRRIIRI { Int, Ref, Ref, Int, Int, Ref, Int }
        RRRIIRI { Ref, Ref, Ref, Int, Int, Ref, Int }
        IIIRIRI { Int, Int, Int, Ref, Int, Ref, Int }
        RIIRIRI { Ref, Int, Int, Ref, Int, Ref, Int }
        IRIRIRI { Int, Ref, Int, Ref, Int, Ref, Int }
        RRIRIRI { Ref, Ref, Int, Ref, Int, Ref, Int }
        IIRRIRI { Int, Int, Ref, Ref, Int, Ref, Int }
        RIRRIRI { Ref, Int, Ref, Ref, Int, Ref, Int }
        IRRRIRI { Int, Ref, Ref, Ref, Int, Ref, Int }
        RRRRIRI { Ref, Ref, Ref, Ref, Int, Ref, Int }
        IIIIRRI { Int, Int, Int, Int, Ref, Ref, Int }
        RIIIRRI { Ref, Int, Int, Int, Ref, Ref, Int }
        IRIIRRI { Int, Ref, Int, Int, Ref, Ref, Int }
        RRIIRRI { Ref, Ref, Int, Int, Ref, Ref, Int }
        IIRIRRI { Int, Int, Ref, Int, Ref, Ref, Int }
        RIRIRRI { Ref, Int, Ref, Int, Ref, Ref, Int }
        IRRIRRI { Int, Ref, Ref, Int, Ref, Ref, Int }
        RRRIRRI { Ref, Ref, Ref, Int, Ref, Ref, Int }
        IIIRRRI { Int, Int, Int, Ref, Ref, Ref, Int }
        RIIRRRI { Ref, Int, Int, Ref, Ref, Ref, Int }
        IRIRRRI { Int, Ref, Int, Ref, Ref, Ref, Int }
        RRIRRRI { Ref, Ref, Int, Ref, Ref, Ref, Int }
        IIRRRRI { Int, Int, Ref, Ref, Ref, Ref, Int }
        RIRRRRI { Ref, Int, Ref, Ref, Ref, Ref, Int }
        IRRRRRI { Int, Ref, Ref, Ref, Ref, Ref, Int }
        RRRRRRI { Ref, Ref, Ref, Ref, Ref, Ref, Int }
        IIIIIIR { Int, Int, Int, Int, Int, Int, Ref }
        RIIIIIR { Ref, Int, Int, Int, Int, Int, Ref }
        IRIIIIR { Int, Ref, Int, Int, Int, Int, Ref }
        RRIIIIR { Ref, Ref, Int, Int, Int, Int, Ref }
        IIRIIIR { Int, Int, Ref, Int, Int, Int, Ref }
        RIRIIIR { Ref, Int, Ref, Int, Int, Int, Ref }
        IRRIIIR { Int, Ref, Ref, Int, Int, Int, Ref }
        RRRIIIR { Ref, Ref, Ref, Int, Int, Int, Ref }
        IIIRIIR { Int, Int, Int, Ref, Int, Int, Ref }
        RIIRIIR { Ref, Int, Int, Ref, Int, Int, Ref }
        IRIRIIR { Int, Ref, Int, Ref, Int, Int, Ref }
        RRIRIIR { Ref, Ref, Int, Ref, Int, Int, Ref }
        IIRRIIR { Int, Int, Ref, Ref, Int, Int, Ref }
        RIRRIIR { Ref, Int, Ref, Ref, Int, Int, Ref }
        IRRRIIR { Int, Ref, Ref, Ref, Int, Int, Ref }
        RRRRIIR { Ref, Ref, Ref, Ref, Int, Int, Ref }
        IIIIRIR { Int, Int, Int, Int, Ref, Int, Ref }
        RIIIRIR { Ref, Int, Int, Int, Ref, Int, Ref }
        IRIIRIR { Int, Ref, Int, Int, Ref, Int, Ref }
        RRIIRIR { Ref, Ref, Int, Int, Ref, Int, Ref }
        IIRIRIR { Int, Int, Ref, Int, Ref, Int, Ref }
        RIRIRIR { Ref, Int, Ref, Int, Ref, Int, Ref }
        IRRIRIR { Int, Ref, Ref, Int, Ref, Int, Ref }
        RRRIRIR { Ref, Ref, Ref, Int, Ref, Int, Ref }
        IIIRRIR { Int, Int, Int, Ref, Ref, Int, Ref }
        RIIRRIR { Ref, Int, Int, Ref, Ref, Int, Ref }
        IRIRRIR { Int, Ref, Int, Ref, Ref, Int, Ref }
        RRIRRIR { Ref, Ref, Int, Ref, Ref, Int, Ref }
        IIRRRIR { Int, Int, Ref, Ref, Ref, Int, Ref }
        RIRRRIR { Ref, Int, Ref, Ref, Ref, Int, Ref }
        IRRRRIR { Int, Ref, Ref, Ref, Ref, Int, Ref }
        RRRRRIR { Ref, Ref, Ref, Ref, Ref, Int, Ref }
        IIIIIRR { Int, Int, Int, Int, Int, Ref, Ref }
        RIIIIRR { Ref, Int, Int, Int, Int, Ref, Ref }
        IRIIIRR { Int, Ref, Int, Int, Int, Ref, Ref }
        RRIIIRR { Ref, Ref, Int, Int, Int, Ref, Ref }
        IIRIIRR { Int, Int, Ref, Int, Int, Ref, Ref }
        RIRIIRR { Ref, Int, Ref, Int, Int, Ref, Ref }
        IRRIIRR { Int, Ref, Ref, Int, Int, Ref, Ref }
        RRRIIRR { Ref, Ref, Ref, Int, Int, Ref, Ref }
        IIIRIRR { Int, Int, Int, Ref, Int, Ref, Ref }
        RIIRIRR { Ref, Int, Int, Ref, Int, Ref, Ref }
        IRIRIRR { Int, Ref, Int, Ref, Int, Ref, Ref }
        RRIRIRR { Ref, Ref, Int, Ref, Int, Ref, Ref }
        IIRRIRR { Int, Int, Ref, Ref, Int, Ref, Ref }
        RIRRIRR { Ref, Int, Ref, Ref, Int, Ref, Ref }
        IRRRIRR { Int, Ref, Ref, Ref, Int, Ref, Ref }
        RRRRIRR { Ref, Ref, Ref, Ref, Int, Ref, Ref }
        IIIIRRR { Int, Int, Int, Int, Ref, Ref, Ref }
        RIIIRRR { Ref, Int, Int, Int, Ref, Ref, Ref }
        IRIIRRR { Int, Ref, Int, Int, Ref, Ref, Ref }
        RRIIRRR { Ref, Ref, Int, Int, Ref, Ref, Ref }
        IIRIRRR { Int, Int, Ref, Int, Ref, Ref, Ref }
        RIRIRRR { Ref, Int, Ref, Int, Ref, Ref, Ref }
        IRRIRRR { Int, Ref, Ref, Int, Ref, Ref, Ref }
        RRRIRRR { Ref, Ref, Ref, Int, Ref, Ref, Ref }
        IIIRRRR { Int, Int, Int, Ref, Ref, Ref, Ref }
        RIIRRRR { Ref, Int, Int, Ref, Ref, Ref, Ref }
        IRIRRRR { Int, Ref, Int, Ref, Ref, Ref, Ref }
        RRIRRRR { Ref, Ref, Int, Ref, Ref, Ref, Ref }
        IIRRRRR { Int, Int, Ref, Ref, Ref, Ref, Ref }
        RIRRRRR { Ref, Int, Ref, Ref, Ref, Ref, Ref }
        IRRRRRR { Int, Ref, Ref, Ref, Ref, Ref, Ref }
        RRRRRRR { Ref, Ref, Ref, Ref, Ref, Ref, Ref }
        RRRRRRRR { Ref, Ref, Ref, Ref, Ref, Ref, Ref, Ref }
        RRRRRRRRR { Ref, Ref, Ref, Ref, Ref, Ref, Ref, Ref, Ref }
        RRRRRRRRRR { Ref, Ref, Ref, Ref, Ref, Ref, Ref, Ref, Ref, Ref }
        RRRRRRRRRRR { Ref, Ref, Ref, Ref, Ref, Ref, Ref, Ref, Ref, Ref, Ref }
        RRRRRRRRRRRR { Ref, Ref, Ref, Ref, Ref, Ref, Ref, Ref, Ref, Ref, Ref, Ref }
        RRRRRRRRRRRRR { Ref, Ref, Ref, Ref, Ref, Ref, Ref, Ref, Ref, Ref, Ref, Ref, Ref }
        RRRRRRRRRRRRRR { Ref, Ref, Ref, Ref, Ref, Ref, Ref, Ref, Ref, Ref, Ref, Ref, Ref, Ref }
        RRRRRRRRRRRRRRR { Ref, Ref, Ref, Ref, Ref, Ref, Ref, Ref, Ref, Ref, Ref, Ref, Ref, Ref, Ref }
        RRRRRRRRRRRRRRRR { Ref, Ref, Ref, Ref, Ref, Ref, Ref, Ref, Ref, Ref, Ref, Ref, Ref, Ref, Ref, Ref }
        I6 { Int, Int, Int, Int, Int, Int }
        I7 { Int, Int, Int, Int, Int, Int, Int }
        I8 { Int, Int, Int, Int, Int, Int, Int, Int }
        I9 { Int, Int, Int, Int, Int, Int, Int, Int, Int }
        I10 { Int, Int, Int, Int, Int, Int, Int, Int, Int, Int }
        I11 { Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int }
        I12 { Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int }
        I13 { Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int }
        I14 { Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int }
        I15 { Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int }
        I16 { Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int }
        }
    };
}

macro_rules! define_call_sig_stubs {
    ($($name:ident { $($class:ident),* })*) => {
        fn lookup_stub_i(classes: &[ArgClass]) -> unsafe fn(usize, &[i64]) -> i64 {
            match classes {
                $(
                    [$(ArgClass::$class),*] => {
                        unsafe fn stub(func: usize, args: &[i64]) -> i64 {
                            unsafe { invoke_stub!(func, args, i64 $(, $class)*) }
                        }
                        stub
                    }
                )*
                classes => unsupported_call_sig(classes),
            }
        }

        fn lookup_stub_f(classes: &[ArgClass]) -> unsafe fn(usize, &[i64]) -> f64 {
            match classes {
                $(
                    [$(ArgClass::$class),*] => {
                        unsafe fn stub(func: usize, args: &[i64]) -> f64 {
                            unsafe { invoke_stub!(func, args, f64 $(, $class)*) }
                        }
                        stub
                    }
                )*
                classes => unsupported_call_sig(classes),
            }
        }

        fn lookup_stub_v(classes: &[ArgClass]) -> unsafe fn(usize, &[i64]) {
            match classes {
                $(
                    [$(ArgClass::$class),*] => {
                        unsafe fn stub(func: usize, args: &[i64]) {
                            unsafe { invoke_stub!(func, args, () $(, $class)*) }
                        }
                        stub
                    }
                )*
                classes => unsupported_call_sig(classes),
            }
        }
    };
}

fn unsupported_call_sig(classes: &[ArgClass]) -> ! {
    // `descr.py CallDescr.create_call_stub` generates a
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

call_sig_table!(define_call_sig_stubs);

/// llmodel.py bh_call_i call_stub_i: ABI-correct dispatch in calldescr declaration
/// order.
///
/// Safety: `func` must be a valid function pointer matching `classes`, i.e. an
/// `extern "C" fn(...) -> i64` whose parameter list is the same ordered
/// Int/Float sequence and whose float slots are carried in `args` as
/// `f64::to_bits`.
///
/// The `-> i64` is a requirement on the target, not a convenience of the
/// transmute: this reads the whole return register, and a callee whose result
/// is narrower leaves the bits above it undefined. `#[jit_interp]` meets it by
/// registering a widening shim in place of such a target — see
/// `majit_ir::CallResultWord`.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
/// wasm32 `call_indirect` type-checks the callee. The descr class list is not
/// that type: published targets are widening `i64` shims, raw pointers are
/// `i32`. The host reads the table signature. Native keeps the stub.
/// The static stub table has every sequence through arity 5, all-`Int` and
/// all-`Ref` through [`MAX_HOST_CALL_ARITY`], and mixed `Int`/`Ref` through
/// arity 7. A float appears only through arity 5.
pub fn call_stub_arm_exists(classes: &[ArgClass]) -> bool {
    let n = classes.len();
    if n > MAX_HOST_CALL_ARITY {
        return false;
    }
    let has_float = classes.contains(&ArgClass::Float);
    let has_ref = classes.contains(&ArgClass::Ref);
    let has_int = classes.contains(&ArgClass::Int);
    if has_float {
        return n <= 5;
    }
    if has_ref && has_int {
        return n <= 7;
    }
    true
}

fn wasm_residual_host_call(
    func: usize,
    args: &[i64],
    classes: &[ArgClass],
    result: char,
) -> Option<i64> {
    residual_host_call().and_then(|hook| hook(func, args, classes, result))
}

pub unsafe fn bh_call_i_dispatch(func: usize, classes: &[ArgClass], args: &[i64]) -> i64 {
    assert_eq!(
        classes.len(),
        args.len(),
        "bh_call dispatch: class sequence and positional arg list length differ"
    );
    if let Some(result) = wasm_residual_host_call(func, args, classes, 'i') {
        return result;
    }
    unsafe { (lookup_stub_i(classes))(func, args) }
}

/// `llmodel.py bh_call_r`. The staged wrapper returns the pointer as an `i64`
/// word. A table type that is a real `i32` is host-reflected.
///
/// # Safety
/// `func` must match `classes`, and its result must be a GCREF.
pub unsafe fn bh_call_r_dispatch(func: usize, classes: &[ArgClass], args: &[i64]) -> i64 {
    assert_eq!(
        classes.len(),
        args.len(),
        "bh_call dispatch: class sequence and positional arg list length differ"
    );
    #[cfg(target_arch = "wasm32")]
    {
        if let Some(result) = wasm_residual_host_call(func, args, classes, 'r') {
            return result;
        }
        unsafe { (lookup_stub_i(classes))(func, args) }
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        unsafe { bh_call_i_dispatch(func, classes, args) }
    }
}

/// llmodel.py bh_call_v: void-typed parallel of `bh_call_i_dispatch`.
///
/// Safety: `func` must be a valid function pointer matching `classes`.
/// `descr.py create_call_stub` builds a real void-returning stub for
/// `RESULT == lltype.Void`; calling such a function through an `i64`-returning
/// transmute reads garbage from rax/x0, so the canonical
/// `BC_RESIDUAL_CALL_*_V` blackhole/trace path must use this dispatcher.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn bh_call_v_dispatch(func: usize, classes: &[ArgClass], args: &[i64]) {
    assert_eq!(
        classes.len(),
        args.len(),
        "bh_call dispatch: class sequence and positional arg list length differ"
    );
    if wasm_residual_host_call(func, args, classes, 'v').is_some() {
        return;
    }
    unsafe { (lookup_stub_v(classes))(func, args) }
}

/// llmodel.py bh_call_f: f64-typed parallel of `bh_call_i_dispatch`.
///
/// Safety: `func` must be a valid function pointer matching `classes`.
/// `descr.py create_call_stub` generates a real f64-returning stub for
/// `RESULT == lltype.Float`; the C ABI returns f64 in xmm0 / d0 rather than
/// rax / x0, so an `i64`-typed transmute would read uninitialized integer-bank
/// state.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn bh_call_f_dispatch(func: usize, classes: &[ArgClass], args: &[i64]) -> f64 {
    assert_eq!(
        classes.len(),
        args.len(),
        "bh_call dispatch: class sequence and positional arg list length differ"
    );
    if let Some(bits) = wasm_residual_host_call(func, args, classes, 'f') {
        return f64::from_bits(bits as u64);
    }
    unsafe { (lookup_stub_f(classes))(func, args) }
}

/// The ordered class sequence and matching positional argument list that
/// [`collect_call_args`] hands to `bh_call_*_dispatch`.
///
/// Fixed-size so a residual call allocates nothing. `MAX_HOST_CALL_ARITY` is
/// the same bound the dispatch table's all-`Int` arms stop at, so a signature
/// that does not fit here has no arm either.
pub struct CallArgs {
    classes: [ArgClass; MAX_HOST_CALL_ARITY],
    args: [i64; MAX_HOST_CALL_ARITY],
    len: usize,
}

impl CallArgs {
    fn with_arity(arity: usize) -> Self {
        assert!(
            arity <= MAX_HOST_CALL_ARITY,
            "bh_call dispatch: {arity} arguments exceeds MAX_HOST_CALL_ARITY \
             ({MAX_HOST_CALL_ARITY}); the dispatch table has no arm this wide"
        );
        Self {
            classes: [ArgClass::Int; MAX_HOST_CALL_ARITY],
            args: [0; MAX_HOST_CALL_ARITY],
            len: 0,
        }
    }

    fn push(&mut self, class: ArgClass, arg: i64) {
        self.classes[self.len] = class;
        self.args[self.len] = arg;
        self.len += 1;
    }

    pub fn classes(&self) -> &[ArgClass] {
        &self.classes[..self.len]
    }

    pub fn args(&self) -> &[i64] {
        &self.args[..self.len]
    }
}

/// `descr.py` `assert self.result_type in return_type` — the half of
/// `verify_types` that [`collect_call_args`] does not cover.
///
/// `accepted` is the caller's `return_type`, the literal each `bh_call_*`
/// passes in `llmodel.py/822/828/835`: `"iS"` (`history.INT + 'S'`),
/// `"r"`, `"fL"` (`history.FLOAT + 'L'`) and `"v"`.
///
/// This is not redundant with the argument-class check. Nothing on the call
/// path reads `result_type` to pick a dispatcher: the blackhole reaches one of
/// the four entry points because the codewriter emitted
/// `residual_call_*_i/_r/_f/_v`, so the two records can disagree. They
/// disagree silently — the dispatcher would read the callee's result out of
/// the register file the *opcode* named while the callee returned it in the
/// one `result_type` names, e.g. an f64 result taken from `rax`. Comparing
/// them here is what makes that a failure rather than a garbage value.
///
/// `debug_assert` mirrors upstream's `if not we_are_translated():` guard at
/// each call site — a translated JIT does not run this. `majit-backend` is not
/// one of the LLBC-extracted crates, so a release build really does drop it.
///
/// A NULL function pointer is rejected by every `bh_call_*` entry before this
/// check.  `llmodel.py::bh_call_i/r/f/v` has no fake-success path for a missing
/// callee; callers that carry `JitCode.fnaddr == None` must abort before backend
/// dispatch.
pub fn verify_result_type(result_type: char, accepted: &str) {
    debug_assert!(
        accepted.contains(result_type),
        "BhCallDescr.verify_types: result_type {result_type:?} is not one of {accepted:?}; \
         the emitted residual_call opcode and the calldescr disagree about the return register"
    );
}

/// `descr.py CallDescr.create_call_stub`: walk `arg_classes` once, pick the
/// monomorphic ABI stub, and record the bank mapping. `verify_result_type`
/// and the `verify_types` count of each class run here, not per call.
pub fn create_call_stub(arg_classes: &str, result_type: char) -> BhCallStub {
    let accepted = match result_type {
        'i' | 'S' => "iS",
        'r' => "r",
        'f' | 'L' => "fL",
        'v' => "v",
        _ => "",
    };
    verify_result_type(result_type, accepted);

    let mut slots = [0u8; MAX_HOST_CALL_ARITY];
    let mut classes_buf = [ArgClass::Int; MAX_HOST_CALL_ARITY];
    let mut arity = 0u8;
    let mut expect_i = 0u8;
    let mut expect_r = 0u8;
    let mut expect_f = 0u8;
    let mut ii = 0u8;
    let mut ri = 0u8;
    let mut fi = 0u8;

    for c in arg_classes.chars() {
        if arity as usize >= MAX_HOST_CALL_ARITY {
            panic!(
                "bh_call dispatch: {} arguments exceeds MAX_HOST_CALL_ARITY \
                 ({MAX_HOST_CALL_ARITY}); the dispatch table has no arm this wide",
                arity as usize + 1
            );
        }
        let (bank, class) = match c {
            'i' => (BhCallStub::BANK_I, ArgClass::Int),
            'r' => (BhCallStub::BANK_R, ArgClass::Int),
            'f' => (BhCallStub::BANK_F, ArgClass::Float),
            'L' => (BhCallStub::BANK_F, ArgClass::Int),
            'S' => {
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
        };
        let idx = match bank {
            BhCallStub::BANK_I => {
                let i = ii;
                ii += 1;
                expect_i += 1;
                i
            }
            BhCallStub::BANK_R => {
                let i = ri;
                ri += 1;
                expect_r += 1;
                i
            }
            _ => {
                let i = fi;
                fi += 1;
                expect_f += 1;
                i
            }
        };
        slots[arity as usize] = (bank << 6) | idx;
        classes_buf[arity as usize] = class;
        arity += 1;
    }

    let classes = &classes_buf[..arity as usize];
    let call_i = {
        #[cfg(target_arch = "wasm32")]
        {
            let _ = result_type;
            lookup_stub_i(classes)
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = result_type;
            lookup_stub_i(classes)
        }
    };
    BhCallStub::new(
        slots,
        arity,
        expect_i,
        expect_r,
        expect_f,
        call_i,
        lookup_stub_f(classes),
        lookup_stub_v(classes),
    )
}

fn call_stub_for(calldescr: &BhCallDescr) -> &BhCallStub {
    calldescr
        .call_stub
        .get_or_init(|| create_call_stub(&calldescr.arg_classes, calldescr.result_type))
}

/// `llmodel.py AbstractLLCPU.bh_call_i`: read the per-descr stub and call.
///
/// # Safety
/// `func` must match the ABI [`create_call_stub`] derives from `arg_classes`.
pub unsafe fn bh_call_i_with_descr(
    func: usize,
    args_i: Option<&[i64]>,
    args_r: Option<&[i64]>,
    args_f: Option<&[i64]>,
    calldescr: &BhCallDescr,
) -> i64 {
    if residual_host_call().is_some() {
        let collected = collect_call_args(&calldescr.arg_classes, args_i, args_r, args_f);
        let result_type = if calldescr.void_word_abi {
            'i'
        } else {
            calldescr.result_type
        };
        if let Some(result) =
            wasm_residual_host_call(func, collected.args(), collected.classes(), result_type)
        {
            return result;
        }
    }
    unsafe { call_stub_for(calldescr).call_i(func, args_i, args_r, args_f) }
}

/// `llmodel.py AbstractLLCPU.bh_call_f` parallel of [`bh_call_i_with_descr`].
///
/// # Safety
/// See [`bh_call_i_with_descr`].
pub unsafe fn bh_call_f_with_descr(
    func: usize,
    args_i: Option<&[i64]>,
    args_r: Option<&[i64]>,
    args_f: Option<&[i64]>,
    calldescr: &BhCallDescr,
) -> f64 {
    if residual_host_call().is_some() {
        let collected = collect_call_args(&calldescr.arg_classes, args_i, args_r, args_f);
        if let Some(bits) = wasm_residual_host_call(
            func,
            collected.args(),
            collected.classes(),
            calldescr.result_type,
        ) {
            return f64::from_bits(bits as u64);
        }
    }
    unsafe { call_stub_for(calldescr).call_f(func, args_i, args_r, args_f) }
}

/// `llmodel.py AbstractLLCPU.bh_call_v` parallel of [`bh_call_i_with_descr`].
///
/// # Safety
/// See [`bh_call_i_with_descr`].
pub unsafe fn bh_call_v_with_descr(
    func: usize,
    args_i: Option<&[i64]>,
    args_r: Option<&[i64]>,
    args_f: Option<&[i64]>,
    calldescr: &BhCallDescr,
) {
    if residual_host_call().is_some() {
        let collected = collect_call_args(&calldescr.arg_classes, args_i, args_r, args_f);
        let result_type = if calldescr.void_word_abi { 'i' } else { 'v' };
        if wasm_residual_host_call(func, collected.args(), collected.classes(), result_type)
            .is_some()
        {
            return;
        }
        if calldescr.void_word_abi {
            let _ = unsafe { call_stub_for(calldescr).call_i(func, args_i, args_r, args_f) };
            return;
        }
    }
    unsafe { call_stub_for(calldescr).call_v(func, args_i, args_r, args_f) }
}

/// Build the C-ABI class sequence and positional argument list from
/// `args_i` / `args_r` / `args_f`, following `calldescr.arg_classes` order.
///
/// `arg_classes` is the per-argument class string from
/// `majit_jitcode::jitcode::BhCallDescr`. RPython
/// `rpython/jit/backend/llsupport/descr.py create_call_stub`'s
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
/// Mirrors `rpython/jit/backend/llsupport/descr.py verify_types`:
/// the per-class counts in `arg_classes` must match the corresponding list
/// length, and any unknown class is a codegen bug.
///
/// Returns a stack buffer rather than two `Vec`s: this runs on every residual
/// call the blackhole makes, and upstream's generated stub reaches the callee
/// with no intermediate collection at all.
pub fn collect_call_args(
    arg_classes: &str,
    args_i: Option<&[i64]>,
    args_r: Option<&[i64]>,
    args_f: Option<&[i64]>,
) -> CallArgs {
    // descr.py verify_types parity: assert per-class counts.
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

    let mut out = CallArgs::with_arity(count_i + count_r + count_f);
    let mut ii = 0usize;
    let mut ri = 0usize;
    let mut fi = 0usize;
    for c in arg_classes.chars() {
        match c {
            'i' => {
                out.push(
                    ArgClass::Int,
                    args_i.expect("BhCallDescr.collect_call_args: args_i missing")[ii],
                );
                ii += 1;
            }
            'r' => {
                out.push(
                    ArgClass::Int,
                    args_r.expect("BhCallDescr.collect_call_args: args_r missing")[ri],
                );
                ri += 1;
            }
            'f' => {
                out.push(
                    ArgClass::Float,
                    args_f.expect("BhCallDescr.collect_call_args: args_f missing")[fi],
                );
                fi += 1;
            }
            'L' => {
                // descr.py process('L'): storage bank = `args_f`
                // (PyPy rewrites `c = 'f'` for the lookup); FUNC parameter
                // type = `lltype.SignedLongLong` -> C `long long` ->
                // 8-byte int dispatched in an integer register.
                out.push(
                    ArgClass::Int,
                    args_f.expect("BhCallDescr.collect_call_args: args_f missing")[fi],
                );
                fi += 1;
            }
            'S' => {
                // descr.py process('S'): storage bank = `args_i`
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
    out
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
/// from `arg_classes` declaration order, matching `descr.py create_call_stub`.
/// On wasm32 a `r` parameter is `i32` and a `r` result is zero-extended back
/// to the blackhole word; `i` stays `i64`.
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
    let collected = collect_call_args(arg_classes, args_i, args_r, args_f);
    unsafe { bh_call_i_dispatch(func, collected.classes(), collected.args()) }
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
    let collected = collect_call_args(arg_classes, args_i, args_r, args_f);
    unsafe { bh_call_f_dispatch(func, collected.classes(), collected.args()) }
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
    let collected = collect_call_args(arg_classes, args_i, args_r, args_f);
    unsafe { bh_call_v_dispatch(func, collected.classes(), collected.args()) }
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
/// `None` means the callee's wasm type matches the stub, so the caller uses it.
/// `Some` is the host result when the types differ.
pub type ResidualHostCallFn =
    fn(func_ptr: usize, args: &[i64], classes: &[ArgClass], result: char) -> Option<i64>;

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
    /// `majit_jitcode::codewriter::jitcode::MAX_FLOAT_CARRYING_CALL_ARITY`
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

    /// `collect_call_args` fills a fixed buffer, so a signature wider than the
    /// dispatch table's widest arm is refused while collecting rather than
    /// overrunning it.
    #[test]
    #[should_panic(expected = "exceeds MAX_HOST_CALL_ARITY")]
    fn collect_call_args_refuses_more_arguments_than_the_table_covers() {
        let too_many = MAX_HOST_CALL_ARITY + 1;
        let args_i = vec![0_i64; too_many];
        let _ = collect_call_args(&"i".repeat(too_many), Some(&args_i), None, None);
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

    /// The four `return_type` literals `llmodel.py bh_call_i/822/828/835` pass, each
    /// against every result class `type_to_argclass` can produce. `'L'` and
    /// `'S'` ride along with the float and int sets the way upstream spells
    /// them (`history.FLOAT + 'L'`, `history.INT + 'S'`).
    #[test]
    fn verify_result_type_accepts_exactly_its_caller_s_return_classes() {
        for (accepted, ok) in [("iS", "iS"), ("r", "r"), ("fL", "fL"), ("v", "v")] {
            for c in ok.chars() {
                verify_result_type(c, accepted);
            }
        }
    }

    /// The mis-route this exists to catch: a float-returning callee reached
    /// through the integer entry point would take its result from `rax`.
    ///
    /// Only meaningful where the assertion is compiled in — `verify_result_type`
    /// is a `debug_assert`, so a release test binary would not panic.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "is not one of \"iS\"")]
    fn verify_result_type_rejects_a_float_result_on_the_int_entry_point() {
        verify_result_type('f', "iS");
    }

    /// `BhCallDescr::default()` leaves `result_type` at `char::default()`.
    /// `jitdriver.rs` reaches that through `unwrap_or_default()`, so the
    /// sentinel has to be rejected rather than silently matching some class.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "is not one of \"r\"")]
    fn verify_result_type_rejects_the_default_descr_s_null_result_type() {
        verify_result_type('\0', "r");
    }

    #[test]
    fn mixed_int_ref_past_arity_7_has_no_stub_arm() {
        assert!(!call_stub_arm_exists(&[
            ArgClass::Int,
            ArgClass::Ref,
            ArgClass::Int,
            ArgClass::Ref,
            ArgClass::Int,
            ArgClass::Ref,
            ArgClass::Int,
            ArgClass::Ref,
        ]));
        assert!(call_stub_arm_exists(&[ArgClass::Int; 8]));
        assert!(call_stub_arm_exists(&[ArgClass::Ref; 8]));
        let collected =
            collect_call_args("iriririr", Some(&[1, 2, 3, 4]), Some(&[5, 6, 7, 8]), None);
        assert!(collected.classes().iter().all(|c| *c == ArgClass::Int));
        assert!(call_stub_arm_exists(collected.classes()));
        assert!(!call_stub_arm_exists(&[ArgClass::Float; 6]));
    }

    /// Mixed `i`/`r` past arity 7 has no stub-table arm. The host hook must run
    /// before that lookup.
    #[test]
    fn host_hook_dispatches_a_wide_mixed_signature_without_a_stub_arm() {
        struct Clear;
        impl Drop for Clear {
            fn drop(&mut self) {
                set_residual_host_call(None);
            }
        }
        let _clear = Clear;
        set_residual_host_call(Some(|func, args, _classes, result| {
            assert_eq!(func, 7);
            assert_eq!(args, &[1, 5, 2, 6, 3, 7, 4, 8]);
            assert_eq!(result, 'i');
            Some(42)
        }));
        let descr = BhCallDescr::from_arg_classes(
            "iriririr".to_string(),
            'i',
            majit_ir::descr::EffectInfo::MOST_GENERAL,
        );
        let result = unsafe {
            bh_call_i_with_descr(7, Some(&[1, 2, 3, 4]), Some(&[5, 6, 7, 8]), None, &descr)
        };
        assert_eq!(result, 42);
    }

    /// `'L'` is an i64 return stored in the float bank. The host result is the
    /// raw bits; `bh_call_f_with_descr` reinterprets them.
    #[test]
    fn host_hook_longlong_result_is_stored_as_float_bits() {
        struct Clear;
        impl Drop for Clear {
            fn drop(&mut self) {
                set_residual_host_call(None);
            }
        }
        let _clear = Clear;
        set_residual_host_call(Some(|_func, _args, _classes, result| {
            assert_eq!(result, 'L');
            Some(f64::to_bits(1.0) as i64)
        }));
        let descr = BhCallDescr::from_arg_classes(
            "i".to_string(),
            'L',
            majit_ir::descr::EffectInfo::MOST_GENERAL,
        );
        let value = unsafe { bh_call_f_with_descr(3, Some(&[9]), None, None, &descr) };
        assert_eq!(value, 1.0);
    }

    /// `descr.py CallDescr.create_call_stub` + `llmodel.py AbstractLLCPU.bh_call_f`
    /// for the interleaved float/ref case of `test_call_stubs_2`.
    #[test]
    fn create_call_stub_f_interleaved_float_ref_preserves_declaration_order() {
        let b = [1_i64];
        let descr = BhCallDescr::from_arg_classes(
            "fr".to_string(),
            'f',
            majit_ir::descr::EffectInfo::MOST_GENERAL,
        );
        let result = unsafe {
            bh_call_f_with_descr(
                f2 as *const () as usize,
                None,
                Some(&[b.as_ptr() as i64]),
                Some(&[3.5_f64.to_bits() as i64]),
                &descr,
            )
        };
        assert_eq!(result, 4.5);
    }

    /// `shift` (`arg_classes = "rii"`) is the regex residual that the stub
    /// must serve without walking the class string on the second call.
    #[test]
    fn create_call_stub_i_rii_places_ref_then_two_ints() {
        extern "C" fn rii(a: i64, b: i64, c: i64) -> i64 {
            a + b * 10 + c * 100
        }
        let descr = BhCallDescr::from_arg_classes(
            "rii".to_string(),
            'i',
            majit_ir::descr::EffectInfo::MOST_GENERAL,
        );
        let result = unsafe {
            bh_call_i_with_descr(
                rii as *const () as usize,
                Some(&[2, 3]),
                Some(&[1]),
                None,
                &descr,
            )
        };
        assert_eq!(result, 321);
        let again = unsafe {
            bh_call_i_with_descr(
                rii as *const () as usize,
                Some(&[2, 3]),
                Some(&[1]),
                None,
                &descr,
            )
        };
        assert_eq!(again, 321);
        assert!(descr.call_stub.get().is_some());
    }
}
