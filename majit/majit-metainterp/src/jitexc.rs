//! JIT-internal exception/control-flow types.
//!
//! Mirrors RPython's `jitexc.py`: exceptions raised and caught within
//! the JIT infrastructure, never exposed to user code.

use majit_ir::{GcRef, Value};

/// jitexc.py:10 JitException — base class for all JIT control flow.
///
/// In RPython these are Python exceptions. In Rust we model them as an enum
/// returned via `Result::Err(JitException)` from blackhole execution.
#[derive(Debug, Clone, PartialEq)]
pub enum JitException {
    /// jitexc.py:17 DoneWithThisFrameVoid
    DoneWithThisFrameVoid,
    /// jitexc.py:21 DoneWithThisFrameInt
    DoneWithThisFrameInt(i64),
    /// jitexc.py:29 DoneWithThisFrameRef
    DoneWithThisFrameRef(GcRef),
    /// jitexc.py:37 DoneWithThisFrameFloat
    DoneWithThisFrameFloat(f64),
    /// jitexc.py:45 ExitFrameWithExceptionRef
    ExitFrameWithExceptionRef(GcRef),
    /// jitexc.py:53 ContinueRunningNormally
    ContinueRunningNormally {
        green_int: Vec<i64>,
        green_ref: Vec<i64>,
        green_float: Vec<i64>,
        red_int: Vec<i64>,
        red_ref: Vec<i64>,
        red_float: Vec<i64>,
    },
}

/// Result of a completed trace execution.
///
/// Mirrors DoneWithThisFrame{Void,Int,Ref,Float} in jitexc.py.
#[derive(Debug, Clone, PartialEq)]
pub enum DoneWithThisFrame {
    Void,
    Int(i64),
    Ref(GcRef),
    Float(f64),
}

impl DoneWithThisFrame {
    /// Create from a typed Value.
    pub fn from_value(v: Value) -> Self {
        match v {
            Value::Int(i) => DoneWithThisFrame::Int(i),
            Value::Ref(r) => DoneWithThisFrame::Ref(r),
            Value::Float(f) => DoneWithThisFrame::Float(f),
            Value::Void => DoneWithThisFrame::Void,
        }
    }
}

/// The trace exited with an exception.
///
/// Mirrors ExitFrameWithExceptionRef in jitexc.py.
#[derive(Debug, Clone, PartialEq)]
pub struct ExitFrameWithException {
    pub exc_value: GcRef,
}

/// Request to continue running in the interpreter (leave JIT).
///
/// Mirrors ContinueRunningNormally in jitexc.py.
#[derive(Debug, Clone)]
pub struct ContinueRunningNormally {
    pub green_values: Vec<i64>,
    pub red_values: Vec<i64>,
}

/// The loop is not vectorizable.
///
/// Mirrors NotAVectorizeableLoop in jitexc.py.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NotAVectorizeableLoop;

/// The loop is not profitable to vectorize.
///
/// Mirrors NotAProfitableLoop in jitexc.py.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NotAProfitableLoop;
