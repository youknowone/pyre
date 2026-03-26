//! JIT-internal exception/control-flow types.
//!
//! Mirrors RPython's `jitexc.py`: exceptions raised and caught within
//! the JIT infrastructure, never exposed to user code.

use majit_ir::{GcRef, Value};

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
