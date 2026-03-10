/// Guard failure handling for the Cranelift backend.
///
/// When a guard fails at runtime, execution exits the JIT-compiled loop
/// and the current register/variable state is packaged into a `FrameData`
/// (stored inside a `DeadFrame`) so the frontend can read values back.

use majit_ir::{FailDescr, GcRef, Type, Value};
use std::sync::Arc;

/// Concrete fail descriptor used by the Cranelift backend.
///
/// Carries the fail_index and the types of values that will be
/// saved in the DeadFrame on guard failure.
#[derive(Debug)]
pub struct CraneliftFailDescr {
    pub fail_index: u32,
    pub fail_arg_types: Vec<Type>,
}

impl majit_ir::Descr for CraneliftFailDescr {
    fn index(&self) -> u32 {
        self.fail_index
    }

    fn as_fail_descr(&self) -> Option<&dyn FailDescr> {
        Some(self)
    }
}

impl FailDescr for CraneliftFailDescr {
    fn fail_index(&self) -> u32 {
        self.fail_index
    }

    fn fail_arg_types(&self) -> &[Type] {
        &self.fail_arg_types
    }
}

/// The concrete data stored in a `DeadFrame` by the Cranelift backend.
///
/// Holds the saved values and a reference to the fail descriptor that
/// identifies which guard failed.
pub struct FrameData {
    /// The values of live variables at the point of guard failure (or finish).
    pub values: Vec<Value>,
    /// The fail descriptor identifying the guard that failed.
    pub fail_descr: Arc<CraneliftFailDescr>,
}

impl FrameData {
    pub fn get_int(&self, index: usize) -> i64 {
        match &self.values[index] {
            Value::Int(v) => *v,
            other => panic!("expected Int at index {index}, got {other:?}"),
        }
    }

    pub fn get_float(&self, index: usize) -> f64 {
        match &self.values[index] {
            Value::Float(v) => *v,
            other => panic!("expected Float at index {index}, got {other:?}"),
        }
    }

    pub fn get_ref(&self, index: usize) -> GcRef {
        match &self.values[index] {
            Value::Ref(v) => *v,
            other => panic!("expected Ref at index {index}, got {other:?}"),
        }
    }
}
