/// Guard failure handling for the Cranelift backend.
///
/// When a guard fails at runtime, execution exits the JIT-compiled loop
/// and the current register/variable state is packaged into a `FrameData`
/// (stored inside a `DeadFrame`) so the frontend can read values back.
///
/// Bridge support: when a guard fails frequently, a bridge trace can be
/// compiled and attached to the fail descriptor. On subsequent guard
/// failures, execution transfers to the bridge instead of returning to
/// the interpreter.
use majit_ir::{FailDescr, GcRef, Type, Value};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

/// Compiled bridge data attached to a guard's fail descriptor.
///
/// When a bridge is compiled, its code pointer and metadata are stored
/// here so `execute_token` can dispatch to the bridge on guard failure.
pub struct BridgeData {
    /// Function pointer to the bridge's compiled code.
    /// Same calling convention as a compiled loop:
    ///   fn(inputs_ptr: *const i64, outputs_ptr: *mut i64) -> i64
    pub code_ptr: *const u8,
    /// Fail descriptors within the bridge (guards + finish).
    pub fail_descrs: Vec<Arc<CraneliftFailDescr>>,
    /// Number of input arguments the bridge expects.
    pub num_inputs: usize,
    /// Maximum output slots for guard exits within the bridge.
    pub max_output_slots: usize,
}

unsafe impl Send for BridgeData {}
unsafe impl Sync for BridgeData {}

impl std::fmt::Debug for BridgeData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BridgeData")
            .field("code_ptr", &self.code_ptr)
            .field("num_inputs", &self.num_inputs)
            .finish()
    }
}

/// Concrete fail descriptor used by the Cranelift backend.
///
/// Carries the fail_index and the types of values that will be
/// saved in the DeadFrame on guard failure.
///
/// Also tracks guard failure count and an optional bridge that
/// should be executed instead of returning to the interpreter.
pub struct CraneliftFailDescr {
    pub fail_index: u32,
    pub fail_arg_types: Vec<Type>,
    /// Number of times this guard has failed (for bridge compilation heuristics).
    pub fail_count: AtomicU32,
    /// Compiled bridge attached to this guard, if any.
    pub bridge: Mutex<Option<BridgeData>>,
}

impl std::fmt::Debug for CraneliftFailDescr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CraneliftFailDescr")
            .field("fail_index", &self.fail_index)
            .field("fail_arg_types", &self.fail_arg_types)
            .field("fail_count", &self.fail_count.load(Ordering::Relaxed))
            .field("has_bridge", &self.bridge.lock().unwrap().is_some())
            .finish()
    }
}

impl CraneliftFailDescr {
    /// Create a new fail descriptor.
    pub fn new(fail_index: u32, fail_arg_types: Vec<Type>) -> Self {
        CraneliftFailDescr {
            fail_index,
            fail_arg_types,
            fail_count: AtomicU32::new(0),
            bridge: Mutex::new(None),
        }
    }

    /// Increment the failure counter and return the new value.
    pub fn increment_fail_count(&self) -> u32 {
        self.fail_count.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Get the current failure count.
    pub fn get_fail_count(&self) -> u32 {
        self.fail_count.load(Ordering::Relaxed)
    }

    /// Whether a bridge has been attached to this guard.
    pub fn has_bridge(&self) -> bool {
        self.bridge.lock().unwrap().is_some()
    }

    /// Attach a compiled bridge to this guard.
    pub fn attach_bridge(&self, bridge: BridgeData) {
        *self.bridge.lock().unwrap() = Some(bridge);
    }
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
