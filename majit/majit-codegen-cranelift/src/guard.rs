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
use crate::compiler::{register_gc_roots, release_force_token, unregister_gc_roots};
use majit_gc::GcMap;
use majit_ir::{FailDescr, GcRef, Type};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

/// Compiled bridge data attached to a guard's fail descriptor.
///
/// When a bridge is compiled, its code pointer and metadata are stored
/// here so `execute_token` can dispatch to the bridge on guard failure.
pub struct BridgeData {
    /// Function pointer to the bridge's compiled code.
    /// Same calling convention as a compiled loop:
    ///   fn(inputs_ptr: *const i64, outputs_ptr: *mut i64, roots_ptr: *mut i64) -> i64
    pub code_ptr: *const u8,
    /// Fail descriptors within the bridge (guards + finish).
    pub fail_descrs: Vec<Arc<CraneliftFailDescr>>,
    /// GC runtime used by the compiled bridge, if any.
    pub gc_runtime_id: Option<u64>,
    /// Number of input arguments the bridge expects.
    pub num_inputs: usize,
    /// Number of shadow-root slots the bridge expects.
    pub num_ref_roots: usize,
    /// Maximum output slots for guard exits within the bridge.
    pub max_output_slots: usize,
    /// Whether any guard in this bridge uses FORCE_TOKEN slots.
    pub needs_force_frame: bool,
}

unsafe impl Send for BridgeData {}
unsafe impl Sync for BridgeData {}

impl std::fmt::Debug for BridgeData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BridgeData")
            .field("code_ptr", &self.code_ptr)
            .field("gc_runtime_id", &self.gc_runtime_id)
            .field("num_inputs", &self.num_inputs)
            .field("num_ref_roots", &self.num_ref_roots)
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
    pub trace_id: u64,
    pub fail_arg_types: Vec<Type>,
    pub gc_map: GcMap,
    pub is_finish: bool,
    pub force_token_slots: Vec<usize>,
    /// Number of times this guard has failed (for bridge compilation heuristics).
    pub fail_count: AtomicU32,
    /// Compiled bridge attached to this guard, if any.
    pub bridge: Mutex<Option<BridgeData>>,
}

impl std::fmt::Debug for CraneliftFailDescr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CraneliftFailDescr")
            .field("fail_index", &self.fail_index)
            .field("trace_id", &self.trace_id)
            .field("fail_arg_types", &self.fail_arg_types)
            .field("gc_map", &self.gc_map)
            .field("is_finish", &self.is_finish)
            .field("force_token_slots", &self.force_token_slots)
            .field("fail_count", &self.fail_count.load(Ordering::Relaxed))
            .field("has_bridge", &self.bridge.lock().unwrap().is_some())
            .finish()
    }
}

impl CraneliftFailDescr {
    fn gc_map_for_types(fail_arg_types: &[Type], force_token_slots: &[usize]) -> GcMap {
        let mut gc_map = GcMap::new();
        for (slot, tp) in fail_arg_types.iter().enumerate() {
            if *tp == Type::Ref && !force_token_slots.contains(&slot) {
                gc_map.set_ref(slot);
            }
        }
        gc_map
    }

    /// Create a new fail descriptor.
    pub fn new(fail_index: u32, fail_arg_types: Vec<Type>) -> Self {
        Self::new_with_trace_and_kind_and_force_tokens(
            fail_index,
            0,
            fail_arg_types,
            false,
            Vec::new(),
        )
    }

    pub fn new_with_kind(fail_index: u32, fail_arg_types: Vec<Type>, is_finish: bool) -> Self {
        Self::new_with_trace_and_kind_and_force_tokens(
            fail_index,
            0,
            fail_arg_types,
            is_finish,
            Vec::new(),
        )
    }

    pub fn new_with_kind_and_force_tokens(
        fail_index: u32,
        fail_arg_types: Vec<Type>,
        is_finish: bool,
        force_token_slots: Vec<usize>,
    ) -> Self {
        Self::new_with_trace_and_kind_and_force_tokens(
            fail_index,
            0,
            fail_arg_types,
            is_finish,
            force_token_slots,
        )
    }

    pub fn new_with_trace_and_kind_and_force_tokens(
        fail_index: u32,
        trace_id: u64,
        fail_arg_types: Vec<Type>,
        is_finish: bool,
        mut force_token_slots: Vec<usize>,
    ) -> Self {
        force_token_slots.sort_unstable();
        force_token_slots.dedup();
        CraneliftFailDescr {
            fail_index,
            trace_id,
            gc_map: Self::gc_map_for_types(&fail_arg_types, &force_token_slots),
            fail_arg_types,
            is_finish,
            force_token_slots,
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

    pub fn gc_map(&self) -> &GcMap {
        &self.gc_map
    }

    pub fn is_finish(&self) -> bool {
        self.is_finish
    }

    pub fn is_force_token_slot(&self, slot: usize) -> bool {
        self.force_token_slots.binary_search(&slot).is_ok()
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

    fn is_finish(&self) -> bool {
        self.is_finish
    }

    fn trace_id(&self) -> u64 {
        self.trace_id
    }
}

/// The concrete data stored in a `DeadFrame` by the Cranelift backend.
///
/// Holds the saved values and a reference to the fail descriptor that
/// identifies which guard failed.
pub struct FrameData {
    /// Raw exit slots as produced by compiled code.
    raw_values: Vec<i64>,
    /// Rooted GC references extracted from `raw_values`.
    rooted_refs: Vec<GcRef>,
    /// Mapping from exit slot index to rooted ref index.
    ref_slot_map: Vec<Option<usize>>,
    /// Opaque force-token handles returned as Ref values.
    owned_force_tokens: Vec<u64>,
    /// Optional saved-data GC ref associated with this dead frame.
    saved_data: Option<Box<GcRef>>,
    /// Optional pending exception GC ref associated with this dead frame.
    exception: Option<Box<GcRef>>,
    /// Pending exception class associated with this dead frame.
    exception_class: i64,
    /// The fail descriptor identifying the guard that failed.
    pub fail_descr: Arc<CraneliftFailDescr>,
    gc_runtime_id: Option<u64>,
}

impl FrameData {
    pub fn new(
        raw_values: Vec<i64>,
        fail_descr: Arc<CraneliftFailDescr>,
        gc_runtime_id: Option<u64>,
    ) -> Self {
        Self::new_with_savedata_and_exception(raw_values, fail_descr, gc_runtime_id, None, 0, None)
    }

    pub fn new_preview(
        raw_values: Vec<i64>,
        fail_descr: Arc<CraneliftFailDescr>,
        gc_runtime_id: Option<u64>,
    ) -> Self {
        Self::new_inner(raw_values, fail_descr, gc_runtime_id, None, 0, None, false)
    }

    pub fn new_with_savedata(
        raw_values: Vec<i64>,
        fail_descr: Arc<CraneliftFailDescr>,
        gc_runtime_id: Option<u64>,
        saved_data: Option<GcRef>,
    ) -> Self {
        Self::new_with_savedata_and_exception(
            raw_values,
            fail_descr,
            gc_runtime_id,
            saved_data,
            0,
            None,
        )
    }

    pub fn new_with_savedata_and_exception(
        raw_values: Vec<i64>,
        fail_descr: Arc<CraneliftFailDescr>,
        gc_runtime_id: Option<u64>,
        saved_data: Option<GcRef>,
        exception_class: i64,
        exception: Option<GcRef>,
    ) -> Self {
        Self::new_inner(
            raw_values,
            fail_descr,
            gc_runtime_id,
            saved_data,
            exception_class,
            exception,
            true,
        )
    }

    fn new_inner(
        raw_values: Vec<i64>,
        fail_descr: Arc<CraneliftFailDescr>,
        gc_runtime_id: Option<u64>,
        saved_data: Option<GcRef>,
        exception_class: i64,
        exception: Option<GcRef>,
        own_force_tokens: bool,
    ) -> Self {
        let mut rooted_refs = Vec::new();
        let mut ref_slot_map = vec![None; fail_descr.fail_arg_types.len()];
        let mut owned_force_tokens = Vec::new();
        for (index, tp) in fail_descr.fail_arg_types.iter().enumerate() {
            if *tp == Type::Ref && !fail_descr.is_force_token_slot(index) {
                ref_slot_map[index] = Some(rooted_refs.len());
                rooted_refs.push(GcRef(raw_values[index] as usize));
            } else if own_force_tokens && fail_descr.is_force_token_slot(index) {
                owned_force_tokens.push(raw_values[index] as u64);
            }
        }
        if let Some(runtime_id) = gc_runtime_id {
            register_gc_roots(runtime_id, &mut rooted_refs);
        }
        let mut saved_data = saved_data.map(Box::new);
        if let (Some(runtime_id), Some(saved_data)) = (gc_runtime_id, saved_data.as_mut()) {
            register_gc_roots(runtime_id, std::slice::from_mut(saved_data.as_mut()));
        }
        let mut exception = exception.filter(|exc| !exc.is_null()).map(Box::new);
        if let (Some(runtime_id), Some(exception)) = (gc_runtime_id, exception.as_mut()) {
            register_gc_roots(runtime_id, std::slice::from_mut(exception.as_mut()));
        }

        FrameData {
            raw_values,
            rooted_refs,
            ref_slot_map,
            owned_force_tokens,
            saved_data,
            exception,
            exception_class,
            fail_descr,
            gc_runtime_id,
        }
    }

    pub fn get_int(&self, index: usize) -> i64 {
        match self.fail_descr.fail_arg_types[index] {
            Type::Int => self.raw_values[index],
            other => panic!("expected Int at index {index}, got {other:?}"),
        }
    }

    pub fn get_float(&self, index: usize) -> f64 {
        match self.fail_descr.fail_arg_types[index] {
            Type::Float => f64::from_bits(self.raw_values[index] as u64),
            other => panic!("expected Float at index {index}, got {other:?}"),
        }
    }

    pub fn get_ref(&self, index: usize) -> GcRef {
        match self.fail_descr.fail_arg_types[index] {
            Type::Ref => {
                if self.fail_descr.is_force_token_slot(index) {
                    return GcRef(self.raw_values[index] as usize);
                }
                let rooted_index = self.ref_slot_map[index].expect("missing rooted ref slot");
                self.rooted_refs[rooted_index]
            }
            other => panic!("expected Ref at index {index}, got {other:?}"),
        }
    }

    pub fn set_savedata_ref(&mut self, data: GcRef) {
        if let Some(saved_data) = self.saved_data.as_mut() {
            if let Some(runtime_id) = self.gc_runtime_id {
                unregister_gc_roots(runtime_id, std::slice::from_mut(saved_data.as_mut()));
            }
            **saved_data = data;
            if let Some(runtime_id) = self.gc_runtime_id {
                register_gc_roots(runtime_id, std::slice::from_mut(saved_data.as_mut()));
            }
            return;
        }

        let mut saved_data = Box::new(data);
        if let Some(runtime_id) = self.gc_runtime_id {
            register_gc_roots(runtime_id, std::slice::from_mut(saved_data.as_mut()));
        }
        self.saved_data = Some(saved_data);
    }

    pub fn get_savedata_ref(&self) -> GcRef {
        self.saved_data
            .as_ref()
            .map(|saved_data| **saved_data)
            .expect("dead frame has no saved-data ref")
    }

    pub fn get_exception_ref(&self) -> GcRef {
        self.exception
            .as_ref()
            .map(|exception| **exception)
            .unwrap_or(GcRef::NULL)
    }

    pub fn get_exception_class(&self) -> i64 {
        self.exception_class
    }
}

impl Drop for FrameData {
    fn drop(&mut self) {
        if let Some(runtime_id) = self.gc_runtime_id {
            unregister_gc_roots(runtime_id, &mut self.rooted_refs);
            if let Some(saved_data) = self.saved_data.as_mut() {
                unregister_gc_roots(runtime_id, std::slice::from_mut(saved_data.as_mut()));
            }
            if let Some(exception) = self.exception.as_mut() {
                unregister_gc_roots(runtime_id, std::slice::from_mut(exception.as_mut()));
            }
        }
        for &handle in &self.owned_force_tokens {
            release_force_token(handle);
        }
    }
}
