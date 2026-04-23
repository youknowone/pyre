//! ValueId → ValueType side table (pre-rtyper annotator output) for the
//! jit_codewriter IR.
//!
//! PRE-EXISTING-ADAPTATION. RPython's `RPythonAnnotator.complete()`
//! attaches a `SomeValue` to each `Variable.annotation` slot directly on
//! the flowgraph. Pyre's jit_codewriter consumes a
//! `crate::model::FunctionGraph` (value-id-based), so the annotator
//! output lives in this side table instead. The `annotate()` algorithm
//! remains in `translate_legacy/annotator/annrpython.rs` until the real
//! annotator (`annotator/annrpython.rs`) produces per-Variable
//! annotations end-to-end and replaces it.

use std::collections::HashMap;

use crate::model::{ValueId, ValueType};

/// Annotation state: `ValueId → ValueType`.
#[derive(Debug, Clone)]
pub struct AnnotationState {
    pub types: HashMap<ValueId, ValueType>,
}

impl AnnotationState {
    pub fn new() -> Self {
        Self {
            types: HashMap::new(),
        }
    }

    pub fn get(&self, id: ValueId) -> &ValueType {
        self.types.get(&id).unwrap_or(&ValueType::Unknown)
    }

    pub fn set(&mut self, id: ValueId, ty: ValueType) {
        self.types.insert(id, ty);
    }
}
