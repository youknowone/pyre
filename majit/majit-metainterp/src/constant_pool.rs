//! No direct RPython equivalent — Rust-specific constant deduplication
//! for OpRef indices (RPython manages constants implicitly in Trace).

use std::collections::HashMap;

use majit_ir::{OpRef, Type};

/// Constant pool for trace recording.
///
/// Manages the mapping from OpRef (>= 10_000) to i64 values.
/// Automatically deduplicates identical values.
///
/// RPython manages constants implicitly in Trace with full type info.
/// pyre's ConstantPool stores types separately so the optimizer and
/// backend can distinguish Ref constants (function pointers, object
/// addresses) from Int constants.
pub struct ConstantPool {
    constants: HashMap<u32, i64>,
    /// Type of each constant OpRef. Populated by `get_or_insert_typed`.
    constant_types: HashMap<u32, Type>,
    /// Resume-data-only type overrides. Not exposed to Cranelift backend.
    /// RPython parity: GcRef pointers recorded as const_int need Ref type
    /// for resume data encoding without triggering GC root tracking.
    numbering_type_overrides: HashMap<u32, Type>,
    next_ref: u32,
}

impl ConstantPool {
    pub fn new() -> Self {
        ConstantPool {
            constants: HashMap::new(),
            constant_types: HashMap::new(),
            numbering_type_overrides: HashMap::new(),
            next_ref: 10_000,
        }
    }

    /// Get or create a constant OpRef for a given i64 value.
    /// Returns the same OpRef for the same value (deduplication).
    pub fn get_or_insert(&mut self, value: i64) -> OpRef {
        for (&idx, &v) in &self.constants {
            if v == value {
                return OpRef(idx);
            }
        }
        let opref = OpRef(self.next_ref);
        self.next_ref += 1;
        self.constants.insert(opref.0, value);
        opref
    }

    /// Get or create a typed constant OpRef.
    pub fn get_or_insert_typed(&mut self, value: i64, tp: Type) -> OpRef {
        let opref = self.get_or_insert(value);
        self.constant_types.insert(opref.0, tp);
        opref
    }

    /// Get the type of a constant, if recorded.
    pub fn constant_type(&self, opref: OpRef) -> Option<Type> {
        self.constant_types.get(&opref.0).copied()
    }

    /// Mark an existing constant with a specific type for resume data only.
    /// RPython parity: GcRef pointers stored via const_int() need Ref type
    /// for correct resume data encoding, but Cranelift must treat them as Int
    /// (no GC root tracking for static type descriptor pointers).
    pub fn mark_type(&mut self, opref: OpRef, tp: Type) {
        // Store in numbering_type_overrides only — NOT constant_types.
        // This type info is for resume data (consumer switchover) only,
        // NOT for Cranelift backend (which would trigger GC root tracking).
        self.numbering_type_overrides.insert(opref.0, tp);
    }

    /// Consume the pool and return the constants map and type map.
    pub fn into_inner(self) -> HashMap<u32, i64> {
        self.constants
    }

    /// Consume the pool, returning both value and type maps.
    pub fn into_inner_with_types(self) -> (HashMap<u32, i64>, HashMap<u32, Type>) {
        (self.constants, self.constant_types)
    }

    /// Get numbering type overrides (for resume data only, not Cranelift).
    pub fn numbering_type_overrides(&self) -> &HashMap<u32, Type> {
        &self.numbering_type_overrides
    }

    /// Get a mutable reference to the inner constants map.
    pub fn as_mut(&mut self) -> &mut HashMap<u32, i64> {
        &mut self.constants
    }

    /// Get a shared reference to the inner constants map.
    pub fn as_ref(&self) -> &HashMap<u32, i64> {
        &self.constants
    }

    /// Clone the constants map without consuming the pool.
    /// Used by compile_trace which needs a copy for compilation
    /// while keeping the pool alive for potential continued tracing.
    pub fn snapshot(&self) -> HashMap<u32, i64> {
        self.constants.clone()
    }

    /// Clone the constant type map without consuming the pool.
    pub fn constant_types_snapshot(&self) -> HashMap<u32, Type> {
        self.constant_types.clone()
    }
}

impl Default for ConstantPool {
    fn default() -> Self {
        Self::new()
    }
}
