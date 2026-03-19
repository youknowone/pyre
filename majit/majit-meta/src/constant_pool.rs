//! No direct RPython equivalent — Rust-specific constant deduplication
//! for OpRef indices (RPython manages constants implicitly in Trace).

use std::collections::HashMap;

use majit_ir::OpRef;

/// Constant pool for trace recording.
///
/// Manages the mapping from OpRef (>= 10_000) to i64 values.
/// Automatically deduplicates identical values.
pub struct ConstantPool {
    constants: HashMap<u32, i64>,
    next_ref: u32,
}

impl ConstantPool {
    pub fn new() -> Self {
        ConstantPool {
            constants: HashMap::new(),
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

    /// Consume the pool and return the inner constants map.
    pub fn into_inner(self) -> HashMap<u32, i64> {
        self.constants
    }

    /// Get a mutable reference to the inner constants map.
    pub fn as_mut(&mut self) -> &mut HashMap<u32, i64> {
        &mut self.constants
    }

    /// Get a shared reference to the inner constants map.
    pub fn as_ref(&self) -> &HashMap<u32, i64> {
        &self.constants
    }
}

impl Default for ConstantPool {
    fn default() -> Self {
        Self::new()
    }
}
