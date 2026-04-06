/// jitframe.py / assembler.py frame parity:
/// Stores saved values and descriptor reference from guard failure.
use std::sync::Arc;

use majit_ir::{GcRef, Type};

use crate::guard::DynasmFailDescr;

/// Concrete data stored in DeadFrame by the dynasm backend.
pub struct FrameData {
    /// Raw exit slot values.
    pub(crate) raw_values: Vec<i64>,
    /// The fail descriptor that identifies which guard failed.
    pub(crate) fail_descr: Arc<DynasmFailDescr>,
}

impl std::fmt::Debug for FrameData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FrameData")
            .field("num_values", &self.raw_values.len())
            .field("fail_descr", &self.fail_descr)
            .finish()
    }
}

impl FrameData {
    pub fn new(raw_values: Vec<i64>, fail_descr: Arc<DynasmFailDescr>) -> Self {
        FrameData {
            raw_values,
            fail_descr,
        }
    }

    pub fn get_int(&self, index: usize) -> i64 {
        self.raw_values.get(index).copied().unwrap_or(0)
    }

    pub fn get_float(&self, index: usize) -> f64 {
        let bits = self.raw_values.get(index).copied().unwrap_or(0) as u64;
        f64::from_bits(bits)
    }

    pub fn get_ref(&self, index: usize) -> GcRef {
        let raw = self.raw_values.get(index).copied().unwrap_or(0);
        GcRef(raw as usize)
    }
}
