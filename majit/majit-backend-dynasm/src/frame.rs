/// jitframe.py / assembler.py frame parity:
/// Stores saved values and descriptor reference from guard failure.
use std::sync::Arc;

use majit_ir::DescrRef;
use majit_ir::FailDescr;
use majit_ir::GcRef;

/// Concrete data stored in DeadFrame by the dynasm backend.
pub struct FrameData {
    /// Raw exit slot values.
    pub(crate) raw_values: Vec<i64>,
    /// Fail descriptor used for slot decoding / bridge data.  Trait
    /// object so the deadframe can carry either a backend-native
    /// `DynasmFailDescr` (production guard / FINISH path) or any
    /// other `FailDescr` impl once the Phase C-1 cascade lands the
    /// metainterp synthetic descrs directly through the deadframe.
    /// Backend-specific inherent helpers (`recovery_layout`,
    /// `layout`) reach the concrete type via
    /// `Descr::as_any().and_then(|a| a.downcast_ref::<DynasmFailDescr>())`.
    pub(crate) fail_descr: Arc<dyn FailDescr>,
    /// Original `jf_descr` object identity when the exit used an attached
    /// metainterp descr (`DoneWithThisFrame*` / `ExitFrameWithExceptionDescrRef`).
    pub(crate) latest_descr: Option<DescrRef>,
}

impl std::fmt::Debug for FrameData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FrameData")
            .field("num_values", &self.raw_values.len())
            .field("fail_index", &self.fail_descr.fail_index())
            .field(
                "latest_descr",
                &self.latest_descr.as_ref().map(|descr| descr.repr()),
            )
            .finish()
    }
}

impl FrameData {
    pub fn new(
        raw_values: Vec<i64>,
        fail_descr: Arc<dyn FailDescr>,
        latest_descr: Option<DescrRef>,
    ) -> Self {
        FrameData {
            raw_values,
            fail_descr,
            latest_descr,
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
