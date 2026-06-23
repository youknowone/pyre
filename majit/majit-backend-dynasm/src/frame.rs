/// jitframe.py / assembler.py frame parity:
/// Stores saved values and descriptor reference from guard failure.
use majit_ir::DescrRef;
use majit_ir::GcRef;

/// Concrete data stored in DeadFrame by the dynasm backend.
///
/// `fail_descr` carries the metainterp class-distinct Arc identity
/// (ResumeGuardDescr family for guards, DoneWithThisFrame*/
/// ExitFrameWithExceptionDescrRef for FINISH exits).  FailDescr-trait
/// operations on the descr go through `DescrRef::as_fail_descr`.
pub struct FrameData {
    /// Raw exit slot values.
    pub(crate) raw_values: Vec<i64>,
    /// Backend-local fail descriptor used for slot decoding / bridge data.
    pub(crate) fail_descr: DescrRef,
    /// Original `jf_descr` object identity when the exit used an attached
    /// metainterp descr (`DoneWithThisFrame*` / `ExitFrameWithExceptionDescrRef`).
    pub(crate) latest_descr: Option<DescrRef>,
    /// `jf_guard_exc` captured off the deadframe tip before the libc jitframe
    /// chain is freed.  `cpu.grab_exc_value(deadframe)` (llmodel.py:240) reads
    /// it back; the jitframe is gone by then, so the value is staged here.
    ///
    /// Held as a bare `GcRef` (not a registered GC root): exception instances
    /// are `malloc_typed` Box-immortal today (interp_exceptions.rs:843-844), so the
    /// pointer can never dangle.  The whole exception channel
    /// (`ExceptionState.exc_value`, threaded as a raw `i64`) shares this
    /// assumption; future GC-managed exceptions must root all of it, not just
    /// this slot.
    pub(crate) exc_value: GcRef,
}

impl std::fmt::Debug for FrameData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FrameData")
            .field("num_values", &self.raw_values.len())
            .field("fail_descr", &self.fail_descr.repr())
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
        fail_descr: DescrRef,
        latest_descr: Option<DescrRef>,
        exc_value: GcRef,
    ) -> Self {
        FrameData {
            raw_values,
            fail_descr,
            latest_descr,
            exc_value,
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
