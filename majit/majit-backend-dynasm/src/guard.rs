/// assembler.py ResumeGuardDescr parity: fail descriptor with
/// in-place patchable jump offset.
///
/// Unlike CraneliftFailDescr, this stores `adr_jump_offset` — the
/// address in compiled code where the guard's conditional jump can be
/// patched to redirect to a bridge (assembler.py:966).
use std::cell::UnsafeCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use majit_ir::{AccumVectorInfo, Descr, FailDescr, Type};

/// assembler.py: ResumeGuardDescr concrete type for dynasm backend.
pub struct DynasmFailDescr {
    pub fail_index: u32,
    pub trace_id: u64,
    pub fail_arg_types: Vec<Type>,
    pub is_finish: bool,

    /// regalloc parity: fail_locs — maps fail_args[i] to jitframe slot.
    /// None = virtual/unmapped (not in jitframe).
    pub fail_arg_locs: Vec<Option<usize>>,

    /// compile.py:685 status: packs ST_BUSY_FLAG + type tag + hash.
    pub status: AtomicU64,

    /// assembler.py:966 adr_jump_offset: address in machine code where
    /// the guard's conditional jump offset is stored. Used by
    /// patch_jump_for_descr to redirect to a bridge.
    /// 0 means "already patched" (assembler.py:987).
    pub adr_jump_offset: UnsafeCell<usize>,

    /// Bridge code pointer (if a bridge has been compiled for this guard).
    /// Unlike Cranelift, we don't need bridge data — the machine code is
    /// patched in place to jump directly to the bridge.
    pub bridge_addr: UnsafeCell<usize>,
    // fail_args_slots removed: bridge source_slots are derived from
    // fail_arg_locs via rebuild_faillocs_from_descr (assembler.py:201).
}

// Safety: single-threaded JIT (like RPython with GIL).
unsafe impl Send for DynasmFailDescr {}
unsafe impl Sync for DynasmFailDescr {}

impl DynasmFailDescr {
    // compile.py:687-696 status encoding constants.
    pub const ST_BUSY_FLAG: u64 = 0x01;
    pub const ST_TYPE_MASK: u64 = 0x06;
    pub const ST_SHIFT: u32 = 3;
    pub const ST_SHIFT_MASK: u64 = !((1u64 << Self::ST_SHIFT) - 1);
    pub const TY_NONE: u64 = 0x00;
    pub const TY_INT: u64 = 0x02;
    pub const TY_REF: u64 = 0x04;
    pub const TY_FLOAT: u64 = 0x06;

    pub fn new(fail_index: u32, trace_id: u64, fail_arg_types: Vec<Type>, is_finish: bool) -> Self {
        DynasmFailDescr {
            fail_index,
            trace_id,
            fail_arg_types,
            is_finish,
            fail_arg_locs: Vec::new(),
            status: AtomicU64::new(0),
            adr_jump_offset: UnsafeCell::new(0),
            bridge_addr: UnsafeCell::new(0),
        }
    }

    /// compile.py:826-830 store_hash.
    pub fn store_hash(&self, hash: u64) {
        self.status
            .store(hash & Self::ST_SHIFT_MASK, Ordering::Release);
    }

    /// compile.py:741-745 get_status.
    pub fn get_status(&self) -> u64 {
        self.status.load(Ordering::Acquire)
    }

    /// compile.py:786-788 start_compiling.
    pub fn start_compiling(&self) {
        self.status.fetch_or(Self::ST_BUSY_FLAG, Ordering::AcqRel);
    }

    /// compile.py:790-795 done_compiling.
    pub fn done_compiling(&self) {
        self.status.fetch_and(!Self::ST_BUSY_FLAG, Ordering::AcqRel);
    }

    /// compile.py:813-824 make_a_counter_per_value.
    pub fn make_a_counter_per_value(&self, index: u32, type_tag: u64) {
        let status = type_tag | ((index as u64) << Self::ST_SHIFT);
        self.status.store(status, Ordering::Release);
    }

    /// assembler.py:966 — read adr_jump_offset.
    pub fn adr_jump_offset(&self) -> usize {
        unsafe { *self.adr_jump_offset.get() }
    }

    /// assembler.py:987 — set adr_jump_offset (0 = "patched").
    pub fn set_adr_jump_offset(&self, offset: usize) {
        unsafe { *self.adr_jump_offset.get() = offset };
    }

    /// Check if a bridge has been patched for this guard.
    pub fn has_bridge(&self) -> bool {
        unsafe { *self.bridge_addr.get() != 0 }
    }

    /// Set the bridge address after patching.
    pub fn set_bridge_addr(&self, addr: usize) {
        unsafe { *self.bridge_addr.get() = addr };
    }
}

impl std::fmt::Debug for DynasmFailDescr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynasmFailDescr")
            .field("fail_index", &self.fail_index)
            .field("trace_id", &self.trace_id)
            .field("is_finish", &self.is_finish)
            .field("status", &self.get_status())
            .field("adr_jump_offset", &self.adr_jump_offset())
            .field("has_bridge", &self.has_bridge())
            .finish()
    }
}

/// compile.py:665-674 done_with_this_frame_descr singleton.
/// All Finish ops write this pointer to jf_descr. CALL_ASSEMBLER
/// compares jf_descr against this pointer for the fast path.
static DONE_WITH_THIS_FRAME_DESCR: std::sync::LazyLock<Arc<DynasmFailDescr>> =
    std::sync::LazyLock::new(|| Arc::new(DynasmFailDescr::new(u32::MAX, 0, vec![Type::Int], true)));

/// Return the raw pointer for done_with_this_frame_descr.
/// Used by genop_call_assembler for inline CMP.
pub fn done_with_this_frame_descr_ptr() -> usize {
    Arc::as_ptr(&DONE_WITH_THIS_FRAME_DESCR) as usize
}

impl Descr for DynasmFailDescr {}

impl FailDescr for DynasmFailDescr {
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

    fn get_status(&self) -> u64 {
        self.get_status()
    }

    fn start_compiling(&self) {
        self.start_compiling()
    }

    fn done_compiling(&self) {
        self.done_compiling()
    }

    fn is_compiling(&self) -> bool {
        self.status.load(Ordering::Acquire) & Self::ST_BUSY_FLAG != 0
    }
}
