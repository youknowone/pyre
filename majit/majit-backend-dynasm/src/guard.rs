/// assembler.py ResumeGuardDescr parity: fail descriptor with
/// in-place patchable jump offset.
///
/// Unlike CraneliftFailDescr, this stores `adr_jump_offset` — the
/// address in compiled code where the guard's conditional jump can be
/// patched to redirect to a bridge (assembler.py:966).
use std::cell::UnsafeCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use majit_backend::ExitRecoveryLayout;
use majit_ir::{AccumVectorInfo, Descr, FailDescr, GuardPendingFieldEntry, RdVirtualInfo, Type};

/// assembler.py: ResumeGuardDescr concrete type for dynasm backend.
pub struct DynasmFailDescr {
    pub fail_index: u32,
    pub trace_id: u64,
    pub fail_arg_types: Vec<Type>,
    pub is_finish: bool,

    /// regalloc parity: fail_locs — maps fail_args[i] to jitframe slot.
    /// None = virtual/unmapped (not in jitframe).
    pub fail_arg_locs: Vec<Option<usize>>,

    /// Trace op index of the guard that produced this exit.
    pub source_op_index: Option<usize>,

    /// resume.py:450 — compact resume numbering (varint-encoded tagged values).
    pub rd_numb: Option<Vec<u8>>,
    /// resume.py:451 — shared constant pool referenced by rd_numb.
    pub rd_consts: Option<Vec<(i64, Type)>>,
    /// resume.py:488 — virtual object field info.
    pub rd_virtuals: Option<Vec<RdVirtualInfo>>,
    /// Deferred heap writes (SETFIELD_GC/SETARRAYITEM_GC with virtual values).
    pub rd_pendingfields: Option<Vec<GuardPendingFieldEntry>>,
    /// Backend-origin recovery layout, built at compile time from fail_arg_types.
    pub recovery_layout: UnsafeCell<Option<ExitRecoveryLayout>>,

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
            source_op_index: None,
            rd_numb: None,
            rd_consts: None,
            rd_virtuals: None,
            rd_pendingfields: None,
            recovery_layout: UnsafeCell::new(None),
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

    /// Read the recovery_layout.
    pub fn recovery_layout(&self) -> Option<ExitRecoveryLayout> {
        unsafe { &*self.recovery_layout.get() }.clone()
    }

    /// Set the recovery_layout.
    pub fn set_recovery_layout(&self, layout: ExitRecoveryLayout) {
        unsafe { *self.recovery_layout.get() = Some(layout) };
    }

    /// Build a FailDescrLayout for this descriptor (parity with CraneliftFailDescr::layout).
    pub fn layout(&self) -> majit_backend::FailDescrLayout {
        majit_backend::FailDescrLayout {
            fail_index: self.fail_index,
            fail_arg_types: self.fail_arg_types.clone(),
            is_finish: self.is_finish,
            trace_id: self.trace_id,
            source_op_index: self.source_op_index,
            gc_ref_slots: self
                .fail_arg_types
                .iter()
                .enumerate()
                .filter_map(|(i, tp)| (*tp == Type::Ref).then_some(i))
                .collect(),
            force_token_slots: Vec::new(),
            frame_stack: None,
            recovery_layout: self.recovery_layout(),
            trace_info: None,
        }
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

/// compile.py:618-669 done_with_this_frame_descr — four type-specific
/// singletons, matching RPython's DoneWithThisFrameDescrVoid/Int/Ref/Float.
///
/// CALL_ASSEMBLER's fast path compares jf_descr against the type-appropriate
/// singleton to decide whether the callee finished normally.
static DONE_WITH_THIS_FRAME_DESCR_VOID: std::sync::LazyLock<Arc<DynasmFailDescr>> =
    std::sync::LazyLock::new(|| Arc::new(DynasmFailDescr::new(u32::MAX, 0, vec![], true)));
static DONE_WITH_THIS_FRAME_DESCR_INT: std::sync::LazyLock<Arc<DynasmFailDescr>> =
    std::sync::LazyLock::new(|| Arc::new(DynasmFailDescr::new(u32::MAX, 0, vec![Type::Int], true)));
static DONE_WITH_THIS_FRAME_DESCR_REF: std::sync::LazyLock<Arc<DynasmFailDescr>> =
    std::sync::LazyLock::new(|| Arc::new(DynasmFailDescr::new(u32::MAX, 0, vec![Type::Ref], true)));
static DONE_WITH_THIS_FRAME_DESCR_FLOAT: std::sync::LazyLock<Arc<DynasmFailDescr>> =
    std::sync::LazyLock::new(|| {
        Arc::new(DynasmFailDescr::new(u32::MAX, 0, vec![Type::Float], true))
    });

/// compile.py:667 done_with_this_frame_descr_void
pub fn done_with_this_frame_descr_void_ptr() -> usize {
    Arc::as_ptr(&DONE_WITH_THIS_FRAME_DESCR_VOID) as usize
}
/// compile.py:668 done_with_this_frame_descr_int
pub fn done_with_this_frame_descr_int_ptr() -> usize {
    Arc::as_ptr(&DONE_WITH_THIS_FRAME_DESCR_INT) as usize
}
/// compile.py:669 done_with_this_frame_descr_ref
pub fn done_with_this_frame_descr_ref_ptr() -> usize {
    Arc::as_ptr(&DONE_WITH_THIS_FRAME_DESCR_REF) as usize
}
/// compile.py:670 done_with_this_frame_descr_float
pub fn done_with_this_frame_descr_float_ptr() -> usize {
    Arc::as_ptr(&DONE_WITH_THIS_FRAME_DESCR_FLOAT) as usize
}

/// Return the type-appropriate done_with_this_frame_descr pointer.
/// compile.py:324-336 call_assembler: selects descr by op.type.
pub fn done_with_this_frame_descr_ptr_for_type(tp: Type) -> usize {
    match tp {
        Type::Void => done_with_this_frame_descr_void_ptr(),
        Type::Int => done_with_this_frame_descr_int_ptr(),
        Type::Ref => done_with_this_frame_descr_ref_ptr(),
        Type::Float => done_with_this_frame_descr_float_ptr(),
    }
}

/// Backward compat: return the INT descr ptr (used by find_descr_by_ptr).
pub fn done_with_this_frame_descr_ptr() -> usize {
    done_with_this_frame_descr_int_ptr()
}

/// Check if a raw pointer matches ANY done_with_this_frame_descr variant.
/// Used by runner.rs find_descr_by_ptr to recognize finish exits.
pub fn is_done_with_this_frame_descr(ptr: usize) -> bool {
    ptr == done_with_this_frame_descr_void_ptr()
        || ptr == done_with_this_frame_descr_int_ptr()
        || ptr == done_with_this_frame_descr_ref_ptr()
        || ptr == done_with_this_frame_descr_float_ptr()
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
