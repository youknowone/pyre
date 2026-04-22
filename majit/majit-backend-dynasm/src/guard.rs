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
use majit_ir::{Descr, FailDescr, GuardPendingFieldEntry, RdVirtualInfo, Type};

/// Re-export the shared struct so existing `crate::guard::AttachedDescrPtrs`
/// imports in `x86/assembler.rs` / `aarch64/assembler.rs` / `runner.rs`
/// keep resolving while the canonical definition lives in `majit-backend`
/// (shared with cranelift via `majit_backend::AttachedDescrPtrs`).
pub use majit_backend::AttachedDescrPtrs;

/// assembler.py: ResumeGuardDescr concrete type for dynasm backend.
pub struct DynasmFailDescr {
    pub fail_index: u32,
    pub trace_id: u64,
    pub fail_arg_types: Vec<Type>,
    pub is_finish: bool,
    /// compile.py:658-662 ExitFrameWithExceptionDescrRef parity.
    /// True when this FINISH was emitted via
    /// pyjitpl.py:3238-3245 compile_exit_frame_with_exception.
    pub is_exit_frame_with_exception: bool,

    /// regalloc parity: fail_locs — maps fail_args[i] to jitframe slot.
    /// None = virtual/unmapped (not in jitframe).
    pub fail_arg_locs: Vec<Option<usize>>,
    /// llsupport/assembler.py: rd_locs parity.
    pub rd_locs: Vec<u16>,

    /// Trace op index of the guard that produced this exit.
    pub source_op_index: Option<usize>,

    /// resume.py:450 — compact resume numbering (varint-encoded tagged values).
    pub rd_numb: Option<Vec<u8>>,
    /// resume.py:451 — shared constant pool referenced by rd_numb.
    pub rd_consts: Option<Vec<(i64, Type)>>,
    /// resume.py:488 — virtual object field info. Entries are
    /// `Rc<RdVirtualInfo>` so cache-hit dedup at resume.py:307-315 preserves
    /// object identity across guard boundaries.
    pub rd_virtuals: Option<Vec<std::rc::Rc<RdVirtualInfo>>>,
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
            is_exit_frame_with_exception: false,
            fail_arg_locs: Vec::new(),
            rd_locs: Vec::new(),
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

    /// Read the compiled bridge entry address.
    pub fn bridge_addr(&self) -> usize {
        unsafe { *self.bridge_addr.get() }
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
            // resume.py:450-488 propagate rd_* so `compiled_exit_layout_from_backend`
            // can reach them after the frontend trace cache evicts the owning
            // `CompiledTrace` entry (pyjitpl/mod.rs:817-845).
            rd_numb: self.rd_numb.clone(),
            rd_consts: self.rd_consts.clone(),
            rd_virtuals: self.rd_virtuals.clone(),
            rd_pendingfields: self.rd_pendingfields.clone(),
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

/// `compile.py:665-674` `make_and_attach_done_descrs([self, cpu])` —
/// per-result-type `DoneWithThisFrame*` singleton attached by the
/// metainterp side at `pyjitpl.py:2222`.  The `Arc` lives on
/// `MetaInterpStaticData` and is re-published here via
/// `Backend::set_done_with_this_frame_descr_*` so the CALL_ASSEMBLER
/// fast path (`runner.rs::call_assembler_helper_trampoline`) can
/// compare `jf_descr` against `Arc::as_ptr` of the same `Arc` the
/// metainterp reads back in `handle_fail`.
///
/// `compile.py:665` `setattr(cpu, name, descr)` binds the descr to a
/// specific cpu instance; each `(metainterp_sd, cpu)` pair gets its own
/// attachment, and re-running `make_and_attach_done_descrs` overwrites.
/// Pyre stores the descrs in per-thread slots instead of per-`Backend`
/// instance fields: the extern-C CA helper trampoline
/// (`runner.rs::call_assembler_helper_trampoline`) and Cranelift-emitted
/// machine code resolve the identity without a backend receiver in
/// scope, and production deploys one backend per thread — so the
/// thread-local captures the same "one attachment per backend" lifetime
/// the RPython instance attribute does.  Tests that spin up multiple
/// backend instances on the same thread observe last-attach-wins, also
/// matching `setattr` semantics.
thread_local! {
    static DONE_WITH_THIS_FRAME_DESCR_VOID: std::cell::RefCell<Option<majit_ir::DescrRef>> =
        const { std::cell::RefCell::new(None) };
    static DONE_WITH_THIS_FRAME_DESCR_INT: std::cell::RefCell<Option<majit_ir::DescrRef>> =
        const { std::cell::RefCell::new(None) };
    static DONE_WITH_THIS_FRAME_DESCR_REF: std::cell::RefCell<Option<majit_ir::DescrRef>> =
        const { std::cell::RefCell::new(None) };
    static DONE_WITH_THIS_FRAME_DESCR_FLOAT: std::cell::RefCell<Option<majit_ir::DescrRef>> =
        const { std::cell::RefCell::new(None) };
    static EXIT_FRAME_WITH_EXCEPTION_DESCR_REF: std::cell::RefCell<Option<majit_ir::DescrRef>> =
        const { std::cell::RefCell::new(None) };
    static PROPAGATE_EXCEPTION_DESCR: std::cell::RefCell<Option<majit_ir::DescrRef>> =
        const { std::cell::RefCell::new(None) };
}

/// PRE-EXISTING-ADAPTATION: standalone backend tests construct a
/// `DynasmBackend` without a `MetaInterp`, so the thread-local slots
/// above remain unpopulated.  These fallbacks preserve the per-result-
/// type distinct-pointer invariant the tests assert.  In the production
/// path `MetaInterp::new` runs `attach_descrs_to_cpu` first, which
/// fills the slots; the getter functions prefer the attached value and
/// only fall back here when the slot is still `None`.
static FALLBACK_DONE_VOID: std::sync::LazyLock<Arc<DynasmFailDescr>> =
    std::sync::LazyLock::new(|| Arc::new(DynasmFailDescr::new(u32::MAX, 0, vec![], true)));
static FALLBACK_DONE_INT: std::sync::LazyLock<Arc<DynasmFailDescr>> =
    std::sync::LazyLock::new(|| Arc::new(DynasmFailDescr::new(u32::MAX, 0, vec![Type::Int], true)));
static FALLBACK_DONE_REF: std::sync::LazyLock<Arc<DynasmFailDescr>> =
    std::sync::LazyLock::new(|| Arc::new(DynasmFailDescr::new(u32::MAX, 0, vec![Type::Ref], true)));
static FALLBACK_DONE_FLOAT: std::sync::LazyLock<Arc<DynasmFailDescr>> =
    std::sync::LazyLock::new(|| {
        Arc::new(DynasmFailDescr::new(u32::MAX, 0, vec![Type::Float], true))
    });

pub(crate) fn set_done_with_this_frame_descr_void(descr: majit_ir::DescrRef) {
    DONE_WITH_THIS_FRAME_DESCR_VOID.with(|c| *c.borrow_mut() = Some(descr));
}
pub(crate) fn set_done_with_this_frame_descr_int(descr: majit_ir::DescrRef) {
    DONE_WITH_THIS_FRAME_DESCR_INT.with(|c| *c.borrow_mut() = Some(descr));
}
pub(crate) fn set_done_with_this_frame_descr_ref(descr: majit_ir::DescrRef) {
    DONE_WITH_THIS_FRAME_DESCR_REF.with(|c| *c.borrow_mut() = Some(descr));
}
pub(crate) fn set_done_with_this_frame_descr_float(descr: majit_ir::DescrRef) {
    DONE_WITH_THIS_FRAME_DESCR_FLOAT.with(|c| *c.borrow_mut() = Some(descr));
}
pub(crate) fn set_exit_frame_with_exception_descr_ref(descr: majit_ir::DescrRef) {
    EXIT_FRAME_WITH_EXCEPTION_DESCR_REF.with(|c| *c.borrow_mut() = Some(descr));
}
pub(crate) fn set_propagate_exception_descr(descr: majit_ir::DescrRef) {
    PROPAGATE_EXCEPTION_DESCR.with(|c| *c.borrow_mut() = Some(descr));
}

/// `Arc::as_ptr` of the metainterp-attached descr, falling back to
/// the local `DynasmFailDescr` singleton when no attachment has
/// happened (backend-only tests).
fn descr_ref_ptr(
    slot: &'static std::thread::LocalKey<std::cell::RefCell<Option<majit_ir::DescrRef>>>,
    fallback: &std::sync::LazyLock<Arc<DynasmFailDescr>>,
) -> usize {
    let attached = slot.with(|c| {
        c.borrow()
            .as_ref()
            .map(|arc| Arc::as_ptr(arc) as *const () as usize)
    });
    attached.unwrap_or_else(|| Arc::as_ptr(fallback) as usize)
}

/// compile.py:667 done_with_this_frame_descr_void
pub fn done_with_this_frame_descr_void_ptr() -> usize {
    descr_ref_ptr(&DONE_WITH_THIS_FRAME_DESCR_VOID, &FALLBACK_DONE_VOID)
}
/// compile.py:668 done_with_this_frame_descr_int
pub fn done_with_this_frame_descr_int_ptr() -> usize {
    descr_ref_ptr(&DONE_WITH_THIS_FRAME_DESCR_INT, &FALLBACK_DONE_INT)
}
/// compile.py:669 done_with_this_frame_descr_ref
pub fn done_with_this_frame_descr_ref_ptr() -> usize {
    descr_ref_ptr(&DONE_WITH_THIS_FRAME_DESCR_REF, &FALLBACK_DONE_REF)
}
/// compile.py:670 done_with_this_frame_descr_float
pub fn done_with_this_frame_descr_float_ptr() -> usize {
    descr_ref_ptr(&DONE_WITH_THIS_FRAME_DESCR_FLOAT, &FALLBACK_DONE_FLOAT)
}
/// compile.py:671 exit_frame_with_exception_descr_ref
pub fn exit_frame_with_exception_descr_ref_ptr() -> usize {
    EXIT_FRAME_WITH_EXCEPTION_DESCR_REF.with(|c| {
        c.borrow()
            .as_ref()
            .map(|arc| Arc::as_ptr(arc) as *const () as usize)
            .unwrap_or(0)
    })
}
/// pyjitpl.py:2283 propagate_exception_descr
pub fn propagate_exception_descr_ptr() -> usize {
    PROPAGATE_EXCEPTION_DESCR.with(|c| {
        c.borrow()
            .as_ref()
            .map(|arc| Arc::as_ptr(arc) as *const () as usize)
            .unwrap_or(0)
    })
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

/// `compile.py:658-671` `ExitFrameWithExceptionDescrRef` parity.
/// Check whether `ptr` matches the metainterp-attached
/// `exit_frame_with_exception_descr_ref` singleton.  Used by
/// `runner.rs::find_descr_by_ptr` to route a FINISH exit emitted by
/// `pyjitpl.py:3238-3245 compile_exit_frame_with_exception` into
/// `jitexc.ExitFrameWithExceptionRef` instead of `DoneWithThisFrame*`.
pub fn is_exit_frame_with_exception_descr(ptr: usize) -> bool {
    ptr != 0 && ptr == exit_frame_with_exception_descr_ref_ptr()
}

/// `warmspot.py:1022 fail_descr = cpu.get_latest_descr(deadframe)` —
/// turn a raw `jf_descr` pointer into the corresponding attached
/// `DescrRef` (metainterp-side `_DoneWithThisFrameDescr` /
/// `ExitFrameWithExceptionDescrRef` / `PropagateExceptionDescr`).
///
/// Upstream's Python `cpu.get_latest_descr` returns the actual object
/// (via GCREF → AbstractFailDescr cast); pyre's JIT-emitted code
/// stores the raw pointer of the attached `Arc` in `jf_descr`, so we
/// look it back up by Arc-identity against each metainterp-owned
/// thread-local slot. Returns `None` when `ptr` matches none of the
/// attached descrs — the caller then treats `ptr` as a direct
/// `*const DynasmFailDescr` (guard-failure path).
pub fn attached_fail_descr_by_ptr(ptr: usize) -> Option<majit_ir::DescrRef> {
    if ptr == 0 {
        return None;
    }
    let slots: [&'static std::thread::LocalKey<std::cell::RefCell<Option<majit_ir::DescrRef>>>; 6] = [
        &DONE_WITH_THIS_FRAME_DESCR_VOID,
        &DONE_WITH_THIS_FRAME_DESCR_INT,
        &DONE_WITH_THIS_FRAME_DESCR_REF,
        &DONE_WITH_THIS_FRAME_DESCR_FLOAT,
        &EXIT_FRAME_WITH_EXCEPTION_DESCR_REF,
        &PROPAGATE_EXCEPTION_DESCR,
    ];
    for slot in slots.iter() {
        if let Some(d) = slot.with(|c| {
            c.borrow()
                .as_ref()
                .filter(|arc| Arc::as_ptr(arc) as *const () as usize == ptr)
                .cloned()
        }) {
            return Some(d);
        }
    }
    None
}

impl Descr for DynasmFailDescr {}

impl FailDescr for DynasmFailDescr {
    fn fail_index(&self) -> u32 {
        self.fail_index
    }

    fn fail_arg_types(&self) -> &[Type] {
        &self.fail_arg_types
    }

    fn fail_arg_locs(&self) -> &[Option<usize>] {
        &self.fail_arg_locs
    }

    fn rd_locs(&self) -> &[u16] {
        // `llsupport/assembler.py:240-278 store_info_on_descr` populates
        // `self.rd_locs` from the regalloc-layer `fail_arg_locs`
        // (x86/assembler.rs:2541-2562 / aarch64/assembler.rs:2550-2570).
        &self.rd_locs
    }

    fn is_finish(&self) -> bool {
        self.is_finish
    }

    fn is_exit_frame_with_exception(&self) -> bool {
        self.is_exit_frame_with_exception
    }

    /// FINISH carries its one result in `fail_arg_types[0]` (or
    /// nothing for void). compile.py:626-656 parity.
    fn finish_result_type(&self) -> Type {
        self.fail_arg_types.first().copied().unwrap_or(Type::Void)
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

    fn handle_fail(&self, ctx: &mut dyn majit_ir::HandleFailContext) -> majit_ir::HandleFailResult {
        // finish → compile.py:626-656 `_DoneWithThisFrameDescr.handle_fail`;
        // else  → compile.py:701-717 `AbstractResumeGuardDescr.handle_fail`.
        majit_ir::dispatch_handle_fail(self, ctx)
    }
}
