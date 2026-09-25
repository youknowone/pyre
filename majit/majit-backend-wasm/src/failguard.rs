/// Guard failure descriptors and frame data for the wasm backend.
///
/// Simplified from CraneliftFailDescr — no bridge data, GC maps, or force tokens.
use std::cell::{Cell, RefCell};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use majit_ir::{Descr, DescrRef, FailDescr, Type};

/// Wasm-backend guard failure descriptor.
#[derive(Debug)]
pub struct WasmFailDescr {
    pub fail_index: u32,
    pub trace_id: u64,
    pub fail_arg_types: Vec<Type>,
    /// BaseAssembler.store_info_on_descr's fail locations: one physical
    /// i64 slot for each live logical resume position, None for a hole.
    /// An empty list is the identity layout used by reserved FINISH descrs
    /// and backend-only synthetic descriptors.
    pub fail_locs: Vec<Option<usize>>,
    pub is_finish: bool,
    /// Byte offset of GUARD_NOT_FORCED(_2)'s force-only spill area. Native
    /// backends carry these coordinates in their fail locations; keeping a
    /// disjoint area is essential because a following FINISH overwrites the
    /// ordinary exit slots and CALL_ASSEMBLER owns the dispatch-key slot.
    pub force_args_offset: u32,
    /// Compile-time guard gcmap retained for FINISH after a
    /// GUARD_NOT_FORCED_2, matching `assembler._finish_gcmap`.
    pub force_gcmap_ptr: usize,
    /// `history.py:125 id(descr)` parity — when the optimizer
    /// (`store_final_boxes_in_guard` / `make_and_attach_done_descrs`)
    /// stamps a metainterp `ResumeGuardDescr` / `DoneWithThisFrame*` /
    /// `ExitFrameWithExceptionDescrRef` / `PropagateExceptionDescr` on
    /// `op.descr`, we keep it here so `get_latest_descr_arc` returns the
    /// canonical metainterp Arc (matching dynasm/cranelift).  `None`
    /// for synthetic backend-only descrs (`compile_bridge` placeholders,
    /// test scaffolds).
    pub meta_descr: Option<DescrRef>,
}

impl WasmFailDescr {
    pub fn frame_slot(&self, index: usize) -> Option<usize> {
        if self.fail_locs.is_empty() {
            return (index < self.fail_arg_types.len()).then_some(index);
        }
        self.fail_locs.get(index).copied().flatten()
    }
}

impl Descr for WasmFailDescr {
    fn index(&self) -> u32 {
        self.fail_index
    }

    fn as_fail_descr(&self) -> Option<&dyn FailDescr> {
        Some(self)
    }
}

impl FailDescr for WasmFailDescr {
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

/// Where this deadframe reads the jitframe.
///
/// A GC frame is re-read through [`OwnerRootGuard`] so a moving collection
/// cannot leave a raw pointer at the old copy. A host buffer is not a GC
/// object and does not move; its address stays in [`LiveFrame::Fixed`] and
/// [`WasmFrameData::host_frame`] owns the bytes.
enum LiveFrame {
    Rooted(majit_gc::shadow_stack::OwnerRootGuard),
    Fixed(*mut majit_backend::jitframe::JitFrame),
}

/// Wasm-backend dead frame data.
///
/// Stored inside `DeadFrame::Boxed` after `execute_token` returns. The
/// deadframe is the jitframe (`llmodel.py` `return ll_frame`): accessors
/// read `jf_frame` in place. `DeadFrame::JitFrame` is not used —
/// `FailArgSource::from_jitframe` decodes `rd_locs` as identity slots, while
/// this backend spills compactly.
///
/// One root: the frame. `jitframe_trace` walks `jf_savedata`, `jf_guard_exc`,
/// `jf_forward`, and the fail-arg Ref slots named by the guard's `jf_gcmap`.
/// [`Self::boxed`] is the unit-test snapshot that has no jitframe.
pub struct WasmFrameData {
    /// [`Self::boxed`] only: values with no jitframe. Empty when `frame` is set.
    pub raw_values: Vec<i64>,
    pub fail_descr: Arc<WasmFailDescr>,
    /// [`Self::boxed`] only. A live frame reads `jf_guard_exc`
    /// (`llmodel.py` `grab_exc_value`).
    pub exc_value: i64,
    /// [`Self::boxed`] only. A live frame reads `jf_savedata`.
    pub savedata: i64,
    frame: Option<LiveFrame>,
    /// `force()` snapshot: fail args live at `force_args_offset`, tagged.
    read_force: bool,
    /// Off-GC host-buffer owner. Keeps the entry JitFrame alive after
    /// `execute_token` returns. The wasm32 host-buffer path.
    #[allow(dead_code)]
    host_frame: Option<majit_backend::libc_deadframe::LibcJitFrameDeadFrame>,
}

impl WasmFrameData {
    /// Unit-test snapshot with no jitframe. Production exits use
    /// [`Self::from_live_frame`].
    pub fn boxed(
        raw_values: Vec<i64>,
        fail_descr: Arc<WasmFailDescr>,
        exc_value: i64,
    ) -> Box<Self> {
        Box::new(WasmFrameData {
            raw_values,
            fail_descr,
            exc_value,
            savedata: 0,
            frame: None,
            read_force: false,
            host_frame: None,
        })
    }

    /// The jitframe `execute_token` / `force` returned. Fail args stay in
    /// its `jf_frame` slots. The frame is the only root: `jf_gcmap` names
    /// the exit's Ref slots and `jitframe_trace` names the header fields.
    ///
    /// `gc_root` takes an [`OwnerRootGuard`]. A force snapshot of a frame
    /// the running call already rooted takes one too — dropping it releases
    /// only this handle, and the call's shadow-stack root stays. A host
    /// buffer (`gc_root == false`) does not move; `host_frame` owns it when
    /// this deadframe does.
    pub fn from_live_frame(
        jf: *mut majit_backend::jitframe::JitFrame,
        fail_descr: Arc<WasmFailDescr>,
        read_force: bool,
        gc_root: bool,
        host_frame: Option<majit_backend::libc_deadframe::LibcJitFrameDeadFrame>,
    ) -> Box<Self> {
        let frame = if gc_root {
            LiveFrame::Rooted(majit_gc::shadow_stack::OwnerRootGuard::new(
                majit_ir::GcRef(jf as usize),
            ))
        } else {
            LiveFrame::Fixed(jf)
        };
        Box::new(WasmFrameData {
            raw_values: Vec::new(),
            fail_descr,
            exc_value: 0,
            savedata: 0,
            frame: Some(frame),
            read_force,
            host_frame,
        })
    }

    fn frame_ptr(&self) -> Option<*mut majit_backend::jitframe::JitFrame> {
        match self.frame.as_ref() {
            Some(LiveFrame::Rooted(root)) => {
                Some(root.get().0 as *mut majit_backend::jitframe::JitFrame)
            }
            Some(LiveFrame::Fixed(jf)) => Some(*jf),
            None => None,
        }
    }

    /// Current items base. A GC root is re-read so a moving collection
    /// cannot leave this pointing at the old frame.
    pub fn items_base(&self) -> Option<usize> {
        self.frame_ptr()
            .map(|jf| jf as usize + majit_backend::jitframe::FIRST_ITEM_OFFSET)
    }

    pub fn read_force(&self) -> bool {
        self.read_force
    }

    /// `cpu.set_savedata_ref(deadframe, data)` — write `jf_savedata`.
    /// `jitframe_trace` traces that header field; the frame root is enough.
    pub fn set_savedata(&mut self, data: majit_ir::GcRef) {
        let mut data_slot = data.0 as i64;
        if let Some(jf) = self.frame_ptr() {
            let depth = majit_gc::shadow_stack::resume_ref_roots_depth();
            unsafe {
                majit_gc::shadow_stack::push_resume_ref_roots(std::slice::from_mut(&mut data_slot));
            }
            if crate::wasm_gc_owns_object(jf as usize) {
                crate::wasm_active_gc_write_barrier(majit_ir::GcRef(jf as usize));
            }
            majit_gc::shadow_stack::pop_resume_ref_roots_to(depth);
            unsafe { (*jf).jf_savedata = data_slot as usize };
        }
        self.savedata = data_slot;
    }

    #[allow(dead_code)] // wasm32 `execute_token` host-buffer path
    pub(crate) fn take_host_frame(
        &mut self,
        frame: majit_backend::libc_deadframe::LibcJitFrameDeadFrame,
    ) {
        self.host_frame = Some(frame);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use majit_gc::GcAllocator;
    use majit_ir::GcRef;

    use super::{Type, WasmFailDescr, WasmFrameData};

    struct RootCountingGc(Arc<AtomicUsize>);

    impl GcAllocator for RootCountingGc {
        fn alloc_nursery(&mut self, _size: usize) -> GcRef {
            GcRef(0)
        }

        fn alloc_nursery_no_collect(&mut self, _size: usize) -> GcRef {
            GcRef(0)
        }

        fn alloc_varsize(&mut self, _base_size: usize, _item_size: usize, _length: usize) -> GcRef {
            GcRef(0)
        }

        fn alloc_varsize_no_collect(
            &mut self,
            _base_size: usize,
            _item_size: usize,
            _length: usize,
        ) -> GcRef {
            GcRef(0)
        }

        fn write_barrier(&mut self, _obj: GcRef) {}

        fn jit_remember_young_pointer_from_array(&mut self, _obj: GcRef) {}

        fn remember_young_pointer_from_array2(
            &mut self,
            _obj: GcRef,
            _index: usize,
            _card_page_shift: u32,
        ) {
        }

        fn collect_nursery(&mut self) {}

        fn collect_full(&mut self) {}

        fn nursery_free(&self) -> *mut u8 {
            std::ptr::null_mut()
        }

        fn nursery_free_addr(&self) -> usize {
            0
        }

        fn nursery_top(&self) -> *const u8 {
            std::ptr::null()
        }

        fn nursery_top_addr(&self) -> usize {
            0
        }

        fn max_nursery_object_size(&self) -> usize {
            0
        }

        unsafe fn add_root(&mut self, _root: *mut GcRef) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }

        fn remove_root(&mut self, _root: *mut GcRef) {
            self.0.fetch_sub(1, Ordering::SeqCst);
        }
    }

    fn install_root_counting_gc() -> (Arc<AtomicUsize>, crate::ActiveGcBox) {
        let roots = Arc::new(AtomicUsize::new(0));
        let gc_box = crate::install_gc_box(Box::new(RootCountingGc(Arc::clone(&roots))));
        (roots, gc_box)
    }

    fn fail_descr(fail_arg_types: Vec<Type>) -> Arc<WasmFailDescr> {
        Arc::new(WasmFailDescr {
            fail_index: 0,
            trace_id: 0,
            fail_arg_types,
            fail_locs: Vec::new(),
            is_finish: false,
            force_args_offset: 8,
            force_gcmap_ptr: 0,
            meta_descr: None,
        })
    }

    #[test]
    fn a_finish_singleton_resolves_to_its_reserved_exit() {
        let _serialized = super::lock_cpu();
        // The emitted FINISH writes the index this returns and the emitted
        // CALL_ASSEMBLER check compares against the same constant, so a
        // singleton that failed to bind would send every clean callee finish
        // back to the host to be decoded and handed straight over.
        let descr: majit_ir::DescrRef = Arc::new(majit_backend::DoneWithThisFrameDescrRef::new());
        super::attach_finish_descr(super::FINISH_EXIT_INDEX_REF, Arc::clone(&descr));
        assert_eq!(
            super::attached_finish_exit_index(&Some(Arc::clone(&descr))),
            Some(super::FINISH_EXIT_INDEX_REF),
        );
        assert_eq!(
            super::done_with_this_frame_exit_index(Type::Ref),
            super::FINISH_EXIT_INDEX_REF,
        );
        // A descr this cpu was never handed has no shared identity, so it keeps
        // its own exit.
        let unattached: majit_ir::DescrRef =
            Arc::new(majit_backend::DoneWithThisFrameDescrRef::new());
        assert_eq!(super::attached_finish_exit_index(&Some(unattached)), None);
    }

    #[test]
    fn finish_cells_keep_a_stable_address() {
        // CALL_ASSEMBLER compares `jf_descr` with the address baked at
        // compile time. Rebinding the singleton must not move that address.
        let _serialized = super::lock_cpu();
        let ptr = super::finish_descr_ptr(super::FINISH_EXIT_INDEX_REF);
        let again = super::finish_descr_ptr(super::FINISH_EXIT_INDEX_REF);
        assert_eq!(ptr, again);
        assert_ne!(ptr, super::finish_descr_ptr(super::FINISH_EXIT_INDEX_INT));
        let descr = super::descr_at(ptr).expect("finish cell");
        assert!(descr.is_finish);
        assert_eq!(descr.fail_arg_types, vec![Type::Ref]);
    }

    #[test]
    fn boxed_does_not_root_interior_slots() {
        let _serialized = super::lock_cpu();
        let (roots, _gc_box) = install_root_counting_gc();
        let before = roots.load(Ordering::SeqCst);
        let frame = WasmFrameData::boxed(
            vec![0x10, 42, 0, 0x20],
            fail_descr(vec![Type::Ref, Type::Int, Type::Float, Type::Ref]),
            0x30,
        );
        assert_eq!(roots.load(Ordering::SeqCst), before);
        drop(frame);
        assert_eq!(roots.load(Ordering::SeqCst), before);
    }

    #[test]
    fn boxed_without_refs_or_exception_does_not_bracket_roots() {
        let _serialized = super::lock_cpu();
        let (roots, _gc_box) = install_root_counting_gc();
        let before = roots.load(Ordering::SeqCst);
        let frame = WasmFrameData::boxed(vec![1, 2], fail_descr(vec![Type::Int, Type::Float]), 0);
        assert_eq!(roots.load(Ordering::SeqCst), before);
        drop(frame);
        assert_eq!(roots.load(Ordering::SeqCst), before);
    }

    #[test]
    fn set_savedata_writes_the_word_without_an_interior_root() {
        let _serialized = super::lock_cpu();
        let (roots, _gc_box) = install_root_counting_gc();
        let before = roots.load(Ordering::SeqCst);
        let mut frame = WasmFrameData::boxed(vec![1], fail_descr(vec![Type::Int]), 0);
        frame.set_savedata(GcRef(0x40));
        assert_eq!(frame.savedata, 0x40);
        frame.set_savedata(GcRef(0));
        assert_eq!(frame.savedata, 0);
        assert_eq!(roots.load(Ordering::SeqCst), before);
        drop(frame);
        assert_eq!(roots.load(Ordering::SeqCst), before);
    }

    #[test]
    fn set_savedata_on_a_force_snapshot_writes_the_live_jitframe() {
        let _serialized = super::lock_cpu();
        use majit_backend::jitframe::{JitFrame, alloc_off_gc_jitframe, free_off_gc_jitframe};

        // A leftover MiniMark box would treat `GcRef(0x51)` as a heap
        // pointer and rewrite `jf_savedata` when the snapshot's root is
        // dropped.
        crate::clear_gc_allocator();
        let jf = alloc_off_gc_jitframe(JitFrame::alloc_size(4));
        let mut frame =
            WasmFrameData::from_live_frame(jf, fail_descr(vec![Type::Int]), true, false, None);
        // `NO_CONCRETE` is not a heap object (even, non-8-aligned, non-null).
        // A low dummy (0x51) was chased as a nursery pointer when another
        // test's MiniMark was still the process hook target.
        let saved = GcRef::NO_CONCRETE;
        frame.set_savedata(saved);
        unsafe {
            assert_eq!((*jf).jf_savedata, saved.0);
        }
        drop(frame);
        unsafe {
            assert_eq!((*jf).jf_savedata, saved.0);
            free_off_gc_jitframe(jf);
        }
    }

    fn dummy_label_target(func_handle: u32) -> super::LabelTarget {
        super::LabelTarget {
            func_handle,
            wide_slot: 0,
            key: 0,
            num_args: 0,
            resume_safe: true,
            requires_own_frame: false,
            is_last_label: true,
            frame: crate::codegen::FrameGeometry::fixed(),
            owner_token: 0,
        }
    }

    #[test]
    fn retract_label_target_keeps_a_replacement_handle() {
        let _serialized = super::lock_cpu();
        let id = 0x7e71_ac10_usize;
        super::publish_label_target(id, dummy_label_target(7));
        super::retract_label_target_if_handle(id, 7);
        assert!(super::label_target(id).is_none());

        super::publish_label_target(id, dummy_label_target(9));
        super::retract_label_target_if_handle(id, 7);
        assert_eq!(super::label_target(id).map(|t| t.func_handle), Some(9));
        super::retract_label_target_if_handle(id, 9);
        assert!(super::label_target(id).is_none());
    }
}

/// A resumable `LABEL` of a compiled loop, published in `LABEL_TARGETS` so a
/// loop-closing bridge can chain into ANY compiled loop's label in-module
/// (jump-to-existing-trace), not only its own source loop's. Keyed by the
/// label's loop-target descr identity (`Arc::as_ptr`), which the JUMP shares.
#[derive(Clone, Copy, Debug)]
pub struct LabelTarget {
    /// Table slot of the owning loop's compiled function.
    pub func_handle: u32,
    /// Table slot of the fixed-arity label-parameter entry, or 0 when absent.
    pub wide_slot: u32,
    /// Resume dispatch key (`label ordinal + 1`) the bridge's JUMP writes.
    pub key: u32,
    /// The label's arg count — the resume loader reads exactly this many
    /// positional frame slots, so the JUMP arity must equal it.
    pub num_args: usize,
    /// Whether every live-in can be reconstructed from LABEL args or frozen
    /// backend capture slots (`codegen::label_resume_info`).
    pub resume_safe: bool,
    /// Backend capture slots are populated by this target loop's own
    /// fall-through path. If true, a bridge may resume only its source loop,
    /// not a sibling specialization that happens to share the geometry.
    pub requires_own_frame: bool,
    /// Whether this is the owning loop's LAST label (the loop header). A
    /// bridge landing here re-runs no segment code before the `loop`, so the
    /// livelock advance-check applies; earlier labels execute the peeled
    /// segment, which advances the state by itself.
    pub is_last_label: bool,
    /// Frozen frame geometry of the target token. A tail-call can only reuse
    /// a frame when its offsets agree exactly, not merely when its allocation
    /// is large enough.
    pub frame: crate::codegen::FrameGeometry,
    /// `JitCellToken.number` that published this row.
    pub owner_token: u64,
}

/// Frozen metadata for entering a compiled loop from a `CALL_ASSEMBLER` arm.
/// The table slot and frame layout are published only after the loop module is
/// installed, so a caller can decline before baking an unresolved target.
#[derive(Clone, Debug)]
pub struct CallAssemblerTarget {
    /// Owning `JitCellToken` number. Used only by the dormant wasm regression
    /// hook to select one target deterministically.
    pub token_number: u64,
    pub func_handle: u32,
    pub input_types: Vec<Type>,
    /// Byte offset at which the target reads its fresh-entry dispatch key.
    /// Redirects may replace a temporary callback with a differently laid-out
    /// real loop, so this travels with the target's other runtime metadata.
    pub dispatch_key_ofs: u64,
    pub callee_frame_bytes: u32,
    pub callee_gcmap_ptr: i64,
    pub compiled_ptr: u64,
    /// Callee Ref-home origin, in item-base bytes. Redirects may replace a
    /// temporary callback with a differently laid-out loop, so this is
    /// loaded with the other runtime snapshot fields.
    pub home_slot_base: u32,
    pub home_slots: u32,
    /// Callee retains `_finish_gcmap` (`GUARD_NOT_FORCED_2`). The caller
    /// pop footer must use the write-barrier helper when this is set,
    /// even if the caller module itself has no GNF2.
    pub has_guard_not_forced_2: u32,
    /// Homes `build_home_gcmap` marks: the used ordinary prefix, then the
    /// LABEL-capture tail. Reserved padding between them is unmarked.
    pub marked_ordinary: u32,
    pub marked_labels: u32,
    pub label_ref_slots: u32,
}

/// Compiled loop targets keyed by their `JitCellToken` number. Unlike label
/// targets, CALL_ASSEMBLER identifies its callee by that number directly.
pub static CALL_ASSEMBLER_TARGETS: parking_lot::Mutex<
    Option<std::collections::HashMap<u64, CallAssemblerTarget>>,
> = parking_lot::Mutex::new(None);

// ── CALL_ASSEMBLER dispatch table ──
//
// A trace module imports the guest's linear memory, so a boxed wasm-side
// allocation is addressable by every trace with an ordinary i32.load.  The
// box keeps that address stable while the map grows; the emitted code must
// never bake a table slot because redirects and the pending->real transition
// replace it after the caller module was compiled.
#[repr(C)]
pub struct WasmCaRuntimeTarget {
    /// `__indirect_function_table` slot. Zero means pending/unavailable.
    pub func_handle: u32,
    /// `CompiledWasmLoop` address for the deopt helper, in wasm32 memory.
    pub compiled_ptr: u32,
    /// Current callee JitFrame item-region size.  This is redirectable state:
    /// `CompiledLoopToken.update_frame_info` permits the replacement loop to
    /// need a deeper frame than the temporary callback.
    pub callee_frame_bytes: u32,
    /// Byte offset where this target reads its fresh-entry dispatch key.
    /// Redirects may replace a temporary callback with a differently sized
    /// real loop, so callers load this together with the frame allocation.
    pub dispatch_key_ofs: u32,
    /// Current callee GC map.  It must change together with frame depth when
    /// `redirect_call_assembler` installs the real loop.
    pub callee_gcmap_ptr: i64,
    pub home_slot_base: u32,
    pub home_slots: u32,
    pub has_guard_not_forced_2: u32,
    /// Used ordinary Ref homes the callee's static gcmap marks.
    pub marked_ordinary: u32,
    /// Used LABEL-capture homes the same map marks.
    pub marked_labels: u32,
    /// Reserved LABEL tail. The marked tail starts at
    /// `home_slots - label_ref_slots`.
    pub label_ref_slots: u32,
}

/// Stable cell baked by callers.  A redirect publishes one pointer to an
/// immutable target snapshot, so generated code cannot combine the new loop's
/// function with the temporary callback's frame geometry under concurrent
/// execution.  Old snapshots stay owned by the entry for as long as any caller
/// can still hold the pointer it loaded.
///
/// `has_guard_not_forced_2` is not snapshot state: an out-of-line
/// `GUARD_NOT_FORCED_2` bridge can attach while a CALL_ASSEMBLER is
/// already inside the callee. The in-flight footer still holds the
/// pre-call snapshot, so this flag lives on the cell and only goes
/// 0 → 1.
#[repr(C)]
pub struct WasmCaDispatchEntry {
    pub target_ptr: AtomicU32,
    pub has_guard_not_forced_2: AtomicU32,
    pub targets: std::sync::Mutex<Vec<Box<WasmCaRuntimeTarget>>>,
}

pub const WASM_CA_DISPATCH_TARGET_PTR_OFS: u64 =
    std::mem::offset_of!(WasmCaDispatchEntry, target_ptr) as u64;
pub const WASM_CA_DISPATCH_HAS_GNF2_OFS: u64 =
    std::mem::offset_of!(WasmCaDispatchEntry, has_guard_not_forced_2) as u64;
pub const WASM_CA_TARGET_FUNC_HANDLE_OFS: u64 =
    std::mem::offset_of!(WasmCaRuntimeTarget, func_handle) as u64;
pub const WASM_CA_TARGET_COMPILED_PTR_OFS: u64 =
    std::mem::offset_of!(WasmCaRuntimeTarget, compiled_ptr) as u64;
pub const WASM_CA_TARGET_FRAME_BYTES_OFS: u64 =
    std::mem::offset_of!(WasmCaRuntimeTarget, callee_frame_bytes) as u64;
pub const WASM_CA_TARGET_DISPATCH_KEY_OFS_OFS: u64 =
    std::mem::offset_of!(WasmCaRuntimeTarget, dispatch_key_ofs) as u64;
pub const WASM_CA_TARGET_GCMAP_PTR_OFS: u64 =
    std::mem::offset_of!(WasmCaRuntimeTarget, callee_gcmap_ptr) as u64;
pub const WASM_CA_TARGET_HOME_SLOT_BASE_OFS: u64 =
    std::mem::offset_of!(WasmCaRuntimeTarget, home_slot_base) as u64;
pub const WASM_CA_TARGET_HOME_SLOTS_OFS: u64 =
    std::mem::offset_of!(WasmCaRuntimeTarget, home_slots) as u64;
pub const WASM_CA_TARGET_HAS_GNF2_OFS: u64 =
    std::mem::offset_of!(WasmCaRuntimeTarget, has_guard_not_forced_2) as u64;
pub const WASM_CA_TARGET_MARKED_ORDINARY_OFS: u64 =
    std::mem::offset_of!(WasmCaRuntimeTarget, marked_ordinary) as u64;
pub const WASM_CA_TARGET_MARKED_LABELS_OFS: u64 =
    std::mem::offset_of!(WasmCaRuntimeTarget, marked_labels) as u64;
pub const WASM_CA_TARGET_LABEL_REF_SLOTS_OFS: u64 =
    std::mem::offset_of!(WasmCaRuntimeTarget, label_ref_slots) as u64;

/// `make_and_attach_done_descrs` gives every cpu one `DoneWithThisFrame*` per
/// result kind plus one `ExitFrameWithExceptionDescrRef`, and
/// `compile_done_with_this_frame` / `compile_exit_frame_with_exception` stamp
/// that singleton on the FINISH — so every trace that finishes the same way
/// writes the same `jf_descr`, and `_call_assembler_check_descr` recognises a
/// clean callee finish by comparing against one value.
///
/// The shared identity is the cell address stored in `jf_descr`. The five
/// cells never move; `attach_finish_descr` replaces the `Arc` inside the
/// cell. `get_latest_descr` reads that `Arc`'s `meta_descr`.
pub const FINISH_EXIT_INDEX_VOID: u32 = 0;
pub const FINISH_EXIT_INDEX_INT: u32 = 1;
pub const FINISH_EXIT_INDEX_REF: u32 = 2;
pub const FINISH_EXIT_INDEX_FLOAT: u32 = 3;
pub const FINISH_EXIT_INDEX_EXC: u32 = 4;

/// Reserved exit index for the `done_with_this_frame_descr_*` of `ty`.
pub fn done_with_this_frame_exit_index(ty: Type) -> u32 {
    match ty {
        Type::Void => FINISH_EXIT_INDEX_VOID,
        Type::Int => FINISH_EXIT_INDEX_INT,
        Type::Ref => FINISH_EXIT_INDEX_REF,
        Type::Float => FINISH_EXIT_INDEX_FLOAT,
    }
}

/// The four `DoneWithThisFrameDescr*` classes carry the one result of their
/// kind; `ExitFrameWithExceptionDescrRef` carries the exception Ref.
fn reserved_fail_arg_types(exit_index: u32) -> Vec<Type> {
    match exit_index {
        FINISH_EXIT_INDEX_VOID => Vec::new(),
        FINISH_EXIT_INDEX_INT => vec![Type::Int],
        FINISH_EXIT_INDEX_FLOAT => vec![Type::Float],
        _ => vec![Type::Ref],
    }
}

fn reserved_finish_descr(exit_index: u32, meta_descr: Option<DescrRef>) -> Arc<WasmFailDescr> {
    Arc::new(WasmFailDescr {
        fail_index: exit_index,
        trace_id: 0,
        fail_arg_types: reserved_fail_arg_types(exit_index),
        fail_locs: Vec::new(),
        is_finish: true,
        force_args_offset: 0,
        force_gcmap_ptr: 0,
        meta_descr,
    })
}

/// Address baked into `jf_descr` / `jf_force_descr`.
///
/// The cell is the unit `LoopAsmResources` (or the finish singleton array)
/// keeps alive. `get_latest_descr` casts `jf_descr` back to this cell.
pub struct FailDescrCell {
    descr: parking_lot::Mutex<Arc<WasmFailDescr>>,
}

impl FailDescrCell {
    pub fn new(descr: Arc<WasmFailDescr>) -> Self {
        Self {
            descr: parking_lot::Mutex::new(descr),
        }
    }

    pub fn get(&self) -> Arc<WasmFailDescr> {
        Arc::clone(&self.descr.lock())
    }

    pub fn set(&self, descr: Arc<WasmFailDescr>) {
        *self.descr.lock() = descr;
    }
}

/// Read the descr an exit stored in `jf_descr` or `jf_force_descr`.
pub fn descr_at(cell: usize) -> Option<Arc<WasmFailDescr>> {
    if cell == 0 {
        return None;
    }
    Some(unsafe { &*(cell as *const FailDescrCell) }.get())
}

/// Publish the real descr into a cell `build_wasm_module` allocated.
pub fn fill_exit_cell(cell: usize, descr: Arc<WasmFailDescr>) {
    if cell == 0 {
        return;
    }
    unsafe { &*(cell as *const FailDescrCell) }.set(descr);
}

/// Allocate the cell whose address the exit stores in `jf_descr`.
///
/// `sink` is the `LoopAsmResources` the compile will push into
/// `asmmemmgr_blocks`. A null sink (a direct `build_wasm_module` test)
/// leaks the cell the same way `park_gcmap_raw` leaks a map.
pub fn alloc_exit_cell(sink: usize, fail_index: u32) -> usize {
    let descr = Arc::new(WasmFailDescr {
        fail_index,
        trace_id: 0,
        fail_arg_types: Vec::new(),
        fail_locs: Vec::new(),
        is_finish: false,
        force_args_offset: 0,
        force_gcmap_ptr: 0,
        meta_descr: None,
    });
    if sink == 0 {
        let cell = Box::new(FailDescrCell::new(descr));
        let ptr = &*cell as *const FailDescrCell as usize;
        Box::leak(cell);
        return ptr;
    }
    let resources = unsafe { &mut *(sink as *mut crate::release::LoopAsmResources) };
    resources.alloc_fail_cell(descr)
}

/// CPU singletons for the five `done_with_this_frame` / exception exits.
///
/// The `Box` is allocated once and never replaced, so the address baked
/// into a module stays valid when `attach_finish_descr` rebinds the `Arc`.
static FINISH_EXITS: parking_lot::Mutex<[Option<Box<FailDescrCell>>; 5]> =
    parking_lot::Mutex::new([None, None, None, None, None]);

fn finish_descr_ptr_locked(exits: &mut [Option<Box<FailDescrCell>>; 5], index: u32) -> usize {
    let slot = &mut exits[index as usize];
    if slot.is_none() {
        *slot = Some(Box::new(FailDescrCell::new(reserved_finish_descr(
            index, None,
        ))));
    }
    &**slot.as_ref().expect("finish cell") as *const FailDescrCell as usize
}

/// Stable `jf_descr` immediate for one reserved finish exit.
pub fn finish_descr_ptr(index: u32) -> usize {
    let mut exits = FINISH_EXITS.lock();
    finish_descr_ptr_locked(&mut exits, index)
}

fn finish_exit(index: u32) -> Arc<WasmFailDescr> {
    descr_at(finish_descr_ptr(index)).expect("reserved finish exit is uninitialized")
}

fn thin_descr_ptr(descr: &DescrRef) -> usize {
    Arc::as_ptr(descr) as *const () as usize
}

/// `make_and_attach_done_descrs` pointer identity: the reserved exit index for
/// `descr` when it is one of the five singletons this cpu was handed, else
/// `None`.
/// Mirrors `AttachedDescrPtrs::is_done_with_this_frame_descr`, which is how the
/// native backends recognise the same descrs on the FINISH fast path.
///
/// Answered from the reserved entries themselves, so an index this returns
/// always names a registry entry carrying `descr` — `get_latest_descr_arc`
/// keeps its `AbstractDescr` identity whatever order attachment and the first
/// compile happened in.
pub fn attached_finish_exit_index(descr: &Option<DescrRef>) -> Option<u32> {
    let ptr = thin_descr_ptr(descr.as_ref()?);
    let exits = FINISH_EXITS.lock();
    exits.iter().enumerate().find_map(|(index, reserved)| {
        let cell = reserved.as_ref()?;
        cell.get()
            .meta_descr
            .as_ref()
            .is_some_and(|attached| thin_descr_ptr(attached) == ptr)
            .then_some(index as u32)
    })
}

/// `make_and_attach_done_descrs`' per-target attachment for one of the five.
/// Rebinds the `Arc` inside the existing cell, so a module that already
/// baked [`finish_descr_ptr`] still names this descr.
pub fn attach_finish_descr(exit_index: u32, descr: DescrRef) {
    let cell = finish_descr_ptr(exit_index);
    fill_exit_cell(cell, reserved_finish_descr(exit_index, Some(descr)));
}

/// Whether the cpu has been handed `exit_frame_with_exception_descr_ref`.
///
/// The memory-error check the allocation codegen emits leaves through
/// [`FINISH_EXIT_INDEX_EXC`], and only the attached metainterp descr carries
/// `is_exit_frame_with_exception`. The reserved entry on its own reads as a
/// plain finish, which would hand the raised value back as the loop's result
/// instead of raising it, so the emitter has to know which of the two it has.
pub fn exit_frame_with_exception_attached() -> bool {
    finish_exit(FINISH_EXIT_INDEX_EXC).meta_descr.is_some()
}

/// `pyjitpl.py` `self.cpu.propagate_exception_descr = exc_descr`.
///
/// Dynasm and cranelift compare `Arc::as_ptr` of this singleton, and of the
/// `FailDescrCell` a `GUARD_NO_EXCEPTION` recovery stub writes, against
/// `jf_descr`. A wasm frame stores an exit index; `get_latest_descr_arc`
/// recovers this Arc from `WasmFailDescr.meta_descr`, so the same identity
/// compare is `Arc::ptr_eq`.
static PROPAGATE_EXCEPTION_DESCR: parking_lot::Mutex<Option<DescrRef>> =
    parking_lot::Mutex::new(None);

pub fn attach_propagate_exception_descr(descr: DescrRef) {
    *PROPAGATE_EXCEPTION_DESCR.lock() = Some(descr);
}

pub fn is_propagate_exception_descr(descr: &DescrRef) -> bool {
    PROPAGATE_EXCEPTION_DESCR
        .lock()
        .as_ref()
        .is_some_and(|propagate| Arc::ptr_eq(descr, propagate))
}

/// `compile.py` `PropagateExceptionDescr.handle_fail` for the host's
/// outermost exit reader.
///
/// That descr's `fail_index` is `u32::MAX` and `is_finish` is false, so the
/// reader treats the exit as a loop-back JUMP and drops the exception.
/// When `fail_descr` carries the propagate singleton, return the attached
/// `exit_frame_with_exception_descr_ref` and the grabbed value (or
/// `memory_error` when the cell is empty) so the finish reader raises
/// `ExitFrameWithExceptionRef` from slot 0. `None` when this is not that
/// exit, when the exception descr was never attached — a bare finish would
/// hand the object back as the loop result — or when neither cell holds one.
pub fn stage_propagate_exception_exit(
    fail_descr: &WasmFailDescr,
    exc_value: i64,
) -> Option<(Arc<WasmFailDescr>, i64)> {
    let meta = fail_descr.meta_descr.as_ref()?;
    if !is_propagate_exception_descr(meta) || !exit_frame_with_exception_attached() {
        return None;
    }
    let exc = if exc_value != 0 {
        exc_value
    } else {
        majit_backend::memory_error_singleton_ref()
    };
    if exc == 0 {
        return None;
    }
    Some((finish_exit(FINISH_EXIT_INDEX_EXC), exc))
}

/// Stable, guest-memory dispatch entries, keyed by CALL_ASSEMBLER token.
/// `Box` is intentional: an emitted module bakes the entry address.
pub static WASM_CA_DISPATCH: parking_lot::Mutex<
    Option<std::collections::HashMap<u64, Box<WasmCaDispatchEntry>>>,
> = parking_lot::Mutex::new(None);

/// Compiled loops that already have a GNF2 bridge. Consulted under
/// `WASM_CA_DISPATCH` so a redirect that publishes after `mark` but
/// before the source guard cell is written still raises the new alias
/// cell. Pointers are forgotten when that compiled loop is removed.
static CA_GNF2_COMPILED_PTRS: parking_lot::Mutex<Option<std::collections::HashSet<u32>>> =
    parking_lot::Mutex::new(None);

/// Return the stable guest-memory address for `number`, creating a pending
/// (zero-slot) entry when needed.
pub fn ca_dispatch_slot(number: u64) -> u32 {
    let mut table = WASM_CA_DISPATCH.lock();
    let entry = table
        .get_or_insert_with(Default::default)
        .entry(number)
        .or_insert_with(|| {
            Box::new(WasmCaDispatchEntry {
                target_ptr: AtomicU32::new(0),
                has_guard_not_forced_2: AtomicU32::new(0),
                targets: std::sync::Mutex::new(Vec::new()),
            })
        });
    (&**entry as *const WasmCaDispatchEntry as usize) as u32
}

/// Raise the monotonic GNF2 flag on an existing dispatch cell.
///
/// Does not create an entry and does not publish a new snapshot. Call
/// this before arming a newly compiled `GUARD_NOT_FORCED_2` bridge so
/// an in-flight CALL_ASSEMBLER footer sees the flag before the callee
/// can finish through that bridge.
///
/// When `number` already has a snapshot, every cell that still retains
/// that compiled loop — current target or an older snapshot — is
/// raised too. Redirected aliases keep their own cell, and in-flight
/// callers may still hold a historical snapshot pointer, so marking
/// only the replacement token or only `.last()` would leave those
/// footers reading zero.
pub fn ca_dispatch_mark_gnf2(number: u64) {
    let table = WASM_CA_DISPATCH.lock();
    let Some(table) = table.as_ref() else {
        return;
    };
    let compiled_ptr = table.get(&number).and_then(|entry| {
        entry
            .targets
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .last()
            .map(|target| target.compiled_ptr)
    });
    if let Some(compiled_ptr) = compiled_ptr.filter(|&ptr| ptr != 0) {
        mark_gnf2_entries_for_compiled_ptr(table, compiled_ptr);
    } else if let Some(entry) = table.get(&number) {
        entry.has_guard_not_forced_2.store(1, Ordering::Release);
    }
}

/// Raise the monotonic GNF2 flag on every dispatch cell that still
/// retains a snapshot invoking `compiled_ptr`.
pub fn ca_dispatch_mark_gnf2_for_compiled_ptr(compiled_ptr: u32) {
    if compiled_ptr == 0 {
        return;
    }
    let table = WASM_CA_DISPATCH.lock();
    remember_gnf2_compiled_ptr(compiled_ptr);
    if let Some(table) = table.as_ref() {
        mark_gnf2_entries_for_compiled_ptr(table, compiled_ptr);
    }
}

fn remember_gnf2_compiled_ptr(compiled_ptr: u32) {
    if compiled_ptr != 0 {
        CA_GNF2_COMPILED_PTRS
            .lock()
            .get_or_insert_with(Default::default)
            .insert(compiled_ptr);
    }
}

fn compiled_ptr_has_gnf2(compiled_ptr: u32) -> bool {
    compiled_ptr != 0
        && CA_GNF2_COMPILED_PTRS
            .lock()
            .as_ref()
            .is_some_and(|set| set.contains(&compiled_ptr))
}

fn mark_gnf2_entries_for_compiled_ptr(
    table: &std::collections::HashMap<u64, Box<WasmCaDispatchEntry>>,
    compiled_ptr: u32,
) {
    remember_gnf2_compiled_ptr(compiled_ptr);
    for entry in table.values() {
        let aliases = entry
            .targets
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .any(|target| target.compiled_ptr == compiled_ptr);
        if aliases {
            entry.has_guard_not_forced_2.store(1, Ordering::Release);
        }
    }
}

/// Stamp `has_guard_not_forced_2` on every CALL_ASSEMBLER metadata
/// alias that currently names `compiled_ptr`.
pub fn mark_call_assembler_targets_gnf2_for_compiled_ptr(compiled_ptr: u32) {
    if let Some(targets) = CALL_ASSEMBLER_TARGETS.lock().as_mut() {
        for target in targets.values_mut() {
            if target.compiled_ptr as u32 == compiled_ptr {
                target.has_guard_not_forced_2 = 1;
            }
        }
    }
}

/// Publish an installed loop after its module has acquired a shared-table
/// slot.  All runtime fields live in one immutable snapshot, and the release
/// store publishes its address only after the snapshot is fully initialized.
pub fn ca_dispatch_publish(
    number: u64,
    func_handle: u32,
    compiled_ptr: u32,
    callee_frame_bytes: u32,
    dispatch_key_ofs: u32,
    callee_gcmap_ptr: i64,
    home_slot_base: u32,
    home_slots: u32,
    has_guard_not_forced_2: u32,
    marked_ordinary: u32,
    marked_labels: u32,
    label_ref_slots: u32,
) {
    let _ = ca_dispatch_slot(number);
    let table = WASM_CA_DISPATCH.lock();
    let has_guard_not_forced_2 =
        if has_guard_not_forced_2 != 0 || compiled_ptr_has_gnf2(compiled_ptr) {
            1
        } else {
            0
        };
    let entries = table
        .as_ref()
        .expect("CALL_ASSEMBLER dispatch table disappeared while publishing");
    if has_guard_not_forced_2 != 0 && compiled_ptr != 0 {
        mark_gnf2_entries_for_compiled_ptr(entries, compiled_ptr);
    }
    let entry = entries
        .get(&number)
        .expect("CALL_ASSEMBLER dispatch entry disappeared while publishing");
    if has_guard_not_forced_2 != 0 {
        entry.has_guard_not_forced_2.store(1, Ordering::Release);
    }
    let mut targets = entry.targets.lock().unwrap_or_else(|e| e.into_inner());
    if targets.last().is_some_and(|current| {
        current.func_handle == func_handle
            && current.compiled_ptr == compiled_ptr
            && current.callee_frame_bytes == callee_frame_bytes
            && current.dispatch_key_ofs == dispatch_key_ofs
            && current.callee_gcmap_ptr == callee_gcmap_ptr
            && current.home_slot_base == home_slot_base
            && current.home_slots == home_slots
            && current.has_guard_not_forced_2 == has_guard_not_forced_2
            && current.marked_ordinary == marked_ordinary
            && current.marked_labels == marked_labels
            && current.label_ref_slots == label_ref_slots
    }) {
        return;
    }
    let target = Box::new(WasmCaRuntimeTarget {
        func_handle,
        compiled_ptr,
        callee_frame_bytes,
        dispatch_key_ofs,
        callee_gcmap_ptr,
        home_slot_base,
        home_slots,
        has_guard_not_forced_2,
        marked_ordinary,
        marked_labels,
        label_ref_slots,
    });
    let target_ptr = (&*target as *const WasmCaRuntimeTarget as usize) as u32;
    targets.push(target);
    entry.target_ptr.store(target_ptr, Ordering::Release);
}

/// Redirect existing callers of `old_number` to the installed target.
pub fn ca_dispatch_redirect(
    old_number: u64,
    func_handle: u32,
    compiled_ptr: u32,
    callee_frame_bytes: u32,
    dispatch_key_ofs: u32,
    callee_gcmap_ptr: i64,
    home_slot_base: u32,
    home_slots: u32,
    has_guard_not_forced_2: u32,
    marked_ordinary: u32,
    marked_labels: u32,
    label_ref_slots: u32,
) {
    ca_dispatch_publish(
        old_number,
        func_handle,
        compiled_ptr,
        callee_frame_bytes,
        dispatch_key_ofs,
        callee_gcmap_ptr,
        home_slot_base,
        home_slots,
        has_guard_not_forced_2,
        marked_ordinary,
        marked_labels,
        label_ref_slots,
    );
}

/// Remove every dispatch entry that still resolves to `compiled_ptr`.  This
/// also retracts redirects into a dropped replacement loop, while preserving
/// an old token whose entry has already been redirected elsewhere.
pub fn ca_dispatch_remove_compiled_ptr(compiled_ptr: u32) {
    let mut table = WASM_CA_DISPATCH.lock();
    if let Some(table) = table.as_mut() {
        table.retain(|_, entry| {
            entry
                .targets
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .last()
                .is_none_or(|target| target.compiled_ptr != compiled_ptr)
        });
    }
    if compiled_ptr != 0 {
        if let Some(set) = CA_GNF2_COMPILED_PTRS.lock().as_mut() {
            set.remove(&compiled_ptr);
        }
    }
}

pub fn ca_dispatch_remove(number: u64) {
    let mut table = WASM_CA_DISPATCH.lock();
    let Some(table) = table.as_mut() else {
        return;
    };
    let compiled_ptrs: Vec<u32> = table
        .remove(&number)
        .map(|entry| {
            entry
                .targets
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .iter()
                .map(|target| target.compiled_ptr)
                .filter(|&ptr| ptr != 0)
                .collect()
        })
        .unwrap_or_default();
    for compiled_ptr in compiled_ptrs {
        let still_used = table.values().any(|entry| {
            entry
                .targets
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .iter()
                .any(|target| target.compiled_ptr == compiled_ptr)
        });
        if !still_used {
            if let Some(set) = CA_GNF2_COMPILED_PTRS.lock().as_mut() {
                set.remove(&compiled_ptr);
            }
        }
    }
}

pub fn call_assembler_target(number: u64) -> Option<CallAssemblerTarget> {
    CALL_ASSEMBLER_TARGETS
        .lock()
        .as_ref()
        .and_then(|targets| targets.get(&number).cloned())
}

pub fn publish_call_assembler_target(number: u64, target: CallAssemblerTarget) {
    CALL_ASSEMBLER_TARGETS
        .lock()
        .get_or_insert_with(Default::default)
        .insert(number, target);
}

/// Remove metadata and the dispatch entry for an invalidated token.
pub fn remove_call_assembler_target(number: u64) {
    if let Some(targets) = CALL_ASSEMBLER_TARGETS.lock().as_mut() {
        targets.remove(&number);
    }
    ca_dispatch_remove(number);
}

/// Retract all metadata aliases which point at a dropped compiled loop.
pub fn remove_call_assembler_targets_for_compiled_ptr(compiled_ptr: u32) {
    if let Some(targets) = CALL_ASSEMBLER_TARGETS.lock().as_mut() {
        targets.retain(|_, target| target.compiled_ptr as u32 != compiled_ptr);
    }
    ca_dispatch_remove_compiled_ptr(compiled_ptr);
}

/// Serializes tests that mutate cpu-global tables (finish singletons,
/// finish singletons, GC box, label targets). The wasm host never
/// interleaves those; cargo's parallel unit-test runner does. Held by
/// every lib test in this crate.
#[cfg(test)]
pub static FAIL_DESCR_TEST_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

/// Acquires [`FAIL_DESCR_TEST_LOCK`], then installs a fresh cpu.
///
/// `BaseBackendTest.setup_method` does `self.cpu = self.get_cpu()` and
/// sets `done_with_this_frame_descr_* = None`. The wasm host keeps those
/// tables process-global (guest writes an exit index, not a cpu pointer),
/// so the equivalent is to empty them here. Drop is
/// `AsmMemoryManager._delete`: the cpu dies with the frame
/// (`cpu.gc_ll_descr` / shadowstack). Struct fields drop in declaration
/// order, so `_cleanup` is declared first and runs while `_lock` is
/// still held.
#[cfg(test)]
pub fn lock_cpu() -> CpuTestGuard {
    let guard = CpuTestGuard {
        _cleanup: CpuTestCleanup,
        _lock: FAIL_DESCR_TEST_LOCK.lock(),
    };
    reset_cpu_for_tests();
    guard
}

/// Empty the process-global cpu tables and uninstall this thread's
/// `gc_ll_descr`. `BaseBackendTest.setup_method` / `get_cpu`.
#[cfg(test)]
fn reset_cpu_for_tests() {
    {
        let mut exits = FINISH_EXITS.lock();
        *exits = [None, None, None, None, None];
    }
    *LABEL_TARGETS.lock() = None;
    *CALL_ASSEMBLER_TARGETS.lock() = None;
    *WASM_CA_DISPATCH.lock() = None;
    *CA_GNF2_COMPILED_PTRS.lock() = None;
    crate::gc_box::clear();
    majit_gc::shadow_stack::clear();
    crate::clear_pending_inlines_for_tests();
    crate::jit_exc_clear();
    crate::set_wasm_jitframe_tid(0);
}

#[cfg(test)]
struct CpuTestCleanup;

#[cfg(test)]
impl Drop for CpuTestCleanup {
    fn drop(&mut self) {
        // `AsmMemoryManager._delete` / cpu teardown, still under the lock.
        reset_cpu_for_tests();
    }
}

#[cfg(test)]
pub struct CpuTestGuard {
    _cleanup: CpuTestCleanup,
    _lock: parking_lot::MutexGuard<'static, ()>,
}

/// Global `label descr identity → LabelTarget` registry (see `LabelTarget`).
/// The wasm host is single-threaded; the `Mutex` is for `static` soundness
/// only. `compile_loop` inserts every resumable label of a peeled loop;
/// `CompiledWasmLoop::drop` removes its own entries (guarded by
/// `func_handle`, so a recompile that re-stamped the same descr keeps the
/// replacement's entry).
pub static LABEL_TARGETS: parking_lot::Mutex<
    Option<std::collections::HashMap<usize, LabelTarget>>,
> = parking_lot::Mutex::new(None);

/// Look up a label target by descr identity.
pub fn label_target(descr_id: usize) -> Option<LabelTarget> {
    LABEL_TARGETS
        .lock()
        .as_ref()
        .and_then(|m| m.get(&descr_id).copied())
}

/// Publish a label target (see `LABEL_TARGETS`).
pub fn publish_label_target(descr_id: usize, target: LabelTarget) {
    LABEL_TARGETS
        .lock()
        .get_or_insert_with(Default::default)
        .insert(descr_id, target);
}

/// Retract a published label if it still names `func_handle`.
/// Same handle guard as [`CompiledWasmLoop::drop`]: a later publish that
/// re-stamped the same descr onto a different slot keeps the replacement.
pub fn retract_label_target_if_handle(descr_id: usize, func_handle: u32) {
    if descr_id == 0 || func_handle == 0 {
        return;
    }
    let mut reg = LABEL_TARGETS.lock();
    if let Some(map) = reg.as_mut()
        && let Some(t) = map.get(&descr_id)
        && t.func_handle == func_handle
    {
        map.remove(&descr_id);
        crate::BRIDGE_DIAG[22].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

pub(crate) fn write_bridge_cell(base: u32, fail_index: u32, slot: u32) {
    #[cfg(target_arch = "wasm32")]
    if base != 0 {
        let cell = (base as usize + fail_index as usize * 4) as *mut u32;
        unsafe { core::ptr::write(cell, slot) };
    }
    #[cfg(not(target_arch = "wasm32"))]
    let _ = (base, fail_index, slot);
}

pub(crate) fn write_bridge_cell_aliases(primary: u32, retained: u32, fail_index: u32, slot: u32) {
    write_bridge_cell(primary, fail_index, slot);
    if retained != 0 && retained != primary {
        write_bridge_cell(retained, fail_index, slot);
    }
}

/// Guard-dispatch metadata of a bridge chained onto a loop, kept on the
/// source loop's `CompiledWasmLoop.chained_trace_meta` keyed by the bridge's
/// backend `trace_id`. Lets `compile_bridge` chain a NESTED sub-bridge onto a
/// guard that lives inside an already-chained bridge: the failing guard's
/// meta descr carries `(trace_id, per-trace fail_index)`, and this record
/// supplies the owning bridge's cell array and livelock advance flags — the
/// same data `CompiledWasmLoop` holds for the loop's own guards.
pub struct ChainedTraceMeta {
    /// Base address of the bridge's per-guard bridge-slot cell array
    /// (`CompiledWasmLoop::bridge_cells_base` analog); `0` = no dispatch.
    pub cells_base: u32,
    /// Cell array baked into a retained standalone copy of this bridge
    /// after it was inlined. `reemit_loop` points `cells_base` at the
    /// merged-region slice; inbound JUMPs still run the old module, which
    /// reads this alias. `0` = no retained copy.
    pub retained_cells_base: u32,
    /// Cell count = the bridge's own guard count.
    pub num_cells: usize,
    /// Per-guard, per-fail-arg induction-advance flags
    /// (`CompiledWasmLoop::guard_fail_arg_advanced` analog).
    pub guard_fail_arg_advanced: Vec<Vec<bool>>,
    /// Number of values each guard transfers to a bridge.  A parameter entry
    /// is admitted only when this agrees with the bridge's input list.
    pub guard_fail_arg_counts: Vec<usize>,
    /// Whether this trace's guard epilogue has typed parameter dispatch arms.
    pub bridge_param_dispatch: bool,
    /// Ordinary Ref homes this bridge published. After the bridge is
    /// inlined, this is the merged stream's extent — `RefHomes::collect`
    /// reassigns across that stream, so the standalone count is too short.
    /// A nested sub-bridge floors its map to this.
    pub num_ref_homes: usize,
    /// LABEL-capture homes this bridge published.
    pub used_label_homes: usize,
}

/// Two guest words a running loop loads on its back-edge.
/// `generation` at offset 0, `slot` at offset 4.
#[repr(C)]
pub struct ResumeEntry {
    pub generation: std::sync::atomic::AtomicU32,
    pub slot: std::sync::atomic::AtomicU32,
}

impl ResumeEntry {
    pub fn new() -> Self {
        Self {
            generation: std::sync::atomic::AtomicU32::new(1),
            slot: std::sync::atomic::AtomicU32::new(0),
        }
    }

    pub fn addr(&self) -> u32 {
        self as *const Self as usize as u32
    }
}

/// Compiled wasm loop metadata, stored in `JitCellToken.compiled`.
pub struct CompiledWasmLoop {
    /// Owning `JitCellToken` number, used to retract this loop's
    /// CALL_ASSEMBLER target metadata on drop.
    pub token_number: u64,
    pub trace_id: u64,
    pub input_types: Vec<Type>,
    /// Shared-table slot of the materialized wasm function.  Straight-line
    /// function-entry traces may keep this at zero until their first actual
    /// execution: an invalidated trace that never reaches `execute_token`
    /// must not pay the host Wasmtime compilation cost.
    pub(crate) func_handle: Cell<u32>,
    /// `{generation, slot}` the running loop loads on each back-edge.
    /// The allocation stays put across in-place re-emission; the module
    /// bakes its address. `slot` is the table index `replace_module` keeps.
    pub(crate) resume_entry: Box<ResumeEntry>,
    /// Encoded module retained until lazy host materialization.  This is
    /// backend assembler state, not metainterpreter state: the optimized trace
    /// and all per-token descriptors have already been installed exactly as
    /// in the eager path.
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    pub(crate) pending_wasm_bytes: RefCell<Option<Vec<u8>>>,
    /// The owning token metadata receives the guard-descriptor tracer only
    /// after the host accepts this module. Deferred root traces keep this
    /// handle until `materialize_func_handle` crosses that acceptance point.
    pub(crate) compiled_loop_token: Arc<majit_backend::CompiledLoopToken>,
    /// Serializes the eager and lazy acceptance paths. The wasm runtime is
    /// single-threaded, matching the surrounding `Cell`/`RefCell` fields.
    pub(crate) descrs_registered: Cell<bool>,
    /// This loop's own guard/finish exit descriptors (positions `[0,
    /// num_guard_cells)`, per-trace order), followed by the descr slices of
    /// every chained bridge `compile_bridge` appended (positional bookkeeping
    /// for `bridge_descr_ranges` — layouts and jitcounter hashes). An exit
    /// resolves through `jf_descr` (the cell), not by indexing this vec.
    /// `RefCell` because the
    /// append happens through the shared `&JitCellToken` the bridge attaches
    /// to; the wasm host is single-threaded so no cross-thread access occurs.
    pub fail_descrs: RefCell<Vec<Arc<WasmFailDescr>>>,
    pub num_inputs: usize,
    pub max_output_slots: usize,
    /// Number of Ref-typed values given a home slot in the frame's Ref-home
    /// region (`codegen::HOME_SLOT_BASE`). `execute_token` sizes the host
    /// frame to include this region and registers each home slot as a GC root.
    pub num_ref_homes: Cell<usize>,
    /// LABEL-capture homes this loop actually initialized. Frozen geometry
    /// may reserve more; a later bridge's published map must still cover
    /// these so a keyed tail-call cannot drop them. `Cell` so `reemit_loop`
    /// can raise it when a merge publishes a wider capture tail.
    pub used_label_homes: Cell<usize>,
    /// Geometry frozen when this token was first compiled. Every bridge
    /// chained onto it is emitted against this exact layout.
    pub frame: crate::codegen::FrameGeometry,
    /// Per-loop `jf_gcmap` for the Ref-home region.  Like RPython's assembler
    /// gcmap allocation, this remains valid after `execute_token` returns: a
    /// virtualizable token can keep that JITFRAME alive and force it later.
    /// `Cell` so `reemit_loop` can replace it when a merge widens RefHomes.
    pub home_gcmap_ptr: Cell<usize>,
    /// Base address (shared linear memory) of this loop's per-guard bridge-slot
    /// cell array — one i32 per `fail_index`, `0` = no bridge. The trace's
    /// epilogue reads `cells[fail_index]` and `compile_bridge` writes a bridge's
    /// table slot here. `0` when the trace has no in-module dispatch (native, or
    /// a guardless / straight-line trace).
    pub bridge_cells_base: Cell<u32>,
    /// Cell array of a retained pre-growth owner module. When the LABEL
    /// tail grows, the replacement is installed at a new table slot and
    /// this stays the array the old module still reads. `0` until that
    /// split. `compile_bridge` writes both aliases.
    pub retained_owner_cells_base: Cell<u32>,
    /// Byte length of the module this loop was last emitted as, own ops and
    /// every merged region together. A merge re-emits the whole owner, so this
    /// is what the next merge charges cranelift, and
    /// `inline_trip_threshold_for` turns it into the entry count that merge has
    /// to earn before it is taken.
    pub(crate) module_bytes: Cell<u32>,
    /// Number of cells in the `bridge_cells_base` array = this loop's own guard
    /// count at compile time. A bridge attaches only to one of these original
    /// guards (`source_fail_index < num_guard_cells`); descrs appended past this
    /// range belong to already-chained bridges and have no cell of their own.
    pub num_guard_cells: Cell<usize>,
    /// True when this is a peeled loop (`codegen::is_resumable_peeled`) — there
    /// is real work (a preamble = the unrolled first iteration) before the last
    /// `LABEL`, single- or multi-label. Such a loop carries the resume-at-LABEL
    /// entry `br_table` (key = label ordinal + 1) so a loop-closing bridge can
    /// re-enter at any of its labels. A loop-closing bridge re-enters through
    /// the loop's table slot (the function entry); for a peeled loop,
    /// re-running the preamble against mid-loop state would never advance the
    /// induction variable — an infinite loop. `compile_bridge` therefore
    /// declines a loop-closing bridge UNLESS its JUMP's
    /// target label resolves to a published, resumable `LabelTarget`.
    pub has_preamble: bool,
    /// Descr identity (`Arc::as_ptr`) of each `LABEL`, in ordinal order; `0`
    /// for a descr-less label. `compile_bridge` resolves a closing JUMP's
    /// target label by matching its descr identity against this list — a JUMP
    /// whose descr is not here targets ANOTHER trace's label (e.g. a sibling
    /// retrace specialization, whose start label carries the same stamped
    /// ordinal) and must not be chained into this loop.
    pub label_descrs: Vec<usize>,
    /// Per-guard (indexed by this loop's own `fail_index`), per-fail-arg:
    /// whether the value was produced by induction-advancing arithmetic after
    /// the loop-header label — fresh in the failing iteration. Consulted by
    /// `compile_bridge`'s livelock check: a loop-closing bridge that JUMPs
    /// such a fail arg verbatim still advances the chained cycle.
    pub guard_fail_arg_advanced: Vec<Vec<bool>>,
    /// Number of fail arguments for every guard/finish exit in this trace.
    pub guard_fail_arg_counts: Vec<usize>,
    /// Whether this module transfers a compiled bridge's fail arguments as
    /// wasm call parameters instead of reloading their positional frame slots.
    pub bridge_param_dispatch: bool,
    /// `(source_trace_id, source_fail_index, start, count)` ranges into
    /// `fail_descrs` for each chained bridge `compile_bridge` appended (lib.rs
    /// extend site). Lets `compiled_bridge_fail_descr_layouts` map a source
    /// guard back to its bridge's
    /// appended descr slice — the wasm analog of dynasm's
    /// `lookup_bridge_addr` (runner.rs). Keyed by BOTH the source guard's
    /// owning trace and its per-trace fail index: with nested chaining, the
    /// loop's guard `k` and a chained bridge's guard `k` are distinct sources.
    /// Recorded in lockstep with the `extend`, inside the same `borrow_mut`
    /// critical section.
    pub bridge_descr_ranges: RefCell<Vec<(u64, u32, usize, usize)>>,
    /// Guard-dispatch metadata of every bridge chained onto this loop, keyed
    /// by the bridge's backend `trace_id` (see [`ChainedTraceMeta`]). Lets a
    /// guard INSIDE a chained bridge chain its own nested sub-bridge.
    pub chained_trace_meta: RefCell<std::collections::HashMap<u64, ChainedTraceMeta>>,
    /// Owns this loop's current cell array and every bridge cell array chained
    /// onto it. A re-emission retains the old array for an already-running
    /// module before switching its baked base to a new array.
    pub _bridge_owned_cells: RefCell<Vec<Box<[u32]>>>,
    /// Direct-loop guard index to bridge table slot. `patch_jump_for_descr`
    /// rewrites the guard's own jump to reach a newly attached bridge; a wasm
    /// module is immutable once compiled, so the branch instead reads a slot
    /// out of a mutable cell array, and these are the writes a re-emission has
    /// to replay into its fresh array.
    pub bridge_slots: RefCell<std::collections::HashMap<u32, u32>>,
    /// The same, for a guard that lives inside a trace chained onto this loop,
    /// keyed by `(owning trace_id, per-trace fail index)`. A standalone chained
    /// bridge keeps its cells in its own module's array, which survives; a
    /// region merged into this loop does not, because a re-emission reallocates
    /// the loop array its guards are carved out of. Replayed once the rebuilt
    /// `chained_trace_meta` names the new bases.
    pub chained_bridge_slots: RefCell<std::collections::HashMap<(u64, u32), u32>>,
    /// Post-intern module inputs retained for a loop re-emission. Entry
    /// bridges store `None` because they tail-call another loop.
    pub reemit: RefCell<Option<crate::codegen::ModuleBuildInputs>>,
    /// The environment-gated identity re-emission runs once per token.
    pub reemitted: Cell<bool>,
    /// `(descr identity, table slot)` for every label published by a bridge
    /// chained onto this loop. The bridge module lives as long as its source
    /// loop, so `Drop` retracts the entries that still name that bridge's
    /// slot.
    pub bridge_owned_label_targets: RefCell<Vec<(usize, u32)>>,
    /// Set when `compile_bridge` accepts a self-recursive `CallAssemblerR`
    /// bridge (`PYRE_WASM_CA`) for this loop. While set, `compile_bridge`
    /// declines chaining any FURTHER bridge into this recursion (the guard
    /// falls back to host round-trips): a chained bridge deopting inside the
    /// CA recursion trips a resume seam that reads a clobbered class — see
    /// the decline site for the failing suite shapes.
    pub ca_active: Cell<bool>,
    /// A guard reached through this loop as a wasm CALL_ASSEMBLER callee was
    /// structurally declined by `compile_bridge`.  Admission refuses this
    /// target, because entering it from compiled wasm would only blackhole.
    pub ca_terminal_declined: Cell<bool>,
    /// Compiled callers that baked this loop as their CALL_ASSEMBLER target.
    /// A terminal callee decline invalidates them for a no-CA retrace.
    pub ca_callers: RefCell<Vec<std::sync::Arc<std::sync::atomic::AtomicBool>>>,
}

// Compiled loop metadata is transferred through the token's `Any + Send`
// holder, but all access to its IR snapshot and cell arrays is confined to the
// single wasm execution thread. The contained `RefCell`s enforce that runtime
// ownership model; moving the holder does not permit concurrent access.
unsafe impl Send for CompiledWasmLoop {}

impl CompiledWasmLoop {
    pub fn eager_func_handle(&self) -> u32 {
        self.func_handle.get()
    }

    /// Publish the descriptors only for a module the host can execute.
    pub(crate) fn register_descrs_once(&self) {
        if self.descrs_registered.replace(true) {
            return;
        }
        let descrs = self.fail_descrs.borrow();
        let meta: Vec<DescrRef> = descrs
            .iter()
            .filter_map(|descr| descr.meta_descr.clone())
            .collect();
        let tracer: Arc<dyn std::any::Any + Send + Sync> = Arc::new(meta);
        self.compiled_loop_token
            .asmmemmgr_gcreftracers
            .lock()
            .push(tracer);
    }

    /// Materialize a lazily-installed root trace.  The wasm host is
    /// single-threaded, matching the RefCell/Cell ownership used throughout
    /// this structure, so one trace can only cross this gate once.
    pub fn materialize_func_handle(&self) -> Result<u32, majit_backend::BackendError> {
        let current = self.func_handle.get();
        if current != 0 {
            return Ok(current);
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            Ok(0)
        }
        #[cfg(target_arch = "wasm32")]
        {
            let pending = self.pending_wasm_bytes.borrow();
            let Some(bytes) = pending.as_deref() else {
                return Err(majit_backend::BackendError::Unsupported(
                    "wasm trace has neither a function handle nor pending module bytes".into(),
                ));
            };
            let handle = crate::glue::compile_module_cached(bytes);
            if handle == 0 {
                return Err(majit_backend::BackendError::Unsupported(
                    "wasm host rejected the lazily compiled trace module".into(),
                ));
            }
            self.register_descrs_once();
            self.func_handle.set(handle);
            if self
                .resume_entry
                .slot
                .load(std::sync::atomic::Ordering::Relaxed)
                == 0
            {
                self.resume_entry
                    .slot
                    .store(handle, std::sync::atomic::Ordering::Relaxed);
            }
            let mut blocks = self.compiled_loop_token.asmmemmgr_blocks.lock();
            if let Some(resources) = blocks
                .last_mut()
                .and_then(|block| block.downcast_mut::<crate::release::LoopAsmResources>())
            {
                resources.table_slots.push(handle);
            } else {
                let mut resources = crate::release::LoopAsmResources::default();
                resources.table_slots.push(handle);
                blocks.push(Box::new(resources));
            }
            drop(pending);
            self.pending_wasm_bytes.borrow_mut().take();
            Ok(handle)
        }
    }
}

impl Drop for CompiledWasmLoop {
    fn drop(&mut self) {
        // Remove every token alias still targeting this module, including a
        // redirect source. A source redirected to a newer module survives an
        // old-loop drop because its dispatch `compiled_ptr` no longer matches.
        // Label rows and table slots are owned by `LoopAsmResources` in
        // `asmmemmgr_blocks`, dropped by `free_loop_and_bridges`.
        remove_call_assembler_targets_for_compiled_ptr(self as *const Self as usize as u32);
        // Retract this loop's published label targets so a later bridge
        // cannot chain into a dropped loop's stale table slot. Guarded by
        // `func_handle`: a recompile that re-stamped the same descr onto its
        // replacement loop has already overwritten the entry, which must
        // survive the old loop's drop.
        let handle = self.func_handle.get();
        for id in self.label_descrs.iter().copied() {
            retract_label_target_if_handle(id, handle);
        }
        for (id, slot) in self.bridge_owned_label_targets.get_mut().iter().copied() {
            retract_label_target_if_handle(id, slot);
        }
    }
}
