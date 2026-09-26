//! JitFrame layout + GC trace — `rpython/jit/backend/llsupport/jitframe.py`.
//!
//! This module owns only the responsibilities that upstream's
//! `jitframe.py` owns: the `JITFRAMEINFO` / `JITFRAME` struct shape,
//! byte offsets, allocation primitives (`alloc_size`, `init`), the
//! `jitframe_resolve` walk, and the GC `jitframe_trace` +
//! `jitframe_type_info` registration.
//!
//! Cpu-level deadframe accessors (`get_int_value`, `get_ref_value`,
//! `get_float_value`, `get_latest_descr`, `get_savedata_ref`,
//! `set_savedata_ref`) live in `crate::llmodel` to mirror upstream's
//! split between `jitframe.py` and `llmodel.py` on `AbstractLLCPU`.

// jitframe.py:8
// SIZEOFSIGNED = rffi.sizeof(lltype.Signed)
pub const SIZEOFSIGNED: usize = std::mem::size_of::<isize>();

// jitframe.py:9
// IS_32BIT = (SIZEOFSIGNED == 4)
pub const IS_32BIT: bool = SIZEOFSIGNED == 4;

// jitframe.py:15-16
// GCMAP = lltype.Array(lltype.Unsigned)
// NULLGCMAP = lltype.nullptr(GCMAP)
//
// GCMAP layout: [length: usize, data[0]: u64, data[1]: u64, ...]
// RPython Array(Unsigned) has a length prefix followed by items.
pub const NULLGCMAP: *const u8 = std::ptr::null();

// ── JITFRAMEINFO (jitframe.py) ────────────────────────────────

/// RPython JITFRAMEINFO — per-compiled-loop metadata.
///
/// jitframe.py
/// ```python
/// JITFRAMEINFO = lltype.Struct('JITFRAMEINFO',
///     ('jfi_frame_depth', lltype.Signed),
///     ('jfi_frame_size', lltype.Signed),
/// )
/// ```
/// Two `lltype.Signed` words (`JITFRAMEINFO`), the layout `gen_malloc_frame`
/// loads with `GC_LOAD_I`.
///
/// The fields are atomics so a store in `jitframeinfo_update_depth` racing a
/// load is not a data race. The ordering stays relaxed: that update is a
/// plain assignment, and `GC_LOAD_I` carries no ordering. `set_ll_function_addr`
/// is what publishes the machine code; the entry loads the depth only after
/// that, and allocates the frame bytes after the load.
#[repr(C)]
pub struct JitFrameInfo {
    /// jfi_frame_depth: Signed — number of word-sized slots in jf_frame.
    pub jfi_frame_depth: std::sync::atomic::AtomicIsize,
    /// jfi_frame_size: Signed — total byte size of the JitFrame allocation.
    pub jfi_frame_size: std::sync::atomic::AtomicIsize,
}

// jitframe.py:28
// JITFRAMEINFO_SIZE = 2 * SIZEOFSIGNED
pub const JITFRAMEINFO_SIZE: usize = 2 * SIZEOFSIGNED;
const _: () = assert!(std::mem::size_of::<JitFrameInfo>() == JITFRAMEINFO_SIZE);

// jitframe.py — jitframeinfo_update_depth
// jitframe.py — jitframeinfo_clear
impl Default for JitFrameInfo {
    fn default() -> Self {
        Self {
            jfi_frame_depth: std::sync::atomic::AtomicIsize::new(0),
            jfi_frame_size: std::sync::atomic::AtomicIsize::new(0),
        }
    }
}

impl std::fmt::Debug for JitFrameInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JitFrameInfo")
            .field("jfi_frame_depth", &self.depth())
            .field("jfi_frame_size", &self.size())
            .finish()
    }
}

impl JitFrameInfo {
    /// `jfi_frame_depth` — `GC_LOAD_I` of the `lltype.Signed` word.
    #[inline]
    pub fn depth(&self) -> isize {
        self.jfi_frame_depth
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// `jfi_frame_size` — `GC_LOAD_I` of the `lltype.Signed` word.
    #[inline]
    pub fn size(&self) -> isize {
        self.jfi_frame_size
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// jitframe.py `jitframeinfo_update_depth(jfi, base_ofs, new_depth)`.
    ///
    /// The fields are machine words (`lltype.Signed`); the frame-depth call
    /// sites thread `base_ofs` and `new_depth` as `i64`, so the conversion
    /// stays here.
    ///
    /// `jitframeinfo_update_depth` assigns only when the depth grows. A bridge
    /// compile and `realloc_frame` both call this, so the depth moves with
    /// `fetch_max` and a smaller store cannot replace a larger one.
    /// `gen_malloc_frame` loads `jfi_frame_size` and `jfi_frame_depth` with
    /// separate `GC_LOAD_I`s: the size is raised first and only grows, then
    /// the depth is published. A reader that observes this depth has a size
    /// at least as large. A reader that still observes the old depth with the
    /// new size over-allocates.
    pub fn update_frame_depth(&self, base_ofs: i64, new_depth: i64) {
        let base_ofs = base_ofs as isize;
        let new_depth = new_depth as isize;
        if new_depth <= self.depth() {
            return;
        }
        let new_size = base_ofs + new_depth * SIZEOFSIGNED as isize;
        self.jfi_frame_size
            .fetch_max(new_size, std::sync::atomic::Ordering::Relaxed);
        self.jfi_frame_depth
            .fetch_max(new_depth, std::sync::atomic::Ordering::Relaxed);
    }

    /// jitframe.py — `jitframeinfo_clear` assigns both words.
    pub fn clear(&self) {
        self.jfi_frame_size
            .store(0, std::sync::atomic::Ordering::Relaxed);
        self.jfi_frame_depth
            .store(0, std::sync::atomic::Ordering::Relaxed);
    }
}

// jitframe.py:42-43
pub const NULLFRAMEINFO: *const JitFrameInfo = std::ptr::null();
pub type JitFrameInfoPtr = *const JitFrameInfo;

// ── JITFRAME (jitframe.py) ────────────────────────────────────

/// RPython JITFRAME — the GC-managed frame for compiled code.
///
/// jitframe.py
/// ```python
/// JITFRAME = GcStruct('JITFRAME',
///     ('jf_frame_info', Ptr(JITFRAMEINFO)),
///     ('jf_descr', GCREF),
///     ('jf_force_descr', GCREF),
///     ('jf_gcmap', Ptr(GCMAP)),
///     ('jf_savedata', GCREF),
///     ('jf_guard_exc', GCREF),
///     ('jf_forward', Ptr(JITFRAME)),
///     ('jf_frame', Array(Signed)),
///     rtti = True,
/// )
/// ```
///
/// This is the FIXED header. The variable-length `jf_frame` array
/// follows immediately after in memory, preceded by its length field
/// (RPython Array layout: `[length, item0, item1, ...]`).
///
/// Total allocation: `JITFRAME_FIXED_SIZE + SIGN_SIZE * (1 + depth)`
/// where the +1 accounts for the length field.
#[repr(C)]
pub struct JitFrame {
    /// `jf_frame_info: Ptr(JITFRAMEINFO)` — non-GC raw pointer.
    pub jf_frame_info: *const JitFrameInfo,
    /// `jf_descr: GCREF` — last executed descr. Traced by GC.
    pub jf_descr: usize,
    /// `jf_force_descr: GCREF` — guard_not_forced descr. Traced by GC.
    pub jf_force_descr: usize,
    /// `jf_gcmap: Ptr(GCMAP)` — pointer to GC reference bitmap.
    pub jf_gcmap: *const u8,
    /// `jf_savedata: GCREF` — front-end savedata. Traced by GC.
    pub jf_savedata: usize,
    /// `jf_guard_exc: GCREF` — exception from guards. Traced by GC.
    pub jf_guard_exc: usize,
    /// `jf_forward: Ptr(JITFRAME)` — forwarding pointer for GC.
    pub jf_forward: *mut JitFrame,
    // ── jf_frame: Array(Signed) ──
    // RPython Array layout: [length: Signed, items: Signed...]
    // The length field is part of the trailing allocation.
    // jf_frame_length and jf_frame items follow in memory.
}

/// Byte size of the JitFrame fixed header (excludes jf_frame array).
///
/// RPython equivalent: `JITFRAME_FIXED_SIZE` in backend arch.py.
pub const JITFRAME_FIXED_SIZE: usize = std::mem::size_of::<JitFrame>();

// jitframe.py — getofs(name)
// Offset constants for codegen.
pub const JF_FRAME_INFO_OFS: i32 = std::mem::offset_of!(JitFrame, jf_frame_info) as i32;
pub const JF_DESCR_OFS: i32 = std::mem::offset_of!(JitFrame, jf_descr) as i32;
pub const JF_FORCE_DESCR_OFS: i32 = std::mem::offset_of!(JitFrame, jf_force_descr) as i32;
pub const JF_GCMAP_OFS: i32 = std::mem::offset_of!(JitFrame, jf_gcmap) as i32;
pub const JF_SAVEDATA_OFS: i32 = std::mem::offset_of!(JitFrame, jf_savedata) as i32;
pub const JF_GUARD_EXC_OFS: i32 = std::mem::offset_of!(JitFrame, jf_guard_exc) as i32;
pub const JF_FORWARD_OFS: i32 = std::mem::offset_of!(JitFrame, jf_forward) as i32;

// jitframe.py:97-102
// GCMAPLENGTHOFS = arraylengthoffset(GCMAP)
// GCMAPBASEOFS = itemoffsetof(GCMAP, 0)
// BASEITEMOFS = itemoffsetof(JITFRAME.jf_frame, 0)
// LENGTHOFS = arraylengthoffset(JITFRAME.jf_frame)
// SIGN_SIZE = sizeof(Signed)
// UNSIGN_SIZE = sizeof(Unsigned)

/// RPython Array layout: `[length: Signed, items...]`.
/// GCMAPLENGTHOFS = offset of length in GCMAP array = 0.
pub const GCMAPLENGTHOFS: usize = 0;
/// GCMAPBASEOFS = offset of first item in GCMAP array = SIZEOFSIGNED.
pub const GCMAPBASEOFS: usize = SIZEOFSIGNED;

/// LENGTHOFS = offset of jf_frame's length field from jf_frame start.
/// In RPython Array(Signed) layout: length is at offset 0 from the
/// array pointer, items start at offset SIZEOFSIGNED.
pub const LENGTHOFS: usize = 0;

/// BASEITEMOFS = offset of jf_frame[0] from the jf_frame array pointer.
/// = SIZEOFSIGNED (skip the length field).
pub const BASEITEMOFS: usize = SIZEOFSIGNED;

/// Byte offset from JitFrame start to the jf_frame array pointer
/// (which points to the length field, followed by items).
pub const JF_FRAME_OFS: usize = JITFRAME_FIXED_SIZE;

/// Byte offset from JitFrame start to jf_frame[0] (skips length field).
pub const FIRST_ITEM_OFFSET: usize = JF_FRAME_OFS + BASEITEMOFS;

/// SIGN_SIZE = sizeof(Signed)
pub const SIGN_SIZE: usize = SIZEOFSIGNED;
/// UNSIGN_SIZE = sizeof(Unsigned)
pub const UNSIGN_SIZE: usize = std::mem::size_of::<usize>();

// jitframe.py:138
pub type JitFramePtr = *mut JitFrame;

/// An off-GC jitframe's block starts with the total size, so the frame
/// pointer alone is enough to free it, and then the header word the JIT's
/// negative reads land in. See [`alloc_off_gc_jitframe`].
///
/// ```text
///   base            base+8            base+16 == the frame pointer
///   [ total size ]  [ header word ]   [ JitFrame … ]
/// ```
const OFF_GC_SIZE_SLOT: usize = majit_gc::header::GcHeader::SIZE;
const OFF_GC_HEADER: usize = majit_gc::header::GcHeader::SIZE;
const OFF_GC_PREFIX: usize = OFF_GC_SIZE_SLOT + OFF_GC_HEADER;
const _: () = assert!(OFF_GC_SIZE_SLOT >= std::mem::size_of::<u64>());

fn off_gc_layout(total: usize) -> Option<std::alloc::Layout> {
    std::alloc::Layout::from_size_align(total, majit_gc::header::GcHeader::SIZE).ok()
}

/// Allocate a JITFRAME outside the GC, with the header word in front of it.
///
/// `jitframe.py` allocates every frame through the GC, so upstream's
/// compiled code always holds a frame that has a header behind it. Pyre also
/// builds frames off the GC — [`malloc_jitframe`] under a descr with no
/// `JITFRAME` type id, the class `shadow_stack::register_libc_jitframe`
/// tracks — and compiled code cannot tell the two apart.
/// `_reload_frame_if_necessary` (`aarch64/assembler.py`) re-applies
/// the non-array write-barrier fast path to the current frame after every
/// collecting call, and that fast path loads the flag byte at
/// `jit_wb_if_flag_byteofs`, which is *negative* (`gc.py:285-293` measures it
/// from the object pointer, and the flags sit in the header at `obj-4`).
///
/// Handing out a bare allocation base therefore puts that load outside the
/// block: usually it reads unrelated bytes and, when they happen to carry
/// TRACK_YOUNG_PTRS, enters the barrier helper for nothing; when the block
/// lands at the start of a mapped region it faults outright. Reserving the
/// header word makes the read in-bounds, and the zeroed flags give it the
/// same answer a freshly nursery-allocated frame gives — no barrier.
///
/// The block comes off the thread's free list when
/// [`crate::deadframe::jitframe_pool_enabled`] says so, and from the
/// allocator otherwise. A recycled block clears the header word and the
/// fixed fields only; a fresh block also clears the spill area so those
/// bytes are not uninit.
///
/// Returns null when the allocation fails.
pub fn alloc_off_gc_jitframe(size_bytes: usize) -> *mut JitFrame {
    let Some(total) = OFF_GC_PREFIX.checked_add(size_bytes) else {
        return std::ptr::null_mut();
    };
    let Some(layout) = off_gc_layout(total) else {
        return std::ptr::null_mut();
    };
    if let Some(base) = crate::deadframe::take_pooled_block(total) {
        // A parked block keeps its size slot. Zero the header word and the
        // fixed JITFRAME fields (`jf_gcmap` must be null before any walk).
        // Spill slots are written by the entry before they are read or
        // published through a gcmap; `jitframe_allocate` does not scrub a
        // recycled nursery object per slot either — the nursery reset did
        // that in bulk, and a parked block is this path's recycle.
        zero_off_gc_frame_prefix(base);
        return unsafe { base.add(OFF_GC_PREFIX) as *mut JitFrame };
    }
    let base = unsafe { std::alloc::alloc(layout) };
    if base.is_null() {
        return std::ptr::null_mut();
    }
    unsafe {
        *(base as *mut u64) = total as u64;
        zero_off_gc_frame_prefix(base);
        // The length word and spill area of a fresh block were not part of
        // the prefix clear. `JitFrame::init` writes the length; spill slots
        // follow the same write-before-read contract as a pooled block.
        // Zero them once so a read of an unwritten slot is not uninit memory.
        let frame = base.add(OFF_GC_PREFIX);
        let tail = OFF_GC_PREFIX + std::mem::size_of::<JitFrame>();
        if total > tail {
            std::ptr::write_bytes(frame.add(std::mem::size_of::<JitFrame>()), 0, total - tail);
        }
        frame as *mut JitFrame
    }
}

/// Header word plus the fixed `JITFRAME` fields. Does not touch the size
/// slot at `base`, or the spill array.
fn zero_off_gc_frame_prefix(base: *mut u8) {
    let n = OFF_GC_HEADER + std::mem::size_of::<JitFrame>();
    unsafe { std::ptr::write_bytes(base.add(OFF_GC_SIZE_SLOT), 0, n) };
}

/// Release a frame from [`alloc_off_gc_jitframe`].
///
/// The frame pointer is not the block base — the size slot and the header word
/// precede it — so this must be used instead of freeing the frame pointer.
/// A block the thread's free list has room for is parked there instead.
///
/// # Safety
/// `frame` must have come from [`alloc_off_gc_jitframe`] and must no longer be
/// reachable from compiled code or the shadow stack.
pub unsafe fn free_off_gc_jitframe(frame: *mut JitFrame) {
    if frame.is_null() {
        return;
    }
    unsafe {
        let base = (frame as *mut u8).sub(OFF_GC_PREFIX);
        let total = *(base as *const u64) as usize;
        if crate::deadframe::give_back_pooled_block(base, total) {
            return;
        }
        dealloc_off_gc_block(base, total);
    }
}

/// Return an [`alloc_off_gc_jitframe`] block to the allocator.
///
/// # Safety
/// `base` must be the block base and `total` the value of its size slot.
pub(crate) unsafe fn dealloc_off_gc_block(base: *mut u8, total: usize) {
    let layout = off_gc_layout(total).expect("off-GC jitframe size slot was corrupted");
    unsafe { std::alloc::dealloc(base, layout) };
}

// ── The descr's frame allocation ────────────────────────────────────

/// Zero-fill a frame the collector handed back.
///
/// RPython's `gen_malloc_frame` initializes the fixed fields itself; its
/// native nursery reset uses `arena_reset(..., 0)` and does not clear recycled
/// bytes. This conservative allocator remains for callers whose initialization
/// contract has not yet been narrowed to those fields. Old-gen arenas recycle
/// bytes too.
fn zeroed_gc_frame(gcref: majit_ir::GcRef, size_bytes: usize) -> *mut JitFrame {
    assert!(!gcref.is_null(), "JITFRAME allocation failed");
    unsafe { std::ptr::write_bytes(gcref.0 as *mut u8, 0, size_bytes) };
    gcref.0 as *mut JitFrame
}

/// Place a GC-managed JITFRAME according to [`jitframe_prefer_oldgen`].
///
/// `collect` is the nursery arm of [`malloc_jitframe`]; the no-collect
/// sibling asks for `alloc_nursery_no_collect_typed` instead, matching
/// `llmodel.py:140`'s realloc slow path. The old-gen arm is already
/// non-collecting (`alloc_oldgen_typed`).
fn alloc_gc_jitframe(
    gc: &mut dyn majit_gc::GcAllocator,
    type_id: u32,
    size_bytes: usize,
    collect: bool,
) -> *mut JitFrame {
    let gcref = if jitframe_prefer_oldgen() {
        gc.alloc_oldgen_typed(type_id, size_bytes)
    } else if collect {
        gc.alloc_nursery_typed(type_id, size_bytes)
    } else {
        gc.alloc_nursery_no_collect_typed(type_id, size_bytes)
    };
    zeroed_gc_frame(gcref, size_bytes)
}

fn entry_gc_frame(gcref: majit_ir::GcRef) -> *mut JitFrame {
    assert!(!gcref.is_null(), "JITFRAME allocation failed");
    let frame = gcref.0 as *mut JitFrame;
    unsafe { std::ptr::write_bytes(frame, 0, 1) };
    frame
}

/// Allocate a compiled-entry frame without clearing its trailing spill slots.
///
/// Entry code writes every live slot before reading it or publishing it through
/// a GC map. [`JitFrame::init`] supplies the fixed-field initialization that
/// RPython emits in `rewrite.py`'s `gen_malloc_frame`.
pub fn malloc_entry_jitframe(
    gc: &mut dyn majit_gc::GcAllocator,
    size_bytes: usize,
) -> *mut JitFrame {
    match gc.jitframe_type_id() {
        Some(type_id) => {
            let gcref = if jitframe_prefer_oldgen() {
                gc.alloc_oldgen_typed(type_id, size_bytes)
            } else {
                gc.alloc_nursery_typed(type_id, size_bytes)
            };
            entry_gc_frame(gcref)
        }
        None => malloc_host_jitframe(size_bytes),
    }
}

/// `llmodel.py` `frame = self.gc_ll_descr.malloc_jitframe(frame_info)`,
/// reached through `GcLLDescription.malloc_jitframe` (gc.py) and
/// `jitframe_allocate` (`jitframe.py`).
///
/// The descr decides where the frame lives: under its `JITFRAME` type id it
/// is a collector object [`jitframe_prefer_oldgen`] places — nursery, like
/// `lltype.malloc(JITFRAME)`, unless that flag asks for old-gen — traced
/// through `jitframe_trace`; without one — an allocator with no type table,
/// or [`HostHeapGc`] when nothing is installed — it is a host block the
/// libc-jitframe tracer walks in place. The caller sees one frame pointer
/// either way; what differs is only whether the deadframe that later owns
/// it takes a root slot ([`jitframe_is_gc_object`]) and whether a store
/// into it needs a barrier ([`jitframe_write_barrier`]).
///
/// The nursery arm may collect, so the caller must have its inputs rooted.
/// `size_bytes` is the payload size ([`JitFrame::alloc_size`]).
pub fn malloc_jitframe(gc: &mut dyn majit_gc::GcAllocator, size_bytes: usize) -> *mut JitFrame {
    match gc.jitframe_type_id() {
        Some(type_id) => alloc_gc_jitframe(gc, type_id, size_bytes, true),
        None => malloc_host_jitframe(size_bytes),
    }
}

/// [`malloc_jitframe`] for a caller holding unrooted references on the Rust
/// stack: the nursery arm falls back to the old generation instead of
/// collecting. `llmodel.py:140`'s realloc slow path and the compiled-entry
/// frame are built this way.
pub fn malloc_jitframe_no_collect(
    gc: &mut dyn majit_gc::GcAllocator,
    size_bytes: usize,
) -> *mut JitFrame {
    match gc.jitframe_type_id() {
        Some(type_id) => alloc_gc_jitframe(gc, type_id, size_bytes, false),
        None => malloc_host_jitframe(size_bytes),
    }
}

/// The host-heap arm of [`malloc_jitframe`]: an off-GC block, registered so
/// the collector's root walk finds its ref slots.
pub fn malloc_host_jitframe(size_bytes: usize) -> *mut JitFrame {
    let frame = alloc_off_gc_jitframe(size_bytes);
    assert!(!frame.is_null(), "JITFRAME host allocation failed");
    // A collector walking the shadow stack has to recognise this block.
    // With no collector installed nothing walks, and `make_execute_token`
    // does not publish the frame. Debug builds count that case so a later
    // `store_singleton` fails instead of leaving the block out of
    // `LIBC_JF_REGISTRY` and `LIVE_DEADFRAMES`.
    if majit_gc::collector_installed() {
        majit_gc::shadow_stack::register_libc_jitframe(frame as usize);
    } else {
        #[cfg(debug_assertions)]
        {
            majit_gc::shadow_stack::note_unregistered_host_jitframe();
            // The install flag may have been published between the check and
            // the note. Register in that window so the frame is not left out
            // of the set.
            if majit_gc::collector_installed() {
                majit_gc::shadow_stack::register_libc_jitframe(frame as usize);
                majit_gc::shadow_stack::release_unregistered_host_jitframe();
            }
        }
    }
    frame
}

/// Drop the registry entry of a [`malloc_host_jitframe`] block.
///
/// A release build only calls `unregister_libc_jitframe`. Debug builds also
/// drop the pre-collector count when the block was never registered.
#[inline(always)]
pub(crate) fn release_malloc_host_jitframe(frame: *mut JitFrame) {
    #[cfg(not(debug_assertions))]
    {
        majit_gc::shadow_stack::unregister_libc_jitframe(frame as usize);
    }
    #[cfg(debug_assertions)]
    {
        if frame.is_null() {
            return;
        }
        let addr = frame as usize;
        if majit_gc::collector_installed()
            && majit_gc::shadow_stack::unregister_libc_jitframe_if_present(addr)
        {
            return;
        }
        majit_gc::shadow_stack::release_unregistered_host_jitframe();
    }
}

/// Release a frame from [`malloc_jitframe`] that no compiled code, shadow
/// stack or deadframe still names.
///
/// A GC object is left to the collector — RPython never frees a frame — so
/// only the host arm does anything.
///
/// # Safety
/// `frame` must have come from [`malloc_jitframe`] under this descr and be
/// unreachable.
pub unsafe fn free_jitframe(gc: &dyn majit_gc::GcAllocator, frame: *mut JitFrame) {
    if jitframe_is_gc_object(gc) {
        return;
    }
    unsafe { free_host_jitframe(frame) };
}

/// The host arm of [`free_jitframe`], for an owner that already knows its
/// frame is a host block.
///
/// # Safety
/// As [`free_jitframe`].
pub unsafe fn free_host_jitframe(frame: *mut JitFrame) {
    release_malloc_host_jitframe(frame);
    unsafe { free_off_gc_jitframe(frame) };
}

/// Whether this descr's frames are collector objects — the moving,
/// header-carrying kind that must be held through a root slot — as opposed
/// to host blocks. One descr answers this the same way for every frame it
/// ever built.
#[inline]
pub fn jitframe_is_gc_object(gc: &dyn majit_gc::GcAllocator) -> bool {
    gc.jitframe_type_id().is_some()
}

/// `llop.gc_writebarrier(lltype.Void, frame)` (`llmodel.py`) for a raw store
/// into a frame from [`malloc_jitframe`].
///
/// Upstream emits the barrier unconditionally because every frame is a GC
/// object; here a host block has no header for a barrier to mark, so the
/// descr's arm decides.
pub fn jitframe_write_barrier(gc: &mut dyn majit_gc::GcAllocator, frame: *mut JitFrame) {
    if jitframe_is_gc_object(gc) {
        gc.write_barrier(majit_ir::GcRef(frame as usize));
    }
}

/// Refuse an install whose descr cannot allocate frames the collector can
/// trace.
///
/// `jitframe.py` registers `JITFRAME`'s custom trace hook as part of
/// translating the frontend, so the shape is settled before any compiled code
/// exists. Same ordering here, made a precondition: a collector with a type
/// table must arrive with its `JITFRAME` id set (`set_jitframe_type_id`),
/// and the id must name a `JITFRAME` — an id minted for another shape would
/// have the collector copy frames under the wrong layout. A collector without
/// a table is exempt: it has no shape to name, and its frames are host
/// blocks. Registering here instead would mint an id in a table the frontend
/// may already have frozen (`gctypelayout.py encode_type_shapes_now`).
pub fn check_jitframe_descr(gc: &dyn majit_gc::GcAllocator) {
    if !gc.has_type_registry() {
        return;
    }
    let Some(id) = gc.jitframe_type_id() else {
        panic!(
            "installing a collector with a type registry requires the JITFRAME type id: \
             register majit_backend::jitframe::jitframe_type_info() on that collector \
             and pass the id to GcAllocator::set_jitframe_type_id() BEFORE the install"
        );
    };
    assert_eq!(
        gc.type_size(id),
        Some(jitframe_type_info().size),
        "JITFRAME type id {id} does not name a JITFRAME in the collector being installed"
    );
}

/// The descr in force when no collector is installed: `GcLLDescr_boehm`
/// (gc.py), which `get_ll_description(None)` (gc.py) selects.
///
/// Non-moving, no type table, `supports_guard_gc_type = False`. Its only
/// job is [`malloc_jitframe`]'s host arm; compiled code under it allocates
/// nothing else, because a backend without a collector never turns
/// `new_via_gc` on, so every other allocation entry is unreachable and says
/// so.
pub struct HostHeapGc;

impl HostHeapGc {
    fn no_heap() -> ! {
        panic!("HostHeapGc allocates only jitframes; no collector is installed")
    }
}

impl majit_gc::GcAllocator for HostHeapGc {
    fn alloc_nursery(&mut self, _size: usize) -> majit_ir::GcRef {
        Self::no_heap()
    }
    fn alloc_nursery_no_collect(&mut self, _size: usize) -> majit_ir::GcRef {
        Self::no_heap()
    }
    fn alloc_varsize(
        &mut self,
        _base_size: usize,
        _item_size: usize,
        _length: usize,
    ) -> majit_ir::GcRef {
        Self::no_heap()
    }
    fn alloc_varsize_no_collect(
        &mut self,
        _base_size: usize,
        _item_size: usize,
        _length: usize,
    ) -> majit_ir::GcRef {
        Self::no_heap()
    }
    fn write_barrier(&mut self, _obj: majit_ir::GcRef) {}
    fn jit_remember_young_pointer_from_array(&mut self, _obj: majit_ir::GcRef) {}
    fn remember_young_pointer_from_array2(
        &mut self,
        _obj: majit_ir::GcRef,
        _index: usize,
        _card: u32,
    ) {
    }
    fn collect_nursery(&mut self) {}
    fn collect_full(&mut self) {}
    fn nursery_free(&self) -> *mut u8 {
        Self::no_heap()
    }
    fn nursery_free_addr(&self) -> usize {
        Self::no_heap()
    }
    fn nursery_top(&self) -> *const u8 {
        Self::no_heap()
    }
    fn nursery_top_addr(&self) -> usize {
        Self::no_heap()
    }
    fn max_nursery_object_size(&self) -> usize {
        0
    }
}

// ── JitFrame methods ────────────────────────────────────────────────

impl JitFrame {
    /// Total allocation size for a jitframe with `depth` slots.
    ///
    /// Layout: [JitFrame header | jf_frame_length: isize | jf_frame[0..depth]: isize...]
    pub fn alloc_size(depth: usize) -> usize {
        JITFRAME_FIXED_SIZE + SIZEOFSIGNED * (1 + depth) // +1 for length field
    }

    /// jitframe.py — jitframe_allocate.
    ///
    /// Initialize a freshly-allocated JitFrame at `ptr`.
    /// Caller is responsible for allocation (nursery or malloc).
    ///
    /// # Safety
    /// `ptr` must point to at least `alloc_size(depth)` writable bytes with a
    /// zero-filled fixed header. Trailing slots need not be initialized yet.
    pub unsafe fn init(ptr: *mut JitFrame, info: *const JitFrameInfo, depth: usize) {
        unsafe {
            (*ptr).jf_frame_info = info;
            // Write the jf_frame array length
            let len_ptr = (ptr as *mut u8).add(JF_FRAME_OFS) as *mut isize;
            *len_ptr = depth as isize;
        }
    }

    /// Get a mutable slice of the jf_frame items (excluding length field).
    ///
    /// # Safety
    /// `ptr` must be a valid JitFrame with at least `len` trailing slots.
    pub unsafe fn frame_slots_mut(ptr: *mut JitFrame, len: usize) -> &'static mut [isize] {
        unsafe {
            let base = (ptr as *mut u8).add(FIRST_ITEM_OFFSET) as *mut isize;
            std::slice::from_raw_parts_mut(base, len)
        }
    }

    /// Get an immutable slice of the jf_frame items.
    /// # Safety
    /// The caller must uphold every validity, runtime-type, aliasing, and lifetime
    /// invariant required by the object and pointer arguments for the entire call.
    pub unsafe fn frame_slots(ptr: *const JitFrame, len: usize) -> &'static [isize] {
        unsafe {
            let base = (ptr as *const u8).add(FIRST_ITEM_OFFSET) as *const isize;
            std::slice::from_raw_parts(base, len)
        }
    }

    /// Read the jf_frame array length.
    /// # Safety
    /// The caller must uphold every validity, runtime-type, aliasing, and lifetime
    /// invariant required by the object and pointer arguments for the entire call.
    pub unsafe fn frame_length(ptr: *const JitFrame) -> isize {
        unsafe {
            let len_ptr = (ptr as *const u8).add(JF_FRAME_OFS + LENGTHOFS) as *const isize;
            *len_ptr
        }
    }

    /// jitframe.py — jitframe_resolve.
    /// # Safety
    /// The caller must uphold every validity, runtime-type, aliasing, and lifetime
    /// invariant required by the object and pointer arguments for the entire call.
    pub unsafe fn resolve(mut frame: *mut JitFrame) -> *mut JitFrame {
        unsafe {
            while !(*frame).jf_forward.is_null() {
                frame = (*frame).jf_forward;
            }
            frame
        }
    }

    /// Raw pointer to `jf_frame[index]` — an `isize`-sized slot.
    ///
    /// # Safety
    /// `ptr` must be a valid JitFrame whose trailing array length
    /// exceeds `index`.
    pub unsafe fn slot_ptr(ptr: *mut JitFrame, index: usize) -> *mut isize {
        unsafe { (ptr as *mut u8).add(FIRST_ITEM_OFFSET + index * SIZEOFSIGNED) as *mut isize }
    }

    /// Const variant of `slot_ptr`.
    /// # Safety
    /// The caller must uphold every validity, runtime-type, aliasing, and lifetime
    /// invariant required by the object and pointer arguments for the entire call.
    pub unsafe fn slot_ptr_const(ptr: *const JitFrame, index: usize) -> *const isize {
        unsafe { (ptr as *const u8).add(FIRST_ITEM_OFFSET + index * SIZEOFSIGNED) as *const isize }
    }
}

// ── jitframe_trace (jitframe.py) ────────────────────────────

/// GC trace callback for JitFrame.
///
/// jitframe.py jitframe_trace — traces fixed GCREF fields then walks
/// the gcmap bitmap to find Ref-typed jf_frame slots.
///
/// `trace_callback` is called for each GCREF slot address that the
/// GC needs to visit (read and potentially update).
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn jitframe_trace(obj_addr: *mut JitFrame, mut trace_callback: impl FnMut(*mut usize)) {
    unsafe {
        // jitframe.py:105-109 — trace fixed GCREF header fields
        trace_callback(&mut (*obj_addr).jf_descr);
        trace_callback(&mut (*obj_addr).jf_force_descr);
        trace_callback(&mut (*obj_addr).jf_savedata);
        trace_callback(&mut (*obj_addr).jf_guard_exc);
        trace_callback(&mut (*obj_addr).jf_forward as *mut *mut JitFrame as *mut usize);

        jitframe_trace_gcmap(obj_addr, trace_callback);
    }
}

/// The `jf_gcmap`-directed half of [`jitframe_trace`], without the fixed
/// header fields.
///
/// The two halves carry different things: the header slots name the frame's
/// own bookkeeping objects, while these name whatever the compiled trace held
/// in its Ref bank.  A host that must ask something further about a traced
/// value — what kind of object it is, what it points to — can only ask it of
/// the second set, so the walk is reachable on its own.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn jitframe_trace_gcmap(
    obj_addr: *mut JitFrame,
    mut trace_callback: impl FnMut(*mut usize),
) {
    unsafe {
        // jitframe.py:111-114
        let max: usize = if IS_32BIT { 32 } else { 64 };

        // jitframe.py:115-116
        let gcmap_raw = (*obj_addr).jf_gcmap;
        if gcmap_raw.is_null() {
            return; // done
        }

        // jitframe.py — gcmap_lgt = (gcmap + GCMAPLENGTHOFS).signed[0]
        let gcmap_lgt = *(gcmap_raw.add(GCMAPLENGTHOFS) as *const isize);

        // jitframe.py:119-135
        let mut no: isize = 0;
        while no < gcmap_lgt {
            // jitframe.py — cur = (gcmap + GCMAPBASEOFS + UNSIGN_SIZE * no).unsigned[0]
            let cur = *(gcmap_raw.add(GCMAPBASEOFS + UNSIGN_SIZE * no as usize) as *const usize);
            let mut bitindex: usize = 0;
            while bitindex < max {
                if cur & (1usize << bitindex) != 0 {
                    // jitframe.py — index = no * SIZEOFSIGNED * 8 + bitindex
                    let index = no as usize * SIZEOFSIGNED * 8 + bitindex;
                    // jitframe.py:128-130 — sanity check
                    let frame_lgt =
                        *((obj_addr as *const u8).add(JF_FRAME_OFS + LENGTHOFS) as *const isize);
                    // jitframe.py:130 — ll_assert(index < frame_lgt, "bogus ...")
                    // RPython ll_assert = RPyAssert: real assertion, not no-op.
                    assert!(
                        (index as isize) < frame_lgt,
                        "bogus frame field get: index={index} >= frame_lgt={frame_lgt}"
                    );
                    // jitframe.py:131-133 — trace the slot
                    let slot_addr = (obj_addr as *mut u8).add(FIRST_ITEM_OFFSET + SIGN_SIZE * index)
                        as *mut usize;
                    trace_callback(slot_addr);
                }
                bitindex += 1;
            }
            no += 1;
        }
    }
}

// ── GC type registration (jitframe.py:49) ───────────────────────────

/// Custom trace bridge for `TypeInfo::with_custom_trace`.
///
/// Adapts `jitframe_trace(*mut JitFrame, FnMut(*mut usize))` to the
/// `CustomTraceFn(usize, &mut dyn FnMut(*mut GcRef))` interface.
///
/// jitframe.py — `rgc.register_custom_trace_hook(JITFRAME, lambda_jitframe_trace)`
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn jitframe_custom_trace(obj_addr: usize, f: &mut dyn FnMut(*mut majit_ir::GcRef)) {
    unsafe {
        jitframe_trace(obj_addr as *mut JitFrame, |slot_ptr| {
            // GcRef and usize are both word-sized; reinterpret the slot.
            f(slot_ptr as *mut majit_ir::GcRef);
        });
    }
}

/// Build a `TypeInfo` for JitFrame with the custom trace hook registered.
///
/// jitframe.py — the `jitframe_allocate` function registers
/// the custom trace hook on first call. In pyre, the TypeInfo is
/// registered once with `gc.register_type(jitframe_type_info())`.
/// jitframe.py — JITFRAME is a GcStruct with a trailing Array(Signed).
/// GC must know the varsize layout to copy the full object (header + array).
///
/// Layout: [JitFrame header (56 bytes)] [length: Signed] [items: Signed...]
/// - base_size = JITFRAME_FIXED_SIZE + SIGN_SIZE (64) — the fixed header
///   plus the 8-byte jf_frame_length field that precedes the items.
///   `TypeInfo::total_instance_size(length)` computes
///   `base_size + item_size * length`, so base_size MUST include every
///   byte before item[0], otherwise the GC copy is 8 bytes short and the
///   last jf_frame slot is lost across every minor collection.
/// - item_size = SIZEOFSIGNED (8) — each jf_frame slot is one Signed
/// - length_offset = JITFRAME_FIXED_SIZE (56) from obj start
pub fn jitframe_type_info() -> majit_gc::trace::TypeInfo {
    majit_gc::trace::TypeInfo::varsize_with_custom_trace(
        JITFRAME_FIXED_SIZE + SIGN_SIZE, // base_size (header + length field)
        SIZEOFSIGNED,                    // item_size: each array slot is 8 bytes
        JITFRAME_FIXED_SIZE,             // length_offset: jf_frame.length at header end
        jitframe_custom_trace,
    )
}

/// Whether a GC-managed JITFRAME should be born in the old generation.
///
/// `jitframe_allocate` is `lltype.malloc(JITFRAME, depth)` — a regular
/// GcStruct malloc. Framework GC puts that in the nursery unless the
/// request is large enough for `external_malloc`. There is no
/// prefer-oldgen switch upstream, so this is false and
/// [`malloc_jitframe`] reads it: a true here that the allocator ignored
/// was a dead flag claiming the opposite of `lltype.malloc`.
pub fn jitframe_prefer_oldgen() -> bool {
    false
}

// ── realloc_frame (llmodel.py) ──────────────────────────────

/// Reallocate a JITFRAME when the frame-depth requirement exceeds its
/// current allocation. Ported from `rpython/jit/backend/llsupport/
/// llmodel.py realloc_frame`.
///
/// The assembler-emitted `_frame_realloc_slowpath`
/// (`aarch64/assembler.py build_frame_realloc_slowpath`) calls this helper after
/// `_check_frame_depth` (`aarch64/assembler.py`) detects
/// `jf_frame.length < expected_depth`.
///
/// `alloc` is a raw allocator: given `size_bytes` it must return a
/// zero-filled `*mut JitFrame` payload whose GC header is registered
/// with the jitframe custom-trace hook. `write_barrier` covers `new_jf`
/// (`llmodel.py`) — the generational barrier for the copied
/// `jf_frame` / `jf_savedata` / `jf_guard_exc` stores.
///
/// It does **not** cover `old_jf`. `frame.jf_forward = new_frame`
/// (`llmodel.py`) is an ordinary GC-struct field assignment, so upstream's
/// framework transform emits a barrier for it too; this helper writes it raw.
/// The one caller today (`dynasm_realloc_frame`) allocates frames off the GC
/// heap through `alloc_off_gc_jitframe` and passes a shadow-stack registration
/// rather than a barrier, so nothing is missing for it. **A caller whose frames
/// are GC-managed must barrier `old_jf` itself after this returns** — the
/// cranelift backend's own copy of this routine does exactly that.
///
/// # Safety
/// - `old_jf` must be a live, resolved `*mut JitFrame`.
/// - `old_jf->jf_frame_info` must point to a `JitFrameInfo` that is
///   writable when `expected_depth > jfi_frame_depth` — RPython wraps
///   the mutation in `enter_assembler_writing()` /
///   `leave_assembler_writing()`; pyre keeps the frame_info in ordinary
///   heap memory so no permission flip is required.
pub unsafe fn realloc_frame<A, B>(
    old_jf: *mut JitFrame,
    expected_depth: isize,
    base_ofs: isize,
    alloc: A,
    write_barrier: B,
) -> *mut JitFrame
where
    A: FnOnce(isize) -> *mut JitFrame,
    B: FnOnce(*mut JitFrame),
{
    unsafe {
        // llmodel.py:132-139 — widen frame_info when we need more depth.
        let fi = (*old_jf).jf_frame_info as *mut JitFrameInfo;
        if expected_depth > (*fi).depth() {
            (*fi).update_frame_depth(base_ofs as i64, expected_depth as i64);
        }
        let size_bytes = (*fi).size();

        // llmodel.py:140 — new_frame = jitframe.JITFRAME.allocate(frame_info)
        let new_jf = alloc(size_bytes);
        debug_assert!(!new_jf.is_null(), "realloc_frame: alloc returned null");
        JitFrame::init(new_jf, fi, (*fi).depth() as usize);

        // llmodel.py:141 — frame.jf_forward = new_frame
        (*old_jf).jf_forward = new_jf;

        // llmodel.py:142-146 — copy jf_frame items, zero source slots.
        let old_len = JitFrame::frame_length(old_jf) as usize;
        let new_len = JitFrame::frame_length(new_jf) as usize;
        let copy_len = old_len.min(new_len);
        let old_slots = JitFrame::frame_slots_mut(old_jf, old_len);
        let new_slots = JitFrame::frame_slots_mut(new_jf, new_len);
        for i in 0..copy_len {
            new_slots[i] = old_slots[i];
            old_slots[i] = 0;
        }

        // llmodel.py:147-148 — copy jf_savedata / jf_guard_exc.
        (*new_jf).jf_savedata = (*old_jf).jf_savedata;
        (*new_jf).jf_guard_exc = (*old_jf).jf_guard_exc;

        // llmodel.py:150 — llop.gc_writebarrier(new_frame)
        write_barrier(new_jf);

        new_jf
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use majit_gc::GcAllocator;

    #[test]
    fn entry_init_clears_the_header_without_touching_frame_slots() {
        let depth = 4;
        let words = JitFrame::alloc_size(depth) / std::mem::size_of::<usize>();
        let mut storage = vec![usize::MAX; words];
        let frame = entry_gc_frame(majit_ir::GcRef(storage.as_mut_ptr() as usize));
        let info = JitFrameInfo::default();

        unsafe { JitFrame::init(frame, &info, depth) };

        unsafe {
            assert_eq!((*frame).jf_frame_info, &info);
            assert_eq!((*frame).jf_descr, 0);
            assert_eq!((*frame).jf_force_descr, 0);
            assert!((*frame).jf_gcmap.is_null());
            assert_eq!((*frame).jf_savedata, 0);
            assert_eq!((*frame).jf_guard_exc, 0);
            assert!((*frame).jf_forward.is_null());
            assert_eq!(JitFrame::frame_length(frame), depth as isize);
            assert!(
                JitFrame::frame_slots(frame, depth)
                    .iter()
                    .all(|&slot| slot == -1)
            );
        }
    }

    /// The emitted write-barrier fast path loads one byte at
    /// `jit_wb_if_flag_byteofs` from the frame pointer, and that offset is
    /// negative. An off-GC frame must answer that load the way a nursery
    /// frame does — in-bounds, and with the flag clear so the barrier helper
    /// is not entered — which is the whole reason `alloc_off_gc_jitframe`
    /// reserves a header word instead of handing out the block base.
    #[test]
    fn off_gc_jitframe_reserves_the_negative_flag_byte() {
        let descr = majit_gc::WriteBarrierDescr::for_current_gc();
        assert!(
            descr.jit_wb_if_flag_byteofs < 0,
            "the flag byte is expected behind the object pointer"
        );
        assert!(
            (-descr.jit_wb_if_flag_byteofs) as usize <= majit_gc::header::GcHeader::SIZE,
            "the reserved header word must cover the flag byte offset"
        );

        let frame = alloc_off_gc_jitframe(JitFrame::alloc_size(8));
        assert!(!frame.is_null());
        let flag = unsafe { *(frame as *const u8).offset(descr.jit_wb_if_flag_byteofs as isize) };
        assert_eq!(
            flag & descr.jit_wb_if_flag_singlebyte,
            0,
            "a fresh off-GC frame must not look like it tracks young pointers"
        );
        unsafe { free_off_gc_jitframe(frame) };
    }

    /// `jitframe_allocate` is `lltype.malloc(JITFRAME)`; a true flag that
    /// [`malloc_jitframe`] ignored was a dead prefer-oldgen claim.
    #[test]
    fn malloc_jitframe_reads_prefer_oldgen() {
        assert!(
            !jitframe_prefer_oldgen(),
            "lltype.malloc(JITFRAME) is a nursery GcStruct malloc"
        );
        let mut gc = majit_gc::collector::MiniMarkGC::new();
        let tid = gc.register_type(jitframe_type_info());
        gc.set_jitframe_type_id(tid);
        let frame = malloc_jitframe(&mut gc, JitFrame::alloc_size(8));
        assert!(!frame.is_null());
        assert!(
            gc.is_in_nursery(frame as usize),
            "a false prefer-oldgen flag must not land the frame in mark-sweep old-gen"
        );
    }

    /// Two callers of `jitframeinfo_update_depth` with different depths leave
    /// the larger depth in place, and `jfi_frame_size` consistent with it.
    #[test]
    fn update_frame_depth_concurrent_callers_keep_the_larger_depth() {
        use std::sync::{Arc, Barrier};

        let info = Arc::new(JitFrameInfo::default());
        let barrier = Arc::new(Barrier::new(2));
        let base_ofs = 24i64;
        let low = 5i64;
        let high = 48i64;
        let run = |depth: i64| {
            let info = Arc::clone(&info);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                for _ in 0..4_000 {
                    info.update_frame_depth(base_ofs, depth);
                }
            })
        };
        let low_thread = run(low);
        let high_thread = run(high);
        low_thread.join().expect("low depth updater");
        high_thread.join().expect("high depth updater");

        assert_eq!(info.depth(), high as isize);
        let consistent = base_ofs as isize + (high as isize) * SIZEOFSIGNED as isize;
        assert_eq!(info.size(), consistent);
    }
}
