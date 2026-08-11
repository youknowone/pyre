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

// ── JITFRAMEINFO (jitframe.py:30-40) ────────────────────────────────

/// RPython JITFRAMEINFO — per-compiled-loop metadata.
///
/// jitframe.py:30-40
/// ```python
/// JITFRAMEINFO = lltype.Struct('JITFRAMEINFO',
///     ('jfi_frame_depth', lltype.Signed),
///     ('jfi_frame_size', lltype.Signed),
/// )
/// ```
#[derive(Debug, Default)]
#[repr(C)]
pub struct JitFrameInfo {
    /// jfi_frame_depth: Signed — number of word-sized slots in jf_frame.
    pub jfi_frame_depth: isize,
    /// jfi_frame_size: Signed — total byte size of the JitFrame allocation.
    pub jfi_frame_size: isize,
}

// jitframe.py:28
// JITFRAMEINFO_SIZE = 2 * SIZEOFSIGNED
pub const JITFRAMEINFO_SIZE: usize = 2 * SIZEOFSIGNED;
const _: () = assert!(std::mem::size_of::<JitFrameInfo>() == JITFRAMEINFO_SIZE);

// jitframe.py:18-22 — jitframeinfo_update_depth
// jitframe.py:24-26 — jitframeinfo_clear
impl JitFrameInfo {
    /// jitframe.py:18-22 `jitframeinfo_update_depth(jfi, base_ofs, new_depth)`.
    ///
    /// The fields are `isize` (lltype.Signed = machine word); the frame-depth
    /// call sites (CompiledLoopToken / backend assemblers) thread `base_ofs`
    /// and `new_depth` as `i64`, so the word-width conversion is localized
    /// here.
    pub fn update_frame_depth(&mut self, base_ofs: i64, new_depth: i64) {
        let base_ofs = base_ofs as isize;
        let new_depth = new_depth as isize;
        if new_depth > self.jfi_frame_depth {
            self.jfi_frame_depth = new_depth;
            self.jfi_frame_size = base_ofs + new_depth * SIZEOFSIGNED as isize;
        }
    }

    /// jitframe.py:24-26
    pub fn clear(&mut self) {
        self.jfi_frame_size = 0;
        self.jfi_frame_depth = 0;
    }
}

// jitframe.py:42-43
pub const NULLFRAMEINFO: *const JitFrameInfo = std::ptr::null();
pub type JitFrameInfoPtr = *const JitFrameInfo;

// ── JITFRAME (jitframe.py:59-91) ────────────────────────────────────

/// RPython JITFRAME — the GC-managed frame for compiled code.
///
/// jitframe.py:61-91
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

// jitframe.py:93-95 — getofs(name)
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
/// `jitframe.py:48` allocates every frame through the GC, so upstream's
/// compiled code always holds a frame that has a header behind it. Pyre also
/// builds frames off the GC — the runner's entry frame, the realloc slowpath,
/// and the JITFRAME nursery slowpath's fallback, the class
/// `shadow_stack::register_libc_jitframe` tracks — and compiled code cannot
/// tell the two apart. `_reload_frame_if_necessary`
/// (`aarch64/assembler.py:967-980`) re-applies the non-array write-barrier
/// fast path to the current frame after every collecting call, and that fast
/// path loads the flag byte at `jit_wb_if_flag_byteofs`, which is *negative*
/// (`gc.py:285-293` measures it from the object pointer, and the flags sit in
/// the header at `obj-4`).
///
/// Handing out a bare allocation base therefore puts that load outside the
/// block: usually it reads unrelated bytes and, when they happen to carry
/// TRACK_YOUNG_PTRS, enters the barrier helper for nothing; when the block
/// lands at the start of a mapped region it faults outright. Reserving the
/// header word makes the read in-bounds, and the zeroed flags give it the
/// same answer a freshly nursery-allocated frame gives — no barrier.
///
/// Returns null when the allocation fails.
pub fn alloc_off_gc_jitframe(size_bytes: usize) -> *mut JitFrame {
    let Some(total) = OFF_GC_PREFIX.checked_add(size_bytes) else {
        return std::ptr::null_mut();
    };
    let Some(layout) = off_gc_layout(total) else {
        return std::ptr::null_mut();
    };
    let base = unsafe { std::alloc::alloc_zeroed(layout) };
    if base.is_null() {
        return std::ptr::null_mut();
    }
    unsafe {
        *(base as *mut u64) = total as u64;
        base.add(OFF_GC_PREFIX) as *mut JitFrame
    }
}

/// Release a frame from [`alloc_off_gc_jitframe`].
///
/// The frame pointer is not the block base — the size slot and the header word
/// precede it — so this must be used instead of freeing the frame pointer.
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
        let layout = off_gc_layout(total).expect("off-GC jitframe size slot was corrupted");
        std::alloc::dealloc(base, layout);
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

    /// jitframe.py:48-52 — jitframe_allocate.
    ///
    /// Initialize a freshly-allocated (zero-filled) JitFrame at `ptr`.
    /// Caller is responsible for allocation (nursery or malloc).
    ///
    /// # Safety
    /// `ptr` must point to at least `alloc_size(depth)` zero-filled bytes.
    pub unsafe fn init(ptr: *mut JitFrame, info: *const JitFrameInfo, depth: usize) {
        unsafe {
            // RPython: frame.jf_frame_info = frame_info
            // (other fields are zero from malloc)
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

    /// jitframe.py:54-57 — jitframe_resolve.
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

// ── jitframe_trace (jitframe.py:104-136) ────────────────────────────

/// GC trace callback for JitFrame.
///
/// jitframe.py:104-136 — traces fixed GCREF fields then walks
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

        // jitframe.py:111-114
        let max: usize = if IS_32BIT { 32 } else { 64 };

        // jitframe.py:115-116
        let gcmap_raw = (*obj_addr).jf_gcmap;
        if gcmap_raw.is_null() {
            return; // done
        }

        // jitframe.py:118 — gcmap_lgt = (gcmap + GCMAPLENGTHOFS).signed[0]
        let gcmap_lgt = *(gcmap_raw.add(GCMAPLENGTHOFS) as *const isize);

        // jitframe.py:119-135
        let mut no: isize = 0;
        while no < gcmap_lgt {
            // jitframe.py:121 — cur = (gcmap + GCMAPBASEOFS + UNSIGN_SIZE * no).unsigned[0]
            let cur = *(gcmap_raw.add(GCMAPBASEOFS + UNSIGN_SIZE * no as usize) as *const usize);
            let mut bitindex: usize = 0;
            while bitindex < max {
                if cur & (1usize << bitindex) != 0 {
                    // jitframe.py:126 — index = no * SIZEOFSIGNED * 8 + bitindex
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
/// jitframe.py:49 — `rgc.register_custom_trace_hook(JITFRAME, lambda_jitframe_trace)`
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
/// jitframe.py:48-52 — the `jitframe_allocate` function registers
/// the custom trace hook on first call. In pyre, the TypeInfo is
/// registered once with `gc.register_type(jitframe_type_info())`.
/// jitframe.py:48-52 — JITFRAME is a GcStruct with a trailing Array(Signed).
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

/// Allocate-in-oldgen flag: jitframe should NOT be nursery-allocated
/// when possible, to avoid the cost of copying the (potentially large)
/// trailing array during minor collection. When this returns true,
/// the allocator should use `alloc_external` or similar.
pub fn jitframe_prefer_oldgen() -> bool {
    true
}

// ── realloc_frame (llmodel.py:127-154) ──────────────────────────────

/// Reallocate a JITFRAME when the frame-depth requirement exceeds its
/// current allocation. Ported from `rpython/jit/backend/llsupport/
/// llmodel.py:127-154 realloc_frame`.
///
/// The assembler-emitted `_frame_realloc_slowpath`
/// (`aarch64/assembler.py:434-493`) calls this helper after
/// `_check_frame_depth` (`aarch64/assembler.py:927-961`) detects
/// `jf_frame.length < expected_depth`.
///
/// `alloc` is a raw allocator: given `size_bytes` it must return a
/// zero-filled `*mut JitFrame` payload whose GC header is registered
/// with the jitframe custom-trace hook. `write_barrier` covers `new_jf`
/// (`llmodel.py:150`) — the generational barrier for the copied
/// `jf_frame` / `jf_savedata` / `jf_guard_exc` stores.
///
/// It does **not** cover `old_jf`. `frame.jf_forward = new_frame`
/// (`llmodel.py:141`) is an ordinary GC-struct field assignment, so upstream's
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
        if expected_depth > (*fi).jfi_frame_depth {
            (*fi).update_frame_depth(base_ofs as i64, expected_depth as i64);
        }
        let size_bytes = (*fi).jfi_frame_size;

        // llmodel.py:140 — new_frame = jitframe.JITFRAME.allocate(frame_info)
        let new_jf = alloc(size_bytes);
        debug_assert!(!new_jf.is_null(), "realloc_frame: alloc returned null");
        JitFrame::init(new_jf, fi, (*fi).jfi_frame_depth as usize);

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
}
