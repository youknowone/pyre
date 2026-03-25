//! JitFrame — RPython `jitframe.py` parity.
//!
//! A flat, GC-managed frame used by compiled code. Unlike PyFrame (which
//! has Vec/VecDeque/Drop semantics), JitFrame is a bump-allocated struct
//! with a variable-length `jf_frame: [i64]` array — exactly matching
//! RPython's `JITFRAME.jf_frame: Array(Signed)`.
//!
//! PyFrame is only materialized on guard failure (blackhole entry).

/// RPython JITFRAMEINFO — per-compiled-loop metadata.
///
/// jitframe.py:30-40
#[repr(C)]
pub struct JitFrameInfo {
    /// Number of word-sized slots in jf_frame.
    pub jfi_frame_depth: usize,
    /// Total byte size of the JitFrame allocation (header + slots).
    pub jfi_frame_size: usize,
}

/// JITFRAMEINFO_SIZE = 2 * SIZEOFSIGNED
pub const JITFRAMEINFO_SIZE: usize = std::mem::size_of::<JitFrameInfo>();

impl JitFrameInfo {
    /// jitframeinfo_update_depth
    pub fn update_depth(&mut self, base_ofs: usize, new_depth: usize) {
        if new_depth > self.jfi_frame_depth {
            self.jfi_frame_depth = new_depth;
            self.jfi_frame_size = base_ofs + new_depth * SIGN_SIZE;
        }
    }

    pub fn clear(&mut self) {
        self.jfi_frame_size = 0;
        self.jfi_frame_depth = 0;
    }
}

/// Size of a single slot (Signed) in the jf_frame array.
pub const SIGN_SIZE: usize = std::mem::size_of::<i64>();

/// RPython JITFRAME — the GC-managed frame for compiled code.
///
/// jitframe.py:61-91
///
/// This struct represents the FIXED header. The variable-length
/// `jf_frame` array follows immediately after in memory. Allocate
/// `JITFRAME_HEADER_SIZE + frame_depth * SIGN_SIZE` bytes total.
#[repr(C)]
pub struct JitFrame {
    /// Pointer to the per-loop frame info (depth, total size).
    pub jf_frame_info: *const JitFrameInfo,
    /// Last executed descr (GUARD or FINISH).
    /// GCREF slot — traced by GC.
    pub jf_descr: i64,
    /// guard_not_forced descr.
    /// GCREF slot — traced by GC.
    pub jf_force_descr: i64,
    /// Pointer to GC reference bitmap for jf_frame slots.
    pub jf_gcmap: *const u64,
    /// Front-end savedata (GCREF).
    pub jf_savedata: i64,
    /// Exception from GUARD_(NO)_EXCEPTION / GUARD_NOT_FORCED.
    /// GCREF slot — traced by GC.
    pub jf_guard_exc: i64,
    /// Forwarding pointer for GC compaction.
    pub jf_forward: *mut JitFrame,
    /// Length of the jf_frame array (kept here, not in frame_info,
    /// because frame_info may change while frame is alive).
    pub jf_frame_len: usize,
    // jf_frame: [i64; jf_frame_len] follows immediately in memory.
}

/// Byte size of the JitFrame fixed header (before jf_frame array).
pub const JITFRAME_HEADER_SIZE: usize = std::mem::size_of::<JitFrame>();

/// Byte offset of each fixed field, for Cranelift codegen.
pub const JF_FRAME_INFO_OFS: usize = std::mem::offset_of!(JitFrame, jf_frame_info);
pub const JF_DESCR_OFS: usize = std::mem::offset_of!(JitFrame, jf_descr);
pub const JF_FORCE_DESCR_OFS: usize = std::mem::offset_of!(JitFrame, jf_force_descr);
pub const JF_GCMAP_OFS: usize = std::mem::offset_of!(JitFrame, jf_gcmap);
pub const JF_SAVEDATA_OFS: usize = std::mem::offset_of!(JitFrame, jf_savedata);
pub const JF_GUARD_EXC_OFS: usize = std::mem::offset_of!(JitFrame, jf_guard_exc);
pub const JF_FORWARD_OFS: usize = std::mem::offset_of!(JitFrame, jf_forward);
pub const JF_FRAME_LEN_OFS: usize = std::mem::offset_of!(JitFrame, jf_frame_len);

/// Byte offset of the first element of jf_frame (the variable-length array).
/// jitframe.py:99 — BASEITEMOFS
pub const JF_FRAME_BASE_OFS: usize = JITFRAME_HEADER_SIZE;

impl JitFrame {
    /// Total allocation size for a jitframe with `depth` slots.
    pub fn alloc_size(depth: usize) -> usize {
        JITFRAME_HEADER_SIZE + depth * SIGN_SIZE
    }

    /// Initialize a freshly-allocated JitFrame at `ptr`.
    ///
    /// # Safety
    /// `ptr` must point to at least `alloc_size(depth)` bytes of memory.
    pub unsafe fn init(ptr: *mut JitFrame, info: *const JitFrameInfo, depth: usize) {
        let frame = &mut *ptr;
        frame.jf_frame_info = info;
        frame.jf_descr = 0;
        frame.jf_force_descr = 0;
        frame.jf_gcmap = std::ptr::null();
        frame.jf_savedata = 0;
        frame.jf_guard_exc = 0;
        frame.jf_forward = std::ptr::null_mut();
        frame.jf_frame_len = depth;
        // Zero the jf_frame array slots
        let slots = Self::frame_slots_mut(ptr, depth);
        slots.fill(0);
    }

    /// Get a mutable slice of the jf_frame array.
    ///
    /// # Safety
    /// `ptr` must be a valid JitFrame with at least `len` trailing slots.
    pub unsafe fn frame_slots_mut(ptr: *mut JitFrame, len: usize) -> &'static mut [i64] {
        let base = (ptr as *mut u8).add(JF_FRAME_BASE_OFS) as *mut i64;
        std::slice::from_raw_parts_mut(base, len)
    }

    /// Get an immutable slice of the jf_frame array.
    pub unsafe fn frame_slots(ptr: *const JitFrame, len: usize) -> &'static [i64] {
        let base = (ptr as *const u8).add(JF_FRAME_BASE_OFS) as *const i64;
        std::slice::from_raw_parts(base, len)
    }

    /// jitframe.py:54-57 — resolve forwarding chain.
    pub unsafe fn resolve(mut frame: *mut JitFrame) -> *mut JitFrame {
        while !(*frame).jf_forward.is_null() {
            frame = (*frame).jf_forward;
        }
        frame
    }
}

/// GC trace callback for JitFrame.
///
/// jitframe.py:104-136 — traces fixed GCREF fields then walks
/// the gcmap bitmap to find Ref-typed jf_frame slots.
///
/// `trace_ref` is called for each GCREF slot that the GC needs to update.
pub unsafe fn jitframe_trace(frame_ptr: *mut JitFrame, mut trace_ref: impl FnMut(*mut i64)) {
    let frame = &*frame_ptr;

    // Trace fixed GCREF header fields
    trace_ref(&mut (*(frame_ptr)).jf_descr);
    trace_ref(&mut (*(frame_ptr)).jf_force_descr);
    trace_ref(&mut (*(frame_ptr)).jf_savedata);
    trace_ref(&mut (*(frame_ptr)).jf_guard_exc);
    trace_ref(&mut (*(frame_ptr)).jf_forward as *mut *mut JitFrame as *mut i64);

    // Walk gcmap bitmap to trace jf_frame Ref slots
    let gcmap = frame.jf_gcmap;
    if gcmap.is_null() {
        return;
    }
    // gcmap is an Array(Unsigned) — first word is the length
    let gcmap_len = *(gcmap as *const usize);
    let gcmap_data = gcmap.add(1); // skip length field

    let frame_base = (frame_ptr as *mut u8).add(JF_FRAME_BASE_OFS) as *mut i64;
    let frame_len = frame.jf_frame_len;

    for word_idx in 0..gcmap_len {
        let cur = *gcmap_data.add(word_idx);
        if cur == 0 {
            continue;
        }
        for bit in 0..64u32 {
            if cur & (1u64 << bit) != 0 {
                let slot_idx = word_idx * 64 + bit as usize;
                debug_assert!(slot_idx < frame_len, "bogus gcmap bit");
                if slot_idx < frame_len {
                    trace_ref(frame_base.add(slot_idx));
                }
            }
        }
    }
}
