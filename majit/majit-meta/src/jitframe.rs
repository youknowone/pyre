//! JitFrame — RPython `rpython/jit/backend/llsupport/jitframe.py` 1:1 port.
//!
//! A flat, GC-managed frame used by compiled code with a variable-length
//! `jf_frame: Array(Signed)` trailing array.

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
    /// jitframe.py:18-22
    pub fn update_depth(&mut self, base_ofs: isize, new_depth: isize) {
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
// Offset constants for Cranelift codegen.
pub const JF_FRAME_INFO_OFS: i32 = std::mem::offset_of!(JitFrame, jf_frame_info) as i32;
pub const JF_DESCR_OFS: i32 = std::mem::offset_of!(JitFrame, jf_descr) as i32;
pub const JF_FORCE_DESCR_OFS: i32 = std::mem::offset_of!(JitFrame, jf_force_descr) as i32;
pub const JF_GCMAP_OFS: i32 = std::mem::offset_of!(JitFrame, jf_gcmap) as i32;
pub const JF_SAVEDATA_OFS: i32 = std::mem::offset_of!(JitFrame, jf_savedata) as i32;
pub const JF_GUARD_EXC_OFS: i32 = std::mem::offset_of!(JitFrame, jf_guard_exc) as i32;
pub const JF_FORWARD_OFS: i32 = std::mem::offset_of!(JitFrame, jf_forward) as i32;

// jitframe.py:97-101
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

/// SIGN_SIZE = sizeof(Signed)
pub const SIGN_SIZE: usize = SIZEOFSIGNED;
/// UNSIGN_SIZE = sizeof(Unsigned)
pub const UNSIGN_SIZE: usize = std::mem::size_of::<usize>();

// jitframe.py:138
pub type JitFramePtr = *mut JitFrame;

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
        // RPython: frame.jf_frame_info = frame_info
        // (other fields are zero from malloc)
        (*ptr).jf_frame_info = info;
        // Write the jf_frame array length
        let len_ptr = (ptr as *mut u8).add(JF_FRAME_OFS) as *mut isize;
        *len_ptr = depth as isize;
    }

    /// Get a mutable slice of the jf_frame items (excluding length field).
    ///
    /// # Safety
    /// `ptr` must be a valid JitFrame with at least `len` trailing slots.
    pub unsafe fn frame_slots_mut(ptr: *mut JitFrame, len: usize) -> &'static mut [isize] {
        let base = (ptr as *mut u8).add(JF_FRAME_OFS + BASEITEMOFS) as *mut isize;
        std::slice::from_raw_parts_mut(base, len)
    }

    /// Get an immutable slice of the jf_frame items.
    pub unsafe fn frame_slots(ptr: *const JitFrame, len: usize) -> &'static [isize] {
        let base = (ptr as *const u8).add(JF_FRAME_OFS + BASEITEMOFS) as *const isize;
        std::slice::from_raw_parts(base, len)
    }

    /// Read the jf_frame array length.
    pub unsafe fn frame_length(ptr: *const JitFrame) -> isize {
        let len_ptr = (ptr as *const u8).add(JF_FRAME_OFS + LENGTHOFS) as *const isize;
        *len_ptr
    }

    /// jitframe.py:54-57 — jitframe_resolve.
    pub unsafe fn resolve(mut frame: *mut JitFrame) -> *mut JitFrame {
        while !(*frame).jf_forward.is_null() {
            frame = (*frame).jf_forward;
        }
        frame
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
pub unsafe fn jitframe_trace(obj_addr: *mut JitFrame, mut trace_callback: impl FnMut(*mut usize)) {
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
                debug_assert!((index as isize) < frame_lgt, "bogus frame field get");
                // jitframe.py:131-133 — trace the slot
                let slot_addr = (obj_addr as *mut u8)
                    .add(JF_FRAME_OFS + BASEITEMOFS + SIGN_SIZE * index)
                    as *mut usize;
                trace_callback(slot_addr);
            }
            bitindex += 1;
        }
        no += 1;
    }
}
