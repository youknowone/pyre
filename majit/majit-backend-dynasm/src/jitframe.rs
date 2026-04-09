//! jitframe.py / llsupport.llmodel JITFRAME layout for dynasm backend.
//!
//! This mirrors the RPython fixed header layout closely enough for dynasm
//! code generation and runtime execution paths.

pub const SIZEOFSIGNED: usize = std::mem::size_of::<isize>();
pub const NULLGCMAP: *const u8 = std::ptr::null();

#[repr(C)]
pub struct JitFrameInfo {
    pub jfi_frame_depth: isize,
    pub jfi_frame_size: isize,
}

#[repr(C)]
pub struct JitFrame {
    pub jf_frame_info: *const JitFrameInfo,
    pub jf_descr: usize,
    pub jf_force_descr: usize,
    pub jf_gcmap: *const u8,
    pub jf_savedata: usize,
    pub jf_guard_exc: usize,
    pub jf_forward: *mut JitFrame,
}

pub const JF_FRAME_INFO_OFS: i32 = std::mem::offset_of!(JitFrame, jf_frame_info) as i32;
pub const JF_DESCR_OFS: i32 = std::mem::offset_of!(JitFrame, jf_descr) as i32;
pub const JF_FORCE_DESCR_OFS: i32 = std::mem::offset_of!(JitFrame, jf_force_descr) as i32;
pub const JF_GCMAP_OFS: i32 = std::mem::offset_of!(JitFrame, jf_gcmap) as i32;
pub const JF_SAVEDATA_OFS: i32 = std::mem::offset_of!(JitFrame, jf_savedata) as i32;
pub const JF_GUARD_EXC_OFS: i32 = std::mem::offset_of!(JitFrame, jf_guard_exc) as i32;
pub const JF_FORWARD_OFS: i32 = std::mem::offset_of!(JitFrame, jf_forward) as i32;

pub const LENGTHOFS: usize = 0;
pub const BASEITEMOFS: usize = SIZEOFSIGNED;
pub const JF_FRAME_OFS: usize = std::mem::size_of::<JitFrame>();
pub const FIRST_ITEM_OFFSET: usize = JF_FRAME_OFS + BASEITEMOFS;

impl JitFrame {
    pub fn alloc_size(depth: usize) -> usize {
        FIRST_ITEM_OFFSET + depth * SIZEOFSIGNED
    }

    pub unsafe fn init(ptr: *mut JitFrame, depth: usize) {
        (*ptr).jf_frame_info = std::ptr::null();
        (*ptr).jf_descr = 0;
        (*ptr).jf_force_descr = 0;
        (*ptr).jf_gcmap = NULLGCMAP;
        (*ptr).jf_savedata = 0;
        (*ptr).jf_guard_exc = 0;
        (*ptr).jf_forward = std::ptr::null_mut();
        let len_ptr = (ptr as *mut u8).add(JF_FRAME_OFS + LENGTHOFS) as *mut isize;
        *len_ptr = depth as isize;
    }

    pub unsafe fn get_latest_descr(ptr: *const JitFrame) -> usize {
        (*ptr).jf_descr
    }

    pub unsafe fn set_latest_descr(ptr: *mut JitFrame, descr: usize) {
        (*ptr).jf_descr = descr;
    }

    pub unsafe fn set_gcmap(ptr: *mut JitFrame, gcmap: *const u8) {
        (*ptr).jf_gcmap = gcmap;
    }

    pub unsafe fn slot_ptr(ptr: *mut JitFrame, index: usize) -> *mut i64 {
        (ptr as *mut u8).add(FIRST_ITEM_OFFSET + index * SIZEOFSIGNED) as *mut i64
    }

    pub unsafe fn slot_ptr_const(ptr: *const JitFrame, index: usize) -> *const i64 {
        (ptr as *const u8).add(FIRST_ITEM_OFFSET + index * SIZEOFSIGNED) as *const i64
    }

    pub unsafe fn get_int_value(ptr: *const JitFrame, index: usize) -> i64 {
        *Self::slot_ptr_const(ptr, index)
    }

    pub unsafe fn set_int_value(ptr: *mut JitFrame, index: usize, value: i64) {
        *Self::slot_ptr(ptr, index) = value;
    }
}
