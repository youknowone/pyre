//! JitFrame — RPython `rpython/jit/backend/llsupport/jitframe.py` port
//! for the dynasm backend.

pub const SIZEOFSIGNED: usize = std::mem::size_of::<isize>();
pub type GcMap = usize;
pub const NULLGCMAP: *const GcMap = std::ptr::null();

#[repr(C)]
pub struct JitFrameInfo {
    pub jfi_frame_depth: isize,
    pub jfi_frame_size: isize,
}

pub const JITFRAMEINFO_SIZE: usize = 2 * SIZEOFSIGNED;
pub const NULLFRAMEINFO: *const JitFrameInfo = std::ptr::null();
pub type JitFrameInfoPtr = *const JitFrameInfo;

#[repr(C)]
pub struct JitFrame {
    pub jf_frame_info: *const JitFrameInfo,
    pub jf_descr: usize,
    pub jf_force_descr: usize,
    pub jf_gcmap: *const GcMap,
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

pub const GCMAPLENGTHOFS: usize = 0;
pub const GCMAPBASEOFS: usize = SIZEOFSIGNED;
pub const LENGTHOFS: usize = 0;
pub const BASEITEMOFS: usize = SIZEOFSIGNED;
pub const JF_FRAME_OFS: usize = std::mem::size_of::<JitFrame>();
pub const FIRST_ITEM_OFFSET: usize = JF_FRAME_OFS + BASEITEMOFS;
pub const SIGN_SIZE: usize = SIZEOFSIGNED;
pub const UNSIGN_SIZE: usize = std::mem::size_of::<usize>();

impl JitFrame {
    pub fn alloc_size(depth: usize) -> usize {
        JF_FRAME_OFS + SIZEOFSIGNED * (1 + depth)
    }

    pub unsafe fn init(ptr: *mut JitFrame, info: *const JitFrameInfo, depth: usize) {
        unsafe {
            (*ptr).jf_frame_info = info;
            let len_ptr = (ptr as *mut u8).add(JF_FRAME_OFS + LENGTHOFS) as *mut isize;
            *len_ptr = depth as isize;
        }
    }

    pub unsafe fn frame_length(ptr: *const JitFrame) -> isize {
        unsafe {
            let len_ptr = (ptr as *const u8).add(JF_FRAME_OFS + LENGTHOFS) as *const isize;
            *len_ptr
        }
    }

    pub unsafe fn resolve(mut frame: *mut JitFrame) -> *mut JitFrame {
        unsafe {
            while !(*frame).jf_forward.is_null() {
                frame = (*frame).jf_forward;
            }
        }
        frame
    }

    pub unsafe fn get_latest_descr(ptr: *const JitFrame) -> usize {
        unsafe { (*ptr).jf_descr }
    }

    pub unsafe fn set_latest_descr(ptr: *mut JitFrame, descr: usize) {
        unsafe {
            (*ptr).jf_descr = descr;
        }
    }

    pub unsafe fn set_gcmap(ptr: *mut JitFrame, gcmap: *const GcMap) {
        unsafe {
            (*ptr).jf_gcmap = gcmap;
        }
    }

    pub unsafe fn slot_ptr(ptr: *mut JitFrame, index: usize) -> *mut i64 {
        unsafe { (ptr as *mut u8).add(FIRST_ITEM_OFFSET + index * SIZEOFSIGNED) as *mut i64 }
    }

    pub unsafe fn slot_ptr_const(ptr: *const JitFrame, index: usize) -> *const i64 {
        unsafe { (ptr as *const u8).add(FIRST_ITEM_OFFSET + index * SIZEOFSIGNED) as *const i64 }
    }

    pub unsafe fn get_int_value(ptr: *const JitFrame, index: usize) -> i64 {
        unsafe { *Self::slot_ptr_const(ptr, index) }
    }

    pub unsafe fn set_int_value(ptr: *mut JitFrame, index: usize, value: i64) {
        unsafe {
            *Self::slot_ptr(ptr, index) = value;
        }
    }
}
