//! `rpython/rlib/rwin32.py` — the Win32 last-error accessors an external
//! call saves and restores around itself.

#[link(name = "kernel32")]
unsafe extern "system" {
    #[link_name = "GetLastError"]
    pub fn _GetLastError() -> u32;
    #[link_name = "SetLastError"]
    pub fn _SetLastError(code: u32);
}

use crate::rthread;

/// `rwin32.GetLastError_saved` — the last error an external declared
/// `save_err=RFFI_SAVE_LASTERROR` saved.
pub fn GetLastError_saved() -> i64 {
    // extra cast to LONG to match CPython behaviour
    rthread::tlfield_getraw_int(rthread::TLFIELD_RPY_LASTERROR_OFS) as i64
}

/// `rwin32.SetLastError_saved`.
pub fn SetLastError_saved(err: u32) {
    rthread::tlfield_setraw_int(rthread::TLFIELD_RPY_LASTERROR_OFS, err as i32);
}

/// `rwin32.GetLastError_alt_saved` — the last error an external declared
/// `save_err=RFFI_SAVE_LASTERROR | RFFI_ALT_ERRNO` saved.
pub fn GetLastError_alt_saved() -> i64 {
    // extra cast to LONG to match CPython behaviour
    rthread::tlfield_getraw_int(rthread::TLFIELD_ALT_LASTERROR_OFS) as i64
}

/// `rwin32.SetLastError_alt_saved`.
pub fn SetLastError_alt_saved(err: u32) {
    rthread::tlfield_setraw_int(rthread::TLFIELD_ALT_LASTERROR_OFS, err as i32);
}
