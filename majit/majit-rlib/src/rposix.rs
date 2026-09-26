//! `rpython/rlib/rposix.py` — the saved `errno` an external call reads and
//! writes around itself (`llexternal(..., save_err=...)`).

use crate::rthread;
use majit_jitcode::rffi::{
    RFFI_ALT_ERRNO, RFFI_READSAVED_ERRNO, RFFI_READSAVED_LASTERROR, RFFI_SAVE_ERRNO,
    RFFI_SAVE_LASTERROR, RFFI_SAVE_WSALASTERROR, RFFI_ZERO_ERRNO_BEFORE,
};

#[cfg(target_os = "windows")]
use crate::{_rsocket_rffi, rwin32};

/// `rposix._get_errno`.
pub fn _get_errno() -> i32 {
    unsafe { *rthread::tlfield_p_errno() }
}

/// `rposix._set_errno`.
pub fn _set_errno(errno: i32) {
    unsafe { *rthread::tlfield_p_errno() = errno };
}

/// `rposix.get_saved_errno`.
pub fn get_saved_errno() -> i32 {
    rthread::tlfield_getraw_int(rthread::TLFIELD_RPY_ERRNO_OFS)
}

/// `rposix.set_saved_errno`.
pub fn set_saved_errno(errno: i32) {
    rthread::tlfield_setraw_int(rthread::TLFIELD_RPY_ERRNO_OFS, errno);
}

/// `rposix.get_saved_alterrno`.
pub fn get_saved_alterrno() -> i32 {
    rthread::tlfield_getraw_int(rthread::TLFIELD_ALT_ERRNO_OFS)
}

/// `rposix.set_saved_alterrno`.
pub fn set_saved_alterrno(errno: i32) {
    rthread::tlfield_setraw_int(rthread::TLFIELD_ALT_ERRNO_OFS, errno);
}

/// `rposix._errno_before`.
pub fn _errno_before(save_err: i64) {
    if save_err & RFFI_READSAVED_ERRNO != 0 {
        if save_err & RFFI_ALT_ERRNO != 0 {
            _set_errno(rthread::tlfield_getraw_int(rthread::TLFIELD_ALT_ERRNO_OFS));
        } else {
            _set_errno(rthread::tlfield_getraw_int(rthread::TLFIELD_RPY_ERRNO_OFS));
        }
    } else if save_err & RFFI_ZERO_ERRNO_BEFORE != 0 {
        _set_errno(0);
    }
    #[cfg(target_os = "windows")]
    if save_err & RFFI_READSAVED_LASTERROR != 0 {
        let err = if save_err & RFFI_ALT_ERRNO != 0 {
            rthread::tlfield_getraw_int(rthread::TLFIELD_ALT_LASTERROR_OFS)
        } else {
            rthread::tlfield_getraw_int(rthread::TLFIELD_RPY_LASTERROR_OFS)
        };
        // careful, getraw() overwrites GetLastError.
        // We must assign it with _SetLastError() as the last
        // operation, i.e. after the errno handling.
        unsafe { rwin32::_SetLastError(err as u32) };
    }
    #[cfg(not(target_os = "windows"))]
    let _ = RFFI_READSAVED_LASTERROR;
}

/// `rposix._errno_after`.
pub fn _errno_after(save_err: i64) {
    #[cfg(target_os = "windows")]
    {
        let err = if save_err & RFFI_SAVE_LASTERROR != 0 {
            // careful, setraw() overwrites GetLastError.
            // We must read it first, before the errno handling.
            Some(unsafe { rwin32::_GetLastError() } as i32)
        } else if save_err & RFFI_SAVE_WSALASTERROR != 0 {
            Some(unsafe { _rsocket_rffi::_WSAGetLastError() })
        } else {
            None
        };
        if let Some(err) = err {
            let ofs = if save_err & RFFI_ALT_ERRNO != 0 {
                rthread::TLFIELD_ALT_LASTERROR_OFS
            } else {
                rthread::TLFIELD_RPY_LASTERROR_OFS
            };
            rthread::tlfield_setraw_int(ofs, err);
        }
    }
    #[cfg(not(target_os = "windows"))]
    let _ = (RFFI_SAVE_LASTERROR, RFFI_SAVE_WSALASTERROR);
    if save_err & RFFI_SAVE_ERRNO != 0 {
        let ofs = if save_err & RFFI_ALT_ERRNO != 0 {
            rthread::TLFIELD_ALT_ERRNO_OFS
        } else {
            rthread::TLFIELD_RPY_ERRNO_OFS
        };
        rthread::tlfield_setraw_int(ofs, _get_errno());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use majit_jitcode::rffi::RFFI_ERR_ALL;

    #[test]
    fn alt_errno_round_trips_through_the_real_errno() {
        set_saved_alterrno(42);
        _errno_before(RFFI_ERR_ALL | RFFI_ALT_ERRNO);
        assert_eq!(_get_errno(), 42);
        _set_errno(7);
        _errno_after(RFFI_ERR_ALL | RFFI_ALT_ERRNO);
        assert_eq!(get_saved_alterrno(), 7);
        assert_eq!(get_saved_errno(), 0);
    }

    #[test]
    fn zero_errno_before_clears_the_real_errno() {
        _set_errno(5);
        _errno_before(RFFI_ZERO_ERRNO_BEFORE);
        assert_eq!(_get_errno(), 0);
    }
}
