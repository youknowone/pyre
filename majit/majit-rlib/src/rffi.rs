//! `rpython/rtyper/lltypesystem/rffi.py` — C types, `llexternal`, and the
//! errno flags an external call saves.
//!
//! `ExternalCompilationInfo` is `rpython/translator/tool/cbuild.py`.

#![allow(non_camel_case_types)]

pub use majit_jitcode::rffi::{
    RFFI_ALT_ERRNO, RFFI_ERR_ALL, RFFI_ERR_NONE, RFFI_FULL_ERRNO, RFFI_FULL_ERRNO_ZERO,
    RFFI_FULL_LASTERROR, RFFI_READSAVED_ERRNO, RFFI_READSAVED_LASTERROR, RFFI_SAVE_ERRNO,
    RFFI_SAVE_LASTERROR, RFFI_SAVE_WSALASTERROR, RFFI_ZERO_ERRNO_BEFORE,
};
pub use majit_macros::{
    call_aroundstate_target, external_compilation_info, jit_close_stack, llexternal,
};

/// `rffi.CHAR`.
pub type CHAR = std::ffi::c_char;
/// `rffi.UCHAR`.
pub type UCHAR = std::ffi::c_uchar;
/// `rffi.SHORT`.
pub type SHORT = std::ffi::c_short;
/// `rffi.USHORT`.
pub type USHORT = std::ffi::c_ushort;
/// `rffi.INT`.
pub type INT = std::ffi::c_int;
/// `rffi.UINT`.
pub type UINT = std::ffi::c_uint;
/// `rffi.LONG`.
pub type LONG = std::ffi::c_long;
/// `rffi.ULONG`.
pub type ULONG = std::ffi::c_ulong;
/// `rffi.LONGLONG`.
pub type LONGLONG = std::ffi::c_longlong;
/// `rffi.ULONGLONG`.
pub type ULONGLONG = std::ffi::c_ulonglong;
/// `rffi.SIZE_T`.
pub type SIZE_T = usize;
/// `rffi.SSIZE_T`.
pub type SSIZE_T = isize;
/// `rffi.SIGNED` (`lltype.Signed`).
pub type SIGNED = isize;
/// `rffi.UNSIGNED` (`lltype.Unsigned`).
pub type UNSIGNED = usize;
/// `rffi.DOUBLE`.
pub type DOUBLE = std::ffi::c_double;
/// `rffi.FLOAT`.
pub type FLOAT = std::ffi::c_float;
/// `rffi.VOIDP`.
pub type VOIDP = *mut std::ffi::c_void;
/// `rffi.CONST_VOIDP`.
pub type CONST_VOIDP = *const std::ffi::c_void;
/// `rffi.CCHARP`.
pub type CCHARP = *mut std::ffi::c_char;
/// `rffi.CONST_CCHARP`.
pub type CONST_CCHARP = *const std::ffi::c_char;
/// `rffi.CCHARPP`.
pub type CCHARPP = *mut *mut std::ffi::c_char;

/// `ExternalCompilationInfo` (`cbuild.py`).
///
/// Slices stay `'static` so a declaration is a const.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExternalCompilationInfo {
    pub pre_include_bits: &'static [&'static str],
    pub includes: &'static [&'static str],
    pub include_dirs: &'static [&'static str],
    pub post_include_bits: &'static [&'static str],
    pub libraries: &'static [&'static str],
    pub library_dirs: &'static [&'static str],
    pub separate_module_sources: &'static [&'static str],
    pub separate_module_files: &'static [&'static str],
    pub compile_extra: &'static [&'static str],
    pub link_extra: &'static [&'static str],
    pub frameworks: &'static [&'static str],
    pub link_files: &'static [&'static str],
    pub testonly_libraries: &'static [&'static str],
    pub use_cpp_linker: bool,
}

impl ExternalCompilationInfo {
    pub const DEFAULT: Self = Self {
        pre_include_bits: &[],
        includes: &[],
        include_dirs: &[],
        post_include_bits: &[],
        libraries: &[],
        library_dirs: &[],
        separate_module_sources: &[],
        separate_module_files: &[],
        compile_extra: &[],
        link_extra: &[],
        frameworks: &[],
        link_files: &[],
        testonly_libraries: &[],
        use_cpp_linker: false,
    };

    pub const fn new() -> Self {
        Self::DEFAULT
    }
}

impl Default for ExternalCompilationInfo {
    fn default() -> Self {
        Self::DEFAULT
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    static GIL_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    static HELD_DURING: AtomicBool = AtomicBool::new(false);

    struct GilHold;
    impl Drop for GilHold {
        fn drop(&mut self) {
            if majit_gc::rgil::am_i_holding_the_gil() {
                majit_gc::rgil::release();
            }
        }
    }

    fn hold_gil() -> (std::sync::MutexGuard<'static, ()>, GilHold) {
        let guard = GIL_LOCK.lock().unwrap();
        majit_gc::rgil::allocate();
        if !majit_gc::rgil::am_i_holding_the_gil() {
            majit_gc::rgil::acquire();
        }
        (guard, GilHold)
    }

    #[unsafe(no_mangle)]
    unsafe extern "C" fn pyre_rffi_t1_gil_probe() -> i32 {
        HELD_DURING.store(majit_gc::rgil::am_i_holding_the_gil(), Ordering::SeqCst);
        7
    }

    #[unsafe(no_mangle)]
    unsafe extern "C" fn pyre_rffi_t1_set_errno(value: i32) -> i32 {
        #[cfg(any(target_os = "macos", target_os = "ios", target_os = "freebsd"))]
        unsafe {
            *libc::__error() = value;
        }
        #[cfg(any(target_os = "linux", target_os = "android"))]
        unsafe {
            *libc::__errno_location() = value;
        }
        #[cfg(windows)]
        unsafe {
            *_errno() = value;
        }
        value
    }

    #[cfg(windows)]
    unsafe extern "C" {
        fn _errno() -> *mut i32;
    }

    external_compilation_info! {
        const EMPTY_ECI = {};
    }

    #[cfg(unix)]
    external_compilation_info! {
        const LIBC_ECI = {
            libraries: ["c"],
            includes: ["errno.h"],
        };
    }

    llexternal!(
        gil_probe = "pyre_rffi_t1_gil_probe",
        [],
        i32,
        compilation_info = EMPTY_ECI
    );
    llexternal!(
        gil_probe_direct = "pyre_rffi_t1_gil_probe",
        [],
        i32,
        _nowrapper = true
    );
    llexternal!(
        gil_probe_macro = "pyre_rffi_t1_gil_probe",
        [],
        i32,
        macro = pyre_rffi_t1_gil_probe,
        sandboxsafe = true
    );
    llexternal!(
        set_errno_ext = "pyre_rffi_t1_set_errno",
        [i32],
        i32,
        save_err = RFFI_SAVE_ERRNO
    );
    llexternal!(
        pub pub_gil_probe = "pyre_rffi_t1_gil_probe",
        [],
        i32,
        sandboxsafe = true
    );
    #[cfg(unix)]
    llexternal!(
        fcntl_int = "fcntl",
        [INT, INT, INT],
        INT,
        natural_arity = 2,
        save_err = RFFI_SAVE_ERRNO
    );

    #[test]
    fn gil_is_released_during_the_call_and_held_after() {
        let (_lock, _hold) = hold_gil();
        assert!(majit_gc::rgil::am_i_holding_the_gil());
        let value = unsafe { gil_probe() };
        assert_eq!(value, 7);
        assert!(!HELD_DURING.load(Ordering::SeqCst));
        assert!(majit_gc::rgil::am_i_holding_the_gil());
    }

    #[test]
    fn nowrapper_calls_the_funcptr_without_releasing_the_gil() {
        let (_lock, _hold) = hold_gil();
        let value = unsafe { gil_probe_direct() };
        assert_eq!(value, 7);
        assert!(HELD_DURING.load(Ordering::SeqCst));
        assert!(majit_gc::rgil::am_i_holding_the_gil());
    }

    #[test]
    fn macro_path_calls_that_function() {
        let (_lock, _hold) = hold_gil();
        let value = unsafe { gil_probe_macro() };
        assert_eq!(value, 7);
        assert!(HELD_DURING.load(Ordering::SeqCst));
    }

    #[test]
    fn save_errno_is_visible_to_get_saved_errno() {
        let (_lock, _hold) = hold_gil();
        crate::rposix::set_saved_errno(0);
        let value = unsafe { set_errno_ext(123) };
        assert_eq!(value, 123);
        assert_eq!(crate::rposix::get_saved_errno(), 123);
        assert!(majit_gc::rgil::am_i_holding_the_gil());
    }

    #[cfg(unix)]
    #[test]
    fn variadic_fcntl_matches_libc_and_saves_ebadf() {
        let (_lock, _hold) = hold_gil();
        let mut fds = [0; 2];
        assert_eq!(unsafe { libc::pipe(fds.as_mut_ptr()) }, 0);
        let expected = unsafe { libc::fcntl(fds[0], libc::F_GETFD) };
        assert!(expected >= 0);
        assert_eq!(unsafe { fcntl_int(fds[0], libc::F_GETFD, 0) }, expected);
        let with_cloexec = expected | libc::FD_CLOEXEC;
        assert_eq!(unsafe { fcntl_int(fds[0], libc::F_SETFD, with_cloexec) }, 0);
        assert_eq!(unsafe { libc::fcntl(fds[0], libc::F_GETFD) }, with_cloexec);
        assert_eq!(unsafe { libc::close(fds[0]) }, 0);
        crate::rposix::set_saved_errno(0);
        assert!(unsafe { fcntl_int(fds[0], libc::F_GETFD, 0) } < 0);
        assert_eq!(crate::rposix::get_saved_errno(), libc::EBADF);
        unsafe { libc::close(fds[1]) };
    }

    #[test]
    fn empty_compilation_info_starts_empty() {
        assert!(EMPTY_ECI.libraries.is_empty());
        assert!(!EMPTY_ECI.use_cpp_linker);
        assert_eq!(
            ExternalCompilationInfo::new(),
            ExternalCompilationInfo::DEFAULT
        );
        #[cfg(unix)]
        {
            assert_eq!(LIBC_ECI.libraries, ["c"]);
            assert_eq!(LIBC_ECI.includes, ["errno.h"]);
        }
    }
}

#[cfg(test)]
#[test]
fn pub_wrapper_is_visible_from_the_parent() {
    let _probe: unsafe fn() -> i32 = tests::pub_gil_probe;
}
