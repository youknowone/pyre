//! `rpython/rlib/rposix.py` — the saved `errno` an external call reads and
//! writes around itself (`llexternal(..., save_err=...)`).

use crate::rthread;
use majit_jitcode::rffi::{
    RFFI_ALT_ERRNO, RFFI_FULL_ERRNO_ZERO, RFFI_READSAVED_ERRNO, RFFI_READSAVED_LASTERROR,
    RFFI_SAVE_ERRNO, RFFI_SAVE_LASTERROR, RFFI_SAVE_WSALASTERROR, RFFI_ZERO_ERRNO_BEFORE,
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

// `rposix.c_ioctl_voidp`.
//
// `sys.platform == 'darwin'` sets `natural_arity=2`. This records only
// `sys/ioctl.h`; the rest of `rposix.eci` is not ported here.
// `gnu_time_bits64` on `ioctl` names the 32-bit redirect `__ioctl_time64`.
// Native targets are 64-bit, so that `link_name` is not copied.
#[cfg(unix)]
crate::rffi::external_compilation_info! {
    const IOCTL_ECI = {
        includes: ["sys/ioctl.h"],
    };
}

#[cfg(all(unix, target_os = "macos"))]
crate::rffi::llexternal!(
    pub c_ioctl_voidp = "ioctl",
    [crate::rffi::INT, crate::rffi::UINT, crate::rffi::VOIDP],
    crate::rffi::INT,
    compilation_info = IOCTL_ECI,
    save_err = RFFI_SAVE_ERRNO,
    natural_arity = 2
);

#[cfg(all(unix, not(target_os = "macos")))]
crate::rffi::llexternal!(
    pub c_ioctl_voidp = "ioctl",
    [crate::rffi::INT, crate::rffi::UINT, crate::rffi::VOIDP],
    crate::rffi::INT,
    compilation_info = IOCTL_ECI,
    save_err = RFFI_SAVE_ERRNO
);

// `rposix.external` for the fd and path calls below. `c_open` on darwin is
// variadic (`natural_arity=2`, mode is `rffi.INT`); everywhere else the mode
// is `rffi.MODE_T`. `c_close` is `releasegil=False`. `c_lseek` is
// `macro=_MACRO_ON_POSIX` (true here), so the call is `libc::lseek`, which
// already carries the 64-bit `off_t` symbol. `gnu_file_offset_bits64`
// redirects (`open64`, `lseek64`) are 32-bit and are not copied.
// `c_access` and `c_isatty` leave `save_err` at `RFFI_ERR_NONE`. `c_mkdir`'s
// mode is `rffi.MODE_T`. `c_strerror` is `releasegil=False`. `c_readdir` is
// `macro=True` with `save_err=RFFI_FULL_ERRNO_ZERO`, so the dirent matches
// `libc::readdir` and errno is cleared before the call. `c_closedir` is
// `releasegil=False`. `c_opendir` on macOS x86_64 links `opendir$INODE64`
// (x86 adds `$UNIX2003`). This include list is the headers these calls need,
// not the whole `rposix.eci`.
#[cfg(unix)]
crate::rffi::external_compilation_info! {
    const POSIX_ECI = {
        includes: [
            "fcntl.h",
            "unistd.h",
            "sys/types.h",
            "sys/stat.h",
            "dirent.h",
            "string.h",
            "signal.h",
            "stdio.h",
        ],
    };
}

#[cfg(all(unix, target_os = "macos"))]
crate::rffi::llexternal!(
    #[cfg_attr(target_arch = "x86", link_name = "open$UNIX2003")]
    pub c_open = "open",
    [*const libc::c_char, crate::rffi::INT, crate::rffi::INT],
    crate::rffi::INT,
    compilation_info = POSIX_ECI,
    save_err = RFFI_SAVE_ERRNO,
    natural_arity = 2
);

#[cfg(all(unix, not(target_os = "macos")))]
crate::rffi::llexternal!(
    pub c_open = "open",
    [*const libc::c_char, crate::rffi::INT, libc::mode_t],
    crate::rffi::INT,
    compilation_info = POSIX_ECI,
    save_err = RFFI_SAVE_ERRNO
);

#[cfg(unix)]
crate::rffi::llexternal!(
    #[cfg_attr(
        all(target_os = "macos", target_arch = "x86"),
        link_name = "read$UNIX2003"
    )]
    pub c_read = "read",
    [crate::rffi::INT, crate::rffi::VOIDP, crate::rffi::SIZE_T],
    crate::rffi::SSIZE_T,
    compilation_info = POSIX_ECI,
    save_err = RFFI_SAVE_ERRNO
);

#[cfg(unix)]
crate::rffi::llexternal!(
    #[cfg_attr(
        all(target_os = "macos", target_arch = "x86"),
        link_name = "write$UNIX2003"
    )]
    pub c_write = "write",
    [crate::rffi::INT, crate::rffi::VOIDP, crate::rffi::SIZE_T],
    crate::rffi::SSIZE_T,
    compilation_info = POSIX_ECI,
    save_err = RFFI_SAVE_ERRNO
);

#[cfg(unix)]
crate::rffi::llexternal!(
    #[cfg_attr(
        all(target_os = "macos", target_arch = "x86"),
        link_name = "close$NOCANCEL$UNIX2003"
    )]
    #[cfg_attr(
        all(target_os = "macos", target_arch = "x86_64"),
        link_name = "close$NOCANCEL"
    )]
    pub c_close = "close",
    [crate::rffi::INT],
    crate::rffi::INT,
    compilation_info = POSIX_ECI,
    releasegil = false,
    save_err = RFFI_SAVE_ERRNO
);

#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_lseek = "lseek",
    [crate::rffi::INT, crate::rffi::LONGLONG, crate::rffi::INT],
    crate::rffi::LONGLONG,
    compilation_info = POSIX_ECI,
    save_err = RFFI_SAVE_ERRNO,
    macro = libc::lseek
);

#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_dup = "dup",
    [crate::rffi::INT],
    crate::rffi::INT,
    compilation_info = POSIX_ECI,
    save_err = RFFI_SAVE_ERRNO
);

#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_dup2 = "dup2",
    [crate::rffi::INT, crate::rffi::INT],
    crate::rffi::INT,
    compilation_info = POSIX_ECI,
    save_err = RFFI_SAVE_ERRNO
);

#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_access = "access",
    [*const libc::c_char, crate::rffi::INT],
    crate::rffi::INT,
    compilation_info = POSIX_ECI
);

#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_isatty = "isatty",
    [crate::rffi::INT],
    crate::rffi::INT,
    compilation_info = POSIX_ECI
);

#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_unlink = "unlink",
    [*const libc::c_char],
    crate::rffi::INT,
    compilation_info = POSIX_ECI,
    save_err = RFFI_SAVE_ERRNO
);

#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_mkdir = "mkdir",
    [*const libc::c_char, libc::mode_t],
    crate::rffi::INT,
    compilation_info = POSIX_ECI,
    save_err = RFFI_SAVE_ERRNO
);

// `rposix_stat.c_fstat` / `c_stat` / `c_lstat` are `macro=True` and save
// errno. The call is `libc::fstat` / `libc::stat` / `libc::lstat`, which
// already carry the inode64 symbol. Includes are
// `rposix_stat.compilation_info` (`sys/statvfs.h` is in that list; these
// three calls do not use it).
#[cfg(unix)]
crate::rffi::external_compilation_info! {
    const STAT_ECI = {
        includes: ["sys/types.h", "sys/stat.h", "sys/statvfs.h", "unistd.h"],
    };
}

#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_fstat = "fstat",
    [crate::rffi::INT, *mut libc::stat],
    crate::rffi::INT,
    compilation_info = STAT_ECI,
    save_err = RFFI_SAVE_ERRNO,
    macro = libc::fstat
);

#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_stat = "stat",
    [*const libc::c_char, *mut libc::stat],
    crate::rffi::INT,
    compilation_info = STAT_ECI,
    save_err = RFFI_SAVE_ERRNO,
    macro = libc::stat
);

#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_lstat = "lstat",
    [*const libc::c_char, *mut libc::stat],
    crate::rffi::INT,
    compilation_info = STAT_ECI,
    save_err = RFFI_SAVE_ERRNO,
    macro = libc::lstat
);

#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_getcwd = "getcwd",
    [*mut libc::c_char, crate::rffi::SIZE_T],
    *mut libc::c_char,
    compilation_info = POSIX_ECI,
    save_err = RFFI_SAVE_ERRNO
);

#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_strerror = "strerror",
    [crate::rffi::INT],
    *mut libc::c_char,
    compilation_info = POSIX_ECI,
    releasegil = false
);

#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_getuid = "getuid",
    [],
    libc::uid_t,
    compilation_info = POSIX_ECI
);

#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_geteuid = "geteuid",
    [],
    libc::uid_t,
    compilation_info = POSIX_ECI
);

#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_getgid = "getgid",
    [],
    libc::gid_t,
    compilation_info = POSIX_ECI
);

#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_getegid = "getegid",
    [],
    libc::gid_t,
    compilation_info = POSIX_ECI
);

#[cfg(unix)]
crate::rffi::llexternal!(
    #[cfg_attr(
        all(target_os = "macos", target_arch = "x86_64"),
        link_name = "opendir$INODE64"
    )]
    #[cfg_attr(
        all(target_os = "macos", target_arch = "x86"),
        link_name = "opendir$INODE64$UNIX2003"
    )]
    pub c_opendir = "opendir",
    [*const libc::c_char],
    *mut libc::DIR,
    compilation_info = POSIX_ECI,
    save_err = RFFI_SAVE_ERRNO
);

#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_readdir = "readdir",
    [*mut libc::DIR],
    *mut libc::dirent,
    compilation_info = POSIX_ECI,
    save_err = RFFI_FULL_ERRNO_ZERO,
    macro = libc::readdir
);

#[cfg(unix)]
crate::rffi::llexternal!(
    #[cfg_attr(
        all(target_os = "macos", target_arch = "x86"),
        link_name = "closedir$UNIX2003"
    )]
    pub c_closedir = "closedir",
    [*mut libc::DIR],
    crate::rffi::INT,
    compilation_info = POSIX_ECI,
    releasegil = false
);

// `rposix.c_pread`, `c_pwrite`, `c_lockf`, `c_fsync`, `c_fdatasync`,
// `c_ftruncate`, `c_sync`, `c_chdir`, `c_fchdir`, `c_readlink`, `c_rmdir`.
// `c_ftruncate` is `macro=_MACRO_ON_POSIX` (`_os_support._MACRO_ON_POSIX`),
// so the call is `libc::ftruncate`, which already carries the 64-bit `off_t`
// symbol. `gnu_file_offset_bits64` redirects (`pread64`, `pwrite64`,
// `ftruncate64`, `lockf64`) are 32-bit and are not copied. `c_pread` and
// `c_pwrite` on macOS x86 link `pread$UNIX2003` / `pwrite$UNIX2003`.
// `c_fsync` on macOS x86 links `fsync$UNIX2003`. `c_sync` returns void and
// leaves `save_err` at `RFFI_ERR_NONE`. `POSIX_ECI` already lists the
// headers these calls need.
#[cfg(unix)]
crate::rffi::llexternal!(
    #[cfg_attr(
        all(target_os = "macos", target_arch = "x86"),
        link_name = "pread$UNIX2003"
    )]
    pub c_pread = "pread",
    [
        crate::rffi::INT,
        crate::rffi::VOIDP,
        crate::rffi::SIZE_T,
        libc::off_t
    ],
    crate::rffi::SSIZE_T,
    compilation_info = POSIX_ECI,
    save_err = RFFI_SAVE_ERRNO
);

#[cfg(unix)]
crate::rffi::llexternal!(
    #[cfg_attr(
        all(target_os = "macos", target_arch = "x86"),
        link_name = "pwrite$UNIX2003"
    )]
    pub c_pwrite = "pwrite",
    [
        crate::rffi::INT,
        crate::rffi::VOIDP,
        crate::rffi::SIZE_T,
        libc::off_t
    ],
    crate::rffi::SSIZE_T,
    compilation_info = POSIX_ECI,
    save_err = RFFI_SAVE_ERRNO
);

#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_lockf = "lockf",
    [crate::rffi::INT, crate::rffi::INT, libc::off_t],
    crate::rffi::INT,
    compilation_info = POSIX_ECI,
    save_err = RFFI_SAVE_ERRNO
);

#[cfg(unix)]
crate::rffi::llexternal!(
    #[cfg_attr(
        all(target_os = "macos", target_arch = "x86"),
        link_name = "fsync$UNIX2003"
    )]
    pub c_fsync = "fsync",
    [crate::rffi::INT],
    crate::rffi::INT,
    compilation_info = POSIX_ECI,
    save_err = RFFI_SAVE_ERRNO
);

#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_fdatasync = "fdatasync",
    [crate::rffi::INT],
    crate::rffi::INT,
    compilation_info = POSIX_ECI,
    save_err = RFFI_SAVE_ERRNO
);

#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_ftruncate = "ftruncate",
    [crate::rffi::INT, crate::rffi::LONGLONG],
    crate::rffi::INT,
    compilation_info = POSIX_ECI,
    save_err = RFFI_SAVE_ERRNO,
    macro = libc::ftruncate
);

#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_sync = "sync",
    [],
    (),
    compilation_info = POSIX_ECI
);

#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_chdir = "chdir",
    [*const libc::c_char],
    crate::rffi::INT,
    compilation_info = POSIX_ECI,
    save_err = RFFI_SAVE_ERRNO
);

#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_fchdir = "fchdir",
    [crate::rffi::INT],
    crate::rffi::INT,
    compilation_info = POSIX_ECI,
    save_err = RFFI_SAVE_ERRNO
);

#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_readlink = "readlink",
    [*const libc::c_char, *mut libc::c_char, crate::rffi::SIZE_T],
    crate::rffi::SSIZE_T,
    compilation_info = POSIX_ECI,
    save_err = RFFI_SAVE_ERRNO
);

#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_rmdir = "rmdir",
    [*const libc::c_char],
    crate::rffi::INT,
    compilation_info = POSIX_ECI,
    save_err = RFFI_SAVE_ERRNO
);

// `rposix.c_getpid` and `c_getppid` are `releasegil=False` and save errno.
// `c_setsid`, `c_getsid`, `c_getpgid`, `c_setpgid`, `c_getpgrp`, and
// `c_setpgrp` save errno. `GETPGRP_HAVE_ARG` and `SETPGRP_HAVE_ARG` are
// false here, so both pgrp calls take no argument. `c_setuid`,
// `c_seteuid`, `c_setgid`, `c_setegid`, `c_setreuid`, and `c_setregid`
// save errno. `c_kill` and `c_killpg` save errno; `signal.h` is the
// header those two need. `c_getgroups` on Apple links
// `getgroups$DARWIN_EXTSN`, the unlimited alias `<unistd.h>` selects
// under `_DARWIN_C_SOURCE`. `c_setgroups` saves errno.
#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_getpid = "getpid",
    [],
    libc::pid_t,
    compilation_info = POSIX_ECI,
    releasegil = false,
    save_err = RFFI_SAVE_ERRNO
);

#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_getppid = "getppid",
    [],
    libc::pid_t,
    compilation_info = POSIX_ECI,
    releasegil = false,
    save_err = RFFI_SAVE_ERRNO
);

#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_setsid = "setsid",
    [],
    libc::pid_t,
    compilation_info = POSIX_ECI,
    save_err = RFFI_SAVE_ERRNO
);

#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_getsid = "getsid",
    [libc::pid_t],
    libc::pid_t,
    compilation_info = POSIX_ECI,
    save_err = RFFI_SAVE_ERRNO
);

#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_getpgid = "getpgid",
    [libc::pid_t],
    libc::pid_t,
    compilation_info = POSIX_ECI,
    save_err = RFFI_SAVE_ERRNO
);

#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_setpgid = "setpgid",
    [libc::pid_t, libc::pid_t],
    crate::rffi::INT,
    compilation_info = POSIX_ECI,
    save_err = RFFI_SAVE_ERRNO
);

#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_getpgrp = "getpgrp",
    [],
    crate::rffi::INT,
    compilation_info = POSIX_ECI,
    save_err = RFFI_SAVE_ERRNO
);

#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_setpgrp = "setpgrp",
    [],
    crate::rffi::INT,
    compilation_info = POSIX_ECI,
    save_err = RFFI_SAVE_ERRNO
);

#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_setuid = "setuid",
    [libc::uid_t],
    crate::rffi::INT,
    compilation_info = POSIX_ECI,
    save_err = RFFI_SAVE_ERRNO
);

#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_seteuid = "seteuid",
    [libc::uid_t],
    crate::rffi::INT,
    compilation_info = POSIX_ECI,
    save_err = RFFI_SAVE_ERRNO
);

#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_setgid = "setgid",
    [libc::gid_t],
    crate::rffi::INT,
    compilation_info = POSIX_ECI,
    save_err = RFFI_SAVE_ERRNO
);

#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_setegid = "setegid",
    [libc::gid_t],
    crate::rffi::INT,
    compilation_info = POSIX_ECI,
    save_err = RFFI_SAVE_ERRNO
);

#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_setreuid = "setreuid",
    [libc::uid_t, libc::uid_t],
    crate::rffi::INT,
    compilation_info = POSIX_ECI,
    save_err = RFFI_SAVE_ERRNO
);

#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_setregid = "setregid",
    [libc::gid_t, libc::gid_t],
    crate::rffi::INT,
    compilation_info = POSIX_ECI,
    save_err = RFFI_SAVE_ERRNO
);

#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_kill = "kill",
    [libc::pid_t, crate::rffi::INT],
    crate::rffi::INT,
    compilation_info = POSIX_ECI,
    save_err = RFFI_SAVE_ERRNO
);

#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_killpg = "killpg",
    [crate::rffi::INT, crate::rffi::INT],
    crate::rffi::INT,
    compilation_info = POSIX_ECI,
    save_err = RFFI_SAVE_ERRNO
);

#[cfg(unix)]
crate::rffi::llexternal!(
    #[cfg_attr(target_vendor = "apple", link_name = "getgroups$DARWIN_EXTSN")]
    pub c_getgroups = "getgroups",
    [crate::rffi::INT, *mut libc::gid_t],
    crate::rffi::INT,
    compilation_info = POSIX_ECI,
    save_err = RFFI_SAVE_ERRNO
);

#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_setgroups = "setgroups",
    [crate::rffi::SIZE_T, *const libc::gid_t],
    crate::rffi::INT,
    compilation_info = POSIX_ECI,
    save_err = RFFI_SAVE_ERRNO
);

// `rposix.c_nice` uses `RFFI_FULL_ERRNO_ZERO`: errno is cleared before the
// call because -1 is also a successful niceness. `rposix.c_ctermid` takes a
// null buffer and does not save errno; `<stdio.h>` declares it.
// `rposix.c_tcgetpgrp` returns a pid and saves errno. `rposix.c_tcsetpgrp`
// saves errno.
#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_nice = "nice",
    [crate::rffi::INT],
    crate::rffi::INT,
    compilation_info = POSIX_ECI,
    save_err = RFFI_FULL_ERRNO_ZERO
);

#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_ctermid = "ctermid",
    [*mut libc::c_char],
    *mut libc::c_char,
    compilation_info = POSIX_ECI
);

#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_tcgetpgrp = "tcgetpgrp",
    [crate::rffi::INT],
    libc::pid_t,
    compilation_info = POSIX_ECI,
    save_err = RFFI_SAVE_ERRNO
);

#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_tcsetpgrp = "tcsetpgrp",
    [crate::rffi::INT, libc::pid_t],
    crate::rffi::INT,
    compilation_info = POSIX_ECI,
    save_err = RFFI_SAVE_ERRNO
);

// `rposix.c_chmod`, `rposix.c_fchmod`, and `rposix.c_mkfifo` save errno.
// `rposix.c_mknod` is `macro=_MACRO_ON_POSIX` and saves errno. Its device
// argument is `dev_t`; `rposix.c_mknod` spells that parameter `rffi.INT`.
// `rposix.c_umask` returns the previous mask and does not save errno.
#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_chmod = "chmod",
    [*const libc::c_char, libc::mode_t],
    crate::rffi::INT,
    compilation_info = POSIX_ECI,
    save_err = RFFI_SAVE_ERRNO
);

#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_fchmod = "fchmod",
    [crate::rffi::INT, libc::mode_t],
    crate::rffi::INT,
    compilation_info = POSIX_ECI,
    save_err = RFFI_SAVE_ERRNO
);

#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_mkfifo = "mkfifo",
    [*const libc::c_char, libc::mode_t],
    crate::rffi::INT,
    compilation_info = POSIX_ECI,
    save_err = RFFI_SAVE_ERRNO
);

#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_mknod = "mknod",
    [*const libc::c_char, libc::mode_t, libc::dev_t],
    crate::rffi::INT,
    compilation_info = POSIX_ECI,
    save_err = RFFI_SAVE_ERRNO,
    macro = libc::mknod
);

#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_umask = "umask",
    [libc::mode_t],
    libc::mode_t,
    compilation_info = POSIX_ECI
);

// `rposix.c_link` and `rposix.c_symlink` save errno.
#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_link = "link",
    [*const libc::c_char, *const libc::c_char],
    crate::rffi::INT,
    compilation_info = POSIX_ECI,
    save_err = RFFI_SAVE_ERRNO
);

#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_symlink = "symlink",
    [*const libc::c_char, *const libc::c_char],
    crate::rffi::INT,
    compilation_info = POSIX_ECI,
    save_err = RFFI_SAVE_ERRNO
);

// `rposix.c_chown`, `rposix.c_lchown`, and `rposix.c_fchown` save errno.
// Uid and gid are `rffi.INT`. `-1` leaves that id unchanged.
#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_chown = "chown",
    [*const libc::c_char, crate::rffi::INT, crate::rffi::INT],
    crate::rffi::INT,
    compilation_info = POSIX_ECI,
    save_err = RFFI_SAVE_ERRNO
);

#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_lchown = "lchown",
    [*const libc::c_char, crate::rffi::INT, crate::rffi::INT],
    crate::rffi::INT,
    compilation_info = POSIX_ECI,
    save_err = RFFI_SAVE_ERRNO
);

#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_fchown = "fchown",
    [crate::rffi::INT, crate::rffi::INT, crate::rffi::INT],
    crate::rffi::INT,
    compilation_info = POSIX_ECI,
    save_err = RFFI_SAVE_ERRNO
);

// `rposix.c_futimens` and `rposix.c_utimensat` save errno. The time
// argument is `TIMESPEC2P`, a pointer to two `struct timespec` values.
#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_futimens = "futimens",
    [crate::rffi::INT, *const libc::timespec],
    crate::rffi::INT,
    compilation_info = POSIX_ECI,
    save_err = RFFI_SAVE_ERRNO
);

#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_utimensat = "utimensat",
    [
        crate::rffi::INT,
        *const libc::c_char,
        *const libc::timespec,
        crate::rffi::INT
    ],
    crate::rffi::INT,
    compilation_info = POSIX_ECI,
    save_err = RFFI_SAVE_ERRNO
);

// `rposix.c_pipe` saves errno. The argument is an array of two ints.
#[cfg(unix)]
crate::rffi::llexternal!(
    pub c_pipe = "pipe",
    [*mut libc::c_int],
    crate::rffi::INT,
    compilation_info = POSIX_ECI,
    save_err = RFFI_SAVE_ERRNO
);

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

    #[cfg(unix)]
    #[test]
    fn c_open_missing_path_saves_enoent() {
        let path = c"/no/such/pyre-rffi-open";
        let fd = unsafe { c_open(path.as_ptr(), libc::O_RDONLY, 0) };
        assert!(fd < 0);
        assert_eq!(get_saved_errno(), libc::ENOENT);
    }

    #[cfg(unix)]
    #[test]
    fn c_read_write_lseek_close_round_trip() {
        use std::os::unix::ffi::OsStrExt;
        let path = std::env::temp_dir().join(format!("pyre-rffi-posix-{}", std::process::id()));
        let c_path = std::ffi::CString::new(path.as_os_str().as_bytes()).unwrap();
        let fd = unsafe {
            c_open(
                c_path.as_ptr(),
                libc::O_CREAT | libc::O_RDWR | libc::O_TRUNC,
                0o600,
            )
        };
        assert!(fd >= 0, "c_open errno {}", get_saved_errno());
        let msg = b"hi";
        let wrote = unsafe { c_write(fd, msg.as_ptr() as *mut libc::c_void, msg.len()) };
        assert_eq!(wrote, 2, "c_write errno {}", get_saved_errno());
        assert_eq!(unsafe { c_lseek(fd, 0, libc::SEEK_SET) }, 0);
        let mut buf = [0u8; 2];
        let got = unsafe { c_read(fd, buf.as_mut_ptr().cast(), buf.len()) };
        assert_eq!(got, 2, "c_read errno {}", get_saved_errno());
        assert_eq!(&buf, b"hi");
        assert_eq!(
            unsafe { c_close(fd) },
            0,
            "c_close errno {}",
            get_saved_errno()
        );
        let _ = std::fs::remove_file(&path);
    }

    #[cfg(unix)]
    #[test]
    fn c_dup_reads_what_c_write_wrote() {
        use std::os::unix::ffi::OsStrExt;
        let path = std::env::temp_dir().join(format!("pyre-rffi-dup-{}", std::process::id()));
        let c_path = std::ffi::CString::new(path.as_os_str().as_bytes()).unwrap();
        let fd = unsafe {
            c_open(
                c_path.as_ptr(),
                libc::O_CREAT | libc::O_RDWR | libc::O_TRUNC,
                0o600,
            )
        };
        assert!(fd >= 0, "c_open errno {}", get_saved_errno());
        let msg = b"dup";
        assert_eq!(
            unsafe { c_write(fd, msg.as_ptr() as *mut libc::c_void, msg.len()) },
            3
        );
        let duped = unsafe { c_dup(fd) };
        assert!(duped >= 0, "c_dup errno {}", get_saved_errno());
        assert_eq!(unsafe { c_lseek(duped, 0, libc::SEEK_SET) }, 0);
        let mut buf = [0u8; 3];
        assert_eq!(
            unsafe { c_read(duped, buf.as_mut_ptr().cast(), buf.len()) },
            3
        );
        assert_eq!(&buf, b"dup");
        assert!(unsafe { c_isatty(fd) } == 0);
        let sink = unsafe { c_open(c"/dev/null".as_ptr(), libc::O_RDWR, 0) };
        assert!(sink >= 0, "c_open /dev/null errno {}", get_saved_errno());
        assert_eq!(
            unsafe { c_dup2(fd, sink) },
            sink,
            "c_dup2 errno {}",
            get_saved_errno()
        );
        assert_eq!(unsafe { c_lseek(sink, 0, libc::SEEK_SET) }, 0);
        let mut via_dup2 = [0u8; 3];
        assert_eq!(
            unsafe { c_read(sink, via_dup2.as_mut_ptr().cast(), via_dup2.len()) },
            3
        );
        assert_eq!(&via_dup2, b"dup");
        assert_eq!(unsafe { c_close(sink) }, 0);
        assert_eq!(unsafe { c_close(duped) }, 0);
        assert_eq!(unsafe { c_close(fd) }, 0);
        let _ = std::fs::remove_file(&path);
    }

    #[cfg(unix)]
    #[test]
    fn c_stat_lstat_access_mkdir_unlink() {
        use std::os::unix::ffi::OsStrExt;
        let dir = std::env::temp_dir().join(format!("pyre-rffi-stat-{}", std::process::id()));
        let c_dir = std::ffi::CString::new(dir.as_os_str().as_bytes()).unwrap();
        assert_eq!(
            unsafe { c_mkdir(c_dir.as_ptr(), 0o700) },
            0,
            "c_mkdir errno {}",
            get_saved_errno()
        );
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        assert_eq!(unsafe { c_stat(c_dir.as_ptr(), &mut st) }, 0);
        assert_eq!(st.st_mode & libc::S_IFMT, libc::S_IFDIR);
        assert_eq!(unsafe { c_access(c_dir.as_ptr(), libc::F_OK) }, 0);

        let missing = c"/no/such/pyre-rffi-access";
        assert!(unsafe { c_access(missing.as_ptr(), libc::F_OK) } != 0);

        let file = dir.join("f");
        let c_file = std::ffi::CString::new(file.as_os_str().as_bytes()).unwrap();
        let fd = unsafe {
            c_open(
                c_file.as_ptr(),
                libc::O_CREAT | libc::O_RDWR | libc::O_TRUNC,
                0o600,
            )
        };
        assert!(fd >= 0, "c_open errno {}", get_saved_errno());
        assert_eq!(
            unsafe { c_write(fd, b"hi".as_ptr() as *mut libc::c_void, 2) },
            2
        );
        let mut fst: libc::stat = unsafe { std::mem::zeroed() };
        assert_eq!(unsafe { c_fstat(fd, &mut fst) }, 0);
        assert_eq!(fst.st_size, 2 as libc::off_t);
        assert_eq!(unsafe { c_close(fd) }, 0);

        let link = dir.join("l");
        std::os::unix::fs::symlink(&file, &link).unwrap();
        let c_link = std::ffi::CString::new(link.as_os_str().as_bytes()).unwrap();
        let mut lst: libc::stat = unsafe { std::mem::zeroed() };
        assert_eq!(unsafe { c_lstat(c_link.as_ptr(), &mut lst) }, 0);
        assert_eq!(lst.st_mode & libc::S_IFMT, libc::S_IFLNK);
        let mut followed: libc::stat = unsafe { std::mem::zeroed() };
        assert_eq!(unsafe { c_stat(c_link.as_ptr(), &mut followed) }, 0);
        assert_eq!(followed.st_size, 2 as libc::off_t);

        assert_eq!(unsafe { c_unlink(c_file.as_ptr()) }, 0);
        assert!(unsafe { c_stat(c_file.as_ptr(), &mut followed) } < 0);
        assert_eq!(get_saved_errno(), libc::ENOENT);
        let _ = std::fs::remove_file(&link);
        let _ = std::fs::remove_dir(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn c_getcwd_erange_then_fits() {
        let mut tiny = [0u8; 1];
        let res = unsafe { c_getcwd(tiny.as_mut_ptr().cast(), tiny.len()) };
        assert!(res.is_null());
        assert_eq!(get_saved_errno(), libc::ERANGE);
        let mut buf = vec![0u8; 4096];
        let res = unsafe { c_getcwd(buf.as_mut_ptr().cast(), buf.len()) };
        assert!(!res.is_null());
        let bytes = unsafe { std::ffi::CStr::from_ptr(res) }.to_bytes();
        assert!(!bytes.is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn c_strerror_and_ids_match_libc() {
        let msg = unsafe { c_strerror(libc::ENOENT) };
        assert!(!msg.is_null());
        assert!(
            !unsafe { std::ffi::CStr::from_ptr(msg) }
                .to_bytes()
                .is_empty()
        );
        assert_eq!(unsafe { c_getuid() }, unsafe { libc::getuid() });
        assert_eq!(unsafe { c_geteuid() }, unsafe { libc::geteuid() });
        assert_eq!(unsafe { c_getgid() }, unsafe { libc::getgid() });
        assert_eq!(unsafe { c_getegid() }, unsafe { libc::getegid() });
    }

    #[cfg(unix)]
    #[test]
    fn c_opendir_readdir_closedir_lists_the_file() {
        use std::os::unix::ffi::OsStrExt;
        let dir = std::env::temp_dir().join(format!("pyre-rffi-dir-{}", std::process::id()));
        let c_dir = std::ffi::CString::new(dir.as_os_str().as_bytes()).unwrap();
        assert_eq!(unsafe { c_mkdir(c_dir.as_ptr(), 0o700) }, 0);
        let file = dir.join("f");
        let c_file = std::ffi::CString::new(file.as_os_str().as_bytes()).unwrap();
        let fd = unsafe {
            c_open(
                c_file.as_ptr(),
                libc::O_CREAT | libc::O_RDWR | libc::O_TRUNC,
                0o600,
            )
        };
        assert!(fd >= 0, "c_open errno {}", get_saved_errno());
        assert_eq!(unsafe { c_close(fd) }, 0);

        let dirp = unsafe { c_opendir(c_dir.as_ptr()) };
        assert!(!dirp.is_null(), "c_opendir errno {}", get_saved_errno());
        let mut saw = false;
        loop {
            let ent = unsafe { c_readdir(dirp) };
            if ent.is_null() {
                assert_eq!(get_saved_errno(), 0);
                break;
            }
            let name = unsafe { std::ffi::CStr::from_ptr((*ent).d_name.as_ptr()) }.to_bytes();
            if name == b"f" {
                saw = true;
            }
        }
        assert!(saw);
        assert_eq!(unsafe { c_closedir(dirp) }, 0);

        let missing = unsafe { c_opendir(c"/no/such/pyre-rffi-opendir".as_ptr()) };
        assert!(missing.is_null());
        assert_eq!(get_saved_errno(), libc::ENOENT);
        let _ = std::fs::remove_file(&file);
        let _ = std::fs::remove_dir(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn c_pread_pwrite_fsync_truncate_lockf_readlink_rmdir_chdir() {
        use std::os::unix::ffi::OsStrExt;
        let dir = std::env::temp_dir().join(format!("pyre-rffi-io-{}", std::process::id()));
        let c_dir = std::ffi::CString::new(dir.as_os_str().as_bytes()).unwrap();
        assert_eq!(
            unsafe { c_mkdir(c_dir.as_ptr(), 0o700) },
            0,
            "c_mkdir errno {}",
            get_saved_errno()
        );

        let file = dir.join("f");
        let c_file = std::ffi::CString::new(file.as_os_str().as_bytes()).unwrap();
        let fd = unsafe {
            c_open(
                c_file.as_ptr(),
                libc::O_CREAT | libc::O_RDWR | libc::O_TRUNC,
                0o600,
            )
        };
        assert!(fd >= 0, "c_open errno {}", get_saved_errno());
        assert_eq!(
            unsafe { c_pwrite(fd, b"abcd".as_ptr() as *mut libc::c_void, 4, 0) },
            4,
            "c_pwrite errno {}",
            get_saved_errno()
        );
        assert_eq!(
            unsafe { c_fsync(fd) },
            0,
            "c_fsync errno {}",
            get_saved_errno()
        );
        assert_eq!(
            unsafe { c_fdatasync(fd) },
            0,
            "c_fdatasync errno {}",
            get_saved_errno()
        );
        let mut buf = [0u8; 2];
        assert_eq!(
            unsafe { c_pread(fd, buf.as_mut_ptr().cast(), buf.len(), 1) },
            2,
            "c_pread errno {}",
            get_saved_errno()
        );
        assert_eq!(&buf, b"bc");
        assert_eq!(
            unsafe { c_ftruncate(fd, 1) },
            0,
            "c_ftruncate errno {}",
            get_saved_errno()
        );
        let mut one = [0u8; 2];
        assert_eq!(
            unsafe { c_pread(fd, one.as_mut_ptr().cast(), one.len(), 0) },
            1
        );
        assert_eq!(one[0], b'a');
        assert_eq!(
            unsafe { c_lockf(fd, libc::F_TLOCK, 0) },
            0,
            "c_lockf errno {}",
            get_saved_errno()
        );
        assert_eq!(unsafe { c_lockf(fd, libc::F_ULOCK, 0) }, 0);
        assert_eq!(unsafe { c_close(fd) }, 0);

        let link = dir.join("l");
        std::os::unix::fs::symlink("f", &link).unwrap();
        let c_link = std::ffi::CString::new(link.as_os_str().as_bytes()).unwrap();
        let mut name = [0u8; 8];
        let n = unsafe { c_readlink(c_link.as_ptr(), name.as_mut_ptr().cast(), name.len()) };
        assert_eq!(n, 1, "c_readlink errno {}", get_saved_errno());
        assert_eq!(&name[..n as usize], b"f");
        assert!(unsafe { c_readlink(c_file.as_ptr(), name.as_mut_ptr().cast(), name.len()) } < 0);
        assert_eq!(get_saved_errno(), libc::EINVAL);

        assert!(unsafe { c_chdir(c"/no/such/pyre-rffi-chdir".as_ptr()) } < 0);
        assert_eq!(get_saved_errno(), libc::ENOENT);

        let orig = std::env::current_dir().unwrap();
        let c_orig = std::ffi::CString::new(orig.as_os_str().as_bytes()).unwrap();
        struct Back(std::ffi::CString);
        impl Drop for Back {
            fn drop(&mut self) {
                unsafe { c_chdir(self.0.as_ptr()) };
            }
        }
        let _back = Back(c_orig);
        assert_eq!(
            unsafe { c_chdir(c_dir.as_ptr()) },
            0,
            "c_chdir errno {}",
            get_saved_errno()
        );
        let here = unsafe { c_open(c".".as_ptr(), libc::O_RDONLY, 0) };
        assert!(here >= 0, "c_open . errno {}", get_saved_errno());
        let parent = unsafe { c_open(c"..".as_ptr(), libc::O_RDONLY, 0) };
        assert!(parent >= 0, "c_open .. errno {}", get_saved_errno());
        assert_eq!(
            unsafe { c_fchdir(parent) },
            0,
            "c_fchdir errno {}",
            get_saved_errno()
        );
        assert_eq!(unsafe { c_fchdir(here) }, 0);
        assert_eq!(unsafe { c_close(here) }, 0);
        assert_eq!(unsafe { c_close(parent) }, 0);
        assert_eq!(unsafe { c_chdir(_back.0.as_ptr()) }, 0);

        let _ = std::fs::remove_file(&link);
        let _ = std::fs::remove_file(&file);
        assert_eq!(
            unsafe { c_rmdir(c_dir.as_ptr()) },
            0,
            "c_rmdir errno {}",
            get_saved_errno()
        );
        assert!(unsafe { c_rmdir(c_dir.as_ptr()) } < 0);
        assert_eq!(get_saved_errno(), libc::ENOENT);
    }

    #[cfg(unix)]
    #[test]
    fn c_process_ids_kill_and_credentials() {
        let pid = unsafe { c_getpid() };
        assert_eq!(pid, unsafe { libc::getpid() });
        assert_eq!(unsafe { c_getppid() }, unsafe { libc::getppid() });
        assert_eq!(unsafe { c_getpgrp() } as libc::pid_t, unsafe {
            libc::getpgrp()
        });
        assert_eq!(unsafe { c_getpgid(0) }, unsafe { libc::getpgid(0) });
        assert_eq!(unsafe { c_getsid(0) }, unsafe { libc::getsid(0) });
        // A negative group id is `EINVAL` on Linux and macOS. A negative
        // pid is `ESRCH` on macOS, so the probe uses this process's pid.
        assert!(unsafe { c_setpgid(pid, -1) } < 0);
        assert_eq!(get_saved_errno(), libc::EINVAL);
        assert_eq!(
            unsafe { c_kill(pid, 0) },
            0,
            "c_kill errno {}",
            get_saved_errno()
        );
        assert_eq!(
            unsafe { c_killpg(c_getpgrp(), 0) },
            0,
            "c_killpg errno {}",
            get_saved_errno()
        );
        assert!(unsafe { c_kill(pid, -1) } < 0);
        assert_eq!(get_saved_errno(), libc::EINVAL);
        let n = unsafe { c_getgroups(0, std::ptr::null_mut()) };
        assert!(n >= 0, "c_getgroups errno {}", get_saved_errno());
        assert!(unsafe { c_setgroups(usize::MAX, std::ptr::null()) } < 0);
        let saved = get_saved_errno();
        assert!(
            saved == libc::EINVAL || saved == libc::EPERM,
            "c_setgroups errno {saved}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn c_nice_ctermid_and_tc_pgrp() {
        // `nice(0)` reports the current niceness and leaves it unchanged.
        // -1 with a cleared errno is that value, not a failure.
        let before = unsafe { libc::nice(0) };
        let got = unsafe { c_nice(0) };
        if got == -1 {
            assert_eq!(get_saved_errno(), 0, "c_nice errno");
        }
        assert_eq!(got, before);
        let name = unsafe { c_ctermid(std::ptr::null_mut()) };
        assert!(!name.is_null(), "c_ctermid returned null");
        let bytes = unsafe { std::ffi::CStr::from_ptr(name) }.to_bytes();
        assert!(!bytes.is_empty(), "c_ctermid empty");
        assert!(unsafe { c_tcgetpgrp(-1) } < 0);
        assert_eq!(get_saved_errno(), libc::EBADF);
        assert!(unsafe { c_tcsetpgrp(-1, 0) } < 0);
        assert_eq!(get_saved_errno(), libc::EBADF);
    }

    #[cfg(unix)]
    #[test]
    fn c_chmod_fchmod_mkfifo_mknod_and_umask() {
        use std::os::unix::ffi::OsStrExt;
        let dir = std::env::temp_dir().join(format!("pyre-rffi-mode-{}", std::process::id()));
        let c_dir = std::ffi::CString::new(dir.as_os_str().as_bytes()).unwrap();
        assert_eq!(
            unsafe { c_mkdir(c_dir.as_ptr(), 0o700) },
            0,
            "c_mkdir errno {}",
            get_saved_errno()
        );
        let file = dir.join("f");
        let c_file = std::ffi::CString::new(file.as_os_str().as_bytes()).unwrap();
        let fd = unsafe {
            c_open(
                c_file.as_ptr(),
                libc::O_CREAT | libc::O_RDWR | libc::O_TRUNC,
                0o600,
            )
        };
        assert!(fd >= 0, "c_open errno {}", get_saved_errno());
        assert_eq!(
            unsafe { c_chmod(c_file.as_ptr(), 0o640) },
            0,
            "c_chmod errno {}",
            get_saved_errno()
        );
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        assert_eq!(unsafe { c_stat(c_file.as_ptr(), &mut st) }, 0);
        assert_eq!(st.st_mode & 0o777, 0o640);
        assert_eq!(
            unsafe { c_fchmod(fd, 0o600) },
            0,
            "c_fchmod errno {}",
            get_saved_errno()
        );
        assert_eq!(unsafe { c_stat(c_file.as_ptr(), &mut st) }, 0);
        assert_eq!(st.st_mode & 0o777, 0o600);
        assert_eq!(unsafe { c_close(fd) }, 0);
        assert!(unsafe { c_chmod(c"/no/such/pyre-rffi-chmod".as_ptr(), 0o600) } < 0);
        assert_eq!(get_saved_errno(), libc::ENOENT);
        assert!(unsafe { c_fchmod(-1, 0o600) } < 0);
        assert_eq!(get_saved_errno(), libc::EBADF);

        let fifo = dir.join("p");
        let c_fifo = std::ffi::CString::new(fifo.as_os_str().as_bytes()).unwrap();
        assert_eq!(
            unsafe { c_mkfifo(c_fifo.as_ptr(), 0o600) },
            0,
            "c_mkfifo errno {}",
            get_saved_errno()
        );
        assert_eq!(unsafe { c_stat(c_fifo.as_ptr(), &mut st) }, 0);
        assert_eq!(st.st_mode & libc::S_IFMT, libc::S_IFIFO);
        assert_eq!(unsafe { c_unlink(c_fifo.as_ptr()) }, 0);

        let node = dir.join("n");
        let c_node = std::ffi::CString::new(node.as_os_str().as_bytes()).unwrap();
        assert_eq!(
            unsafe { c_mknod(c_node.as_ptr(), libc::S_IFIFO | 0o600, 0) },
            0,
            "c_mknod errno {}",
            get_saved_errno()
        );
        assert_eq!(unsafe { c_stat(c_node.as_ptr(), &mut st) }, 0);
        assert_eq!(st.st_mode & libc::S_IFMT, libc::S_IFIFO);
        assert_eq!(unsafe { c_unlink(c_node.as_ptr()) }, 0);
        assert!(
            unsafe {
                c_mknod(
                    c"/no/such/pyre-rffi-mknod".as_ptr(),
                    libc::S_IFIFO | 0o600,
                    0,
                )
            } < 0
        );
        assert_eq!(get_saved_errno(), libc::ENOENT);

        let prev = unsafe { libc::umask(0) };
        let _ = unsafe { libc::umask(prev) };
        let seen = unsafe { c_umask(0o027) };
        let restored = unsafe { c_umask(prev) };
        assert_eq!(seen, prev);
        assert_eq!(restored, 0o027);

        let _ = std::fs::remove_file(&file);
        assert_eq!(unsafe { c_rmdir(c_dir.as_ptr()) }, 0);
    }

    #[cfg(unix)]
    #[test]
    fn c_link_and_symlink() {
        use std::os::unix::ffi::OsStrExt;
        let dir = std::env::temp_dir().join(format!("pyre-rffi-link-{}", std::process::id()));
        let c_dir = std::ffi::CString::new(dir.as_os_str().as_bytes()).unwrap();
        assert_eq!(
            unsafe { c_mkdir(c_dir.as_ptr(), 0o700) },
            0,
            "c_mkdir errno {}",
            get_saved_errno()
        );
        let file = dir.join("f");
        let c_file = std::ffi::CString::new(file.as_os_str().as_bytes()).unwrap();
        let fd = unsafe {
            c_open(
                c_file.as_ptr(),
                libc::O_CREAT | libc::O_RDWR | libc::O_TRUNC,
                0o600,
            )
        };
        assert!(fd >= 0, "c_open errno {}", get_saved_errno());
        assert_eq!(unsafe { c_close(fd) }, 0);

        let hard = dir.join("h");
        let c_hard = std::ffi::CString::new(hard.as_os_str().as_bytes()).unwrap();
        assert_eq!(
            unsafe { c_link(c_file.as_ptr(), c_hard.as_ptr()) },
            0,
            "c_link errno {}",
            get_saved_errno()
        );
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        assert_eq!(unsafe { c_stat(c_hard.as_ptr(), &mut st) }, 0);
        assert!(st.st_nlink >= 2);

        let soft = dir.join("s");
        let c_soft = std::ffi::CString::new(soft.as_os_str().as_bytes()).unwrap();
        assert_eq!(
            unsafe { c_symlink(c"f".as_ptr(), c_soft.as_ptr()) },
            0,
            "c_symlink errno {}",
            get_saved_errno()
        );
        let mut name = [0u8; 8];
        let n = unsafe { c_readlink(c_soft.as_ptr(), name.as_mut_ptr().cast(), name.len()) };
        assert_eq!(n, 1, "c_readlink errno {}", get_saved_errno());
        assert_eq!(&name[..n as usize], b"f");

        assert!(unsafe { c_link(c_file.as_ptr(), c"/no/such/pyre-rffi-link".as_ptr()) } < 0);
        assert_eq!(get_saved_errno(), libc::ENOENT);
        assert!(unsafe { c_symlink(c"f".as_ptr(), c"/no/such/pyre-rffi-symlink".as_ptr()) } < 0);
        assert_eq!(get_saved_errno(), libc::ENOENT);

        assert_eq!(unsafe { c_unlink(c_soft.as_ptr()) }, 0);
        assert_eq!(unsafe { c_unlink(c_hard.as_ptr()) }, 0);
        assert_eq!(unsafe { c_unlink(c_file.as_ptr()) }, 0);
        assert_eq!(unsafe { c_rmdir(c_dir.as_ptr()) }, 0);
    }

    #[cfg(unix)]
    #[test]
    fn c_chown_lchown_and_fchown() {
        use std::os::unix::ffi::OsStrExt;
        let dir = std::env::temp_dir().join(format!("pyre-rffi-chown-{}", std::process::id()));
        let c_dir = std::ffi::CString::new(dir.as_os_str().as_bytes()).unwrap();
        assert_eq!(
            unsafe { c_mkdir(c_dir.as_ptr(), 0o700) },
            0,
            "c_mkdir errno {}",
            get_saved_errno()
        );
        let file = dir.join("f");
        let c_file = std::ffi::CString::new(file.as_os_str().as_bytes()).unwrap();
        let fd = unsafe {
            c_open(
                c_file.as_ptr(),
                libc::O_CREAT | libc::O_RDWR | libc::O_TRUNC,
                0o600,
            )
        };
        assert!(fd >= 0, "c_open errno {}", get_saved_errno());
        let uid = unsafe { libc::getuid() } as libc::c_int;
        let gid = unsafe { libc::getgid() } as libc::c_int;
        assert_eq!(
            unsafe { c_chown(c_file.as_ptr(), uid, gid) },
            0,
            "c_chown errno {}",
            get_saved_errno()
        );
        assert_eq!(
            unsafe { c_fchown(fd, -1, -1) },
            0,
            "c_fchown errno {}",
            get_saved_errno()
        );
        assert_eq!(unsafe { c_close(fd) }, 0);
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        assert_eq!(unsafe { c_stat(c_file.as_ptr(), &mut st) }, 0);
        assert_eq!(st.st_uid, uid as libc::uid_t);
        assert_eq!(st.st_gid, gid as libc::gid_t);

        let soft = dir.join("s");
        let c_soft = std::ffi::CString::new(soft.as_os_str().as_bytes()).unwrap();
        assert_eq!(unsafe { c_symlink(c"f".as_ptr(), c_soft.as_ptr()) }, 0);
        assert_eq!(
            unsafe { c_lchown(c_soft.as_ptr(), -1, -1) },
            0,
            "c_lchown errno {}",
            get_saved_errno()
        );
        assert_eq!(unsafe { c_lstat(c_soft.as_ptr(), &mut st) }, 0);
        assert_eq!(st.st_mode & libc::S_IFMT, libc::S_IFLNK);

        assert!(unsafe { c_chown(c"/no/such/pyre-rffi-chown".as_ptr(), -1, -1) } < 0);
        assert_eq!(get_saved_errno(), libc::ENOENT);
        assert!(unsafe { c_lchown(c"/no/such/pyre-rffi-lchown".as_ptr(), -1, -1) } < 0);
        assert_eq!(get_saved_errno(), libc::ENOENT);
        assert!(unsafe { c_fchown(-1, -1, -1) } < 0);
        assert_eq!(get_saved_errno(), libc::EBADF);

        assert_eq!(unsafe { c_unlink(c_soft.as_ptr()) }, 0);
        assert_eq!(unsafe { c_unlink(c_file.as_ptr()) }, 0);
        assert_eq!(unsafe { c_rmdir(c_dir.as_ptr()) }, 0);
    }

    #[cfg(unix)]
    #[test]
    fn c_futimens_utimensat_and_pipe() {
        use std::os::unix::ffi::OsStrExt;
        let dir = std::env::temp_dir().join(format!("pyre-rffi-utime-{}", std::process::id()));
        let c_dir = std::ffi::CString::new(dir.as_os_str().as_bytes()).unwrap();
        assert_eq!(
            unsafe { c_mkdir(c_dir.as_ptr(), 0o700) },
            0,
            "c_mkdir errno {}",
            get_saved_errno()
        );
        let file = dir.join("f");
        let c_file = std::ffi::CString::new(file.as_os_str().as_bytes()).unwrap();
        let fd = unsafe {
            c_open(
                c_file.as_ptr(),
                libc::O_CREAT | libc::O_RDWR | libc::O_TRUNC,
                0o600,
            )
        };
        assert!(fd >= 0, "c_open errno {}", get_saved_errno());
        let stamp = libc::timespec {
            tv_sec: 1_700_000_000,
            tv_nsec: 0,
        };
        let times = [stamp, stamp];
        assert_eq!(
            unsafe { c_futimens(fd, times.as_ptr()) },
            0,
            "c_futimens errno {}",
            get_saved_errno()
        );
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        assert_eq!(unsafe { c_fstat(fd, &mut st) }, 0);
        assert_eq!(st.st_mtime, 1_700_000_000);
        assert_eq!(unsafe { c_close(fd) }, 0);
        assert!(unsafe { c_futimens(-1, times.as_ptr()) } < 0);
        assert_eq!(get_saved_errno(), libc::EBADF);

        let later = libc::timespec {
            tv_sec: 1_700_000_111,
            tv_nsec: 0,
        };
        let later_times = [later, later];
        assert_eq!(
            unsafe { c_utimensat(libc::AT_FDCWD, c_file.as_ptr(), later_times.as_ptr(), 0) },
            0,
            "c_utimensat errno {}",
            get_saved_errno()
        );
        assert_eq!(unsafe { c_stat(c_file.as_ptr(), &mut st) }, 0);
        assert_eq!(st.st_mtime, 1_700_000_111);
        assert!(
            unsafe {
                c_utimensat(
                    libc::AT_FDCWD,
                    c"/no/such/pyre-rffi-utime".as_ptr(),
                    later_times.as_ptr(),
                    0,
                )
            } < 0
        );
        assert_eq!(get_saved_errno(), libc::ENOENT);

        let mut fds = [0; 2];
        assert_eq!(
            unsafe { c_pipe(fds.as_mut_ptr()) },
            0,
            "c_pipe errno {}",
            get_saved_errno()
        );
        assert_eq!(
            unsafe { c_write(fds[1], b"z".as_ptr() as *mut libc::c_void, 1) },
            1,
            "c_write errno {}",
            get_saved_errno()
        );
        let mut buf = [0u8; 1];
        assert_eq!(
            unsafe { c_read(fds[0], buf.as_mut_ptr().cast(), 1) },
            1,
            "c_read errno {}",
            get_saved_errno()
        );
        assert_eq!(buf, *b"z");
        assert_eq!(unsafe { c_close(fds[0]) }, 0);
        assert_eq!(unsafe { c_close(fds[1]) }, 0);

        assert_eq!(unsafe { c_unlink(c_file.as_ptr()) }, 0);
        assert_eq!(unsafe { c_rmdir(c_dir.as_ptr()) }, 0);
    }
}
