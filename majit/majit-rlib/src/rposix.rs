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
        includes: ["fcntl.h", "unistd.h", "sys/types.h", "sys/stat.h", "dirent.h", "string.h"],
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
}
