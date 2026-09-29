//! `rpython/rlib/rtermios.py`.
//!
//! `tcgetattr`, `tcsetattr`, `tcsendbreak`, `tcdrain`, `tcflush`, `tcflow`.

#![allow(non_upper_case_globals)]

use crate::rffi::{INT, RFFI_SAVE_ERRNO, SIGNED, cast};
use crate::{getintfield, setintfield};

/// `rposix.TERMIOS_P` (`lltype.Ptr` of `struct termios`).
#[allow(non_camel_case_types)]
pub type TERMIOS_P = *mut libc::termios;
/// `rtermios.SPEED_T`. `libc` spells `speed_t`; the wrappers cast through it.
#[allow(non_camel_case_types)]
pub type SPEED_T = libc::speed_t;
/// `rposix.TCFLAG_T`.
#[allow(non_camel_case_types)]
pub type TCFLAG_T = libc::tcflag_t;
/// `rposix.CC_T`.
#[allow(non_camel_case_types)]
pub type CC_T = libc::cc_t;
/// `rposix.NCCS`.
pub const NCCS: usize = libc::NCCS;

/// `rtermios.ICANON`. `interp_termios.tcgetattr` masks `c_lflag` with it.
pub const ICANON: TCFLAG_T = libc::ICANON;
/// `rtermios.VMIN`.
pub const VMIN: usize = libc::VMIN;
/// `rtermios.VTIME`.
pub const VTIME: usize = libc::VTIME;

// `rtermios.eci`.
crate::rffi::external_compilation_info! {
    const eci = {
        includes: ["termios.h", "unistd.h", "sys/ioctl.h"],
    };
}

/// `rtermios.c_external`: `rffi.llexternal` with `compilation_info = eci`.
///
/// `compilation_info` is emitted before the caller's kwargs so a trailing
/// comma stays a single trailing comma.
macro_rules! c_external {
    (
        $vis:vis $name:ident = $c_name:literal,
        $args:tt,
        $result:ty
        $(, $($rest:tt)*)?
    ) => {
        $crate::rffi::llexternal!(
            $vis $name = $c_name,
            $args,
            $result,
            compilation_info = eci
            $(, $($rest)*)?
        );
    };
}

// libc gives these a `link_name` on some targets (`cfgetispeed@GLIBC_*`,
// `tcgetattr@GLIBC_*`, `tcdrain$UNIX2003` on macOS x86). `macro = libc::<fn>`
// calls that declaration. The others have no `link_name`.
c_external!(
    pub c_tcgetattr = "tcgetattr",
    [INT, TERMIOS_P],
    INT,
    save_err = RFFI_SAVE_ERRNO,
    macro = libc::tcgetattr
);
c_external!(
    pub c_tcsetattr = "tcsetattr",
    [INT, INT, TERMIOS_P],
    INT,
    save_err = RFFI_SAVE_ERRNO,
    macro = libc::tcsetattr
);
c_external!(
    pub c_cfgetispeed = "cfgetispeed",
    [TERMIOS_P],
    SPEED_T,
    macro = libc::cfgetispeed
);
c_external!(
    pub c_cfgetospeed = "cfgetospeed",
    [TERMIOS_P],
    SPEED_T,
    macro = libc::cfgetospeed
);
c_external!(
    pub c_cfsetispeed = "cfsetispeed",
    [TERMIOS_P, SPEED_T],
    INT,
    save_err = RFFI_SAVE_ERRNO,
    macro = libc::cfsetispeed
);
c_external!(
    pub c_cfsetospeed = "cfsetospeed",
    [TERMIOS_P, SPEED_T],
    INT,
    save_err = RFFI_SAVE_ERRNO,
    macro = libc::cfsetospeed
);
c_external!(
    pub c_tcsendbreak = "tcsendbreak",
    [INT, INT],
    INT,
    save_err = RFFI_SAVE_ERRNO
);
c_external!(
    pub c_tcdrain = "tcdrain",
    [INT],
    INT,
    save_err = RFFI_SAVE_ERRNO,
    macro = libc::tcdrain
);
c_external!(
    pub c_tcflush = "tcflush",
    [INT, INT],
    INT,
    save_err = RFFI_SAVE_ERRNO
);
c_external!(
    pub c_tcflow = "tcflow",
    [INT, INT],
    INT,
    save_err = RFFI_SAVE_ERRNO
);

/// `OSError(errno, message)` from `rposix.get_saved_errno`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OsError {
    pub errno: i32,
    pub message: &'static str,
}

impl std::fmt::Display for OsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.message)
    }
}

impl std::error::Error for OsError {}

fn os_error(message: &'static str) -> OsError {
    OsError {
        errno: crate::rposix::get_saved_errno(),
        message,
    }
}

/// `rtermios.tcgetattr` result. `cc[i]` is the one-byte string `chr(c_cc[i])`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Attributes {
    pub c_iflag: SIGNED,
    pub c_oflag: SIGNED,
    pub c_cflag: SIGNED,
    pub c_lflag: SIGNED,
    pub ispeed: SIGNED,
    pub ospeed: SIGNED,
    pub cc: [[u8; 1]; NCCS],
}

/// `rtermios.tcgetattr`.
pub fn tcgetattr(fd: INT) -> Result<Attributes, OsError> {
    let mut c_struct: libc::termios = unsafe { core::mem::zeroed() };
    if unsafe { c_tcgetattr(fd, &mut c_struct) } < 0 {
        return Err(os_error("tcgetattr failed"));
    }
    let mut cc = [[0u8; 1]; NCCS];
    for i in 0..NCCS {
        cc[i][0] = c_struct.c_cc[i];
    }
    let ispeed = unsafe { c_cfgetispeed(&mut c_struct) };
    let ospeed = unsafe { c_cfgetospeed(&mut c_struct) };
    Ok(Attributes {
        c_iflag: getintfield!(&c_struct, c_iflag),
        c_oflag: getintfield!(&c_struct, c_oflag),
        c_cflag: getintfield!(&c_struct, c_cflag),
        c_lflag: getintfield!(&c_struct, c_lflag),
        ispeed: cast::<SIGNED>(ispeed),
        ospeed: cast::<SIGNED>(ospeed),
        cc,
    })
}

/// `rtermios.tcsetattr`. `cc` entries are one-byte strings; the body reads `[0]`.
pub fn tcsetattr(fd: INT, when: INT, attributes: &Attributes) -> Result<(), OsError> {
    let mut c_struct: libc::termios = unsafe { core::mem::zeroed() };
    setintfield!(&mut c_struct, c_iflag, attributes.c_iflag);
    setintfield!(&mut c_struct, c_oflag, attributes.c_oflag);
    setintfield!(&mut c_struct, c_cflag, attributes.c_cflag);
    setintfield!(&mut c_struct, c_lflag, attributes.c_lflag);
    let ispeed = attributes.ispeed;
    let ospeed = attributes.ospeed;
    let cc = &attributes.cc;
    for i in 0..NCCS {
        c_struct.c_cc[i] = cast::<CC_T>(cc[i][0]);
    }
    if unsafe { c_cfsetispeed(&mut c_struct, cast::<SPEED_T>(ispeed)) } < 0 {
        return Err(os_error("tcsetattr failed"));
    }
    if unsafe { c_cfsetospeed(&mut c_struct, cast::<SPEED_T>(ospeed)) } < 0 {
        return Err(os_error("tcsetattr failed"));
    }
    if unsafe { c_tcsetattr(fd, when, &mut c_struct) } < 0 {
        return Err(os_error("tcsetattr failed"));
    }
    Ok(())
}

/// `rtermios.tcsendbreak`.
pub fn tcsendbreak(fd: INT, duration: INT) -> Result<(), OsError> {
    if unsafe { c_tcsendbreak(fd, duration) } < 0 {
        return Err(os_error("tcsendbreak failed"));
    }
    Ok(())
}

/// `rtermios.tcdrain`.
pub fn tcdrain(fd: INT) -> Result<(), OsError> {
    if unsafe { c_tcdrain(fd) } < 0 {
        return Err(os_error("tcdrain failed"));
    }
    Ok(())
}

/// `rtermios.tcflush`.
pub fn tcflush(fd: INT, queue_selector: INT) -> Result<(), OsError> {
    if unsafe { c_tcflush(fd, queue_selector) } < 0 {
        return Err(os_error("tcflush failed"));
    }
    Ok(())
}

/// `rtermios.tcflow`.
pub fn tcflow(fd: INT, action: INT) -> Result<(), OsError> {
    if unsafe { c_tcflow(fd, action) } < 0 {
        return Err(os_error("tcflow failed"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // `openpty` lives in libutil on Linux and Android, and in libc on the BSDs.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[link(name = "util")]
    unsafe extern "C" {}

    #[test]
    fn tcgetattr_on_a_pipe_fails_with_enotty() {
        let mut fds = [0; 2];
        assert_eq!(unsafe { libc::pipe(fds.as_mut_ptr()) }, 0);
        let err = tcgetattr(fds[0]).expect_err("a pipe is not a tty");
        unsafe {
            libc::close(fds[0]);
            libc::close(fds[1]);
        }
        assert_eq!(err.errno, libc::ENOTTY);
        assert_eq!(err.message, "tcgetattr failed");
    }

    #[test]
    fn pty_round_trip_tcgetattr_tcsetattr() {
        let mut master = -1;
        let mut slave = -1;
        let rc = unsafe {
            libc::openpty(
                &mut master,
                &mut slave,
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
            )
        };
        if rc != 0 {
            // Not usable in this process (no pty, or the symbol did not resolve).
            return;
        }
        struct Close(i32, i32);
        impl Drop for Close {
            fn drop(&mut self) {
                unsafe {
                    libc::close(self.0);
                    libc::close(self.1);
                }
            }
        }
        let _close = Close(master, slave);

        let attr = tcgetattr(slave).expect("tcgetattr");
        tcsetattr(slave, libc::TCSANOW, &attr).expect("tcsetattr");
        let again = tcgetattr(slave).expect("tcgetattr after tcsetattr");
        assert_eq!(again, attr);

        let mut updated = attr;
        let vintr = libc::VINTR;
        let flipped = if updated.cc[vintr][0] == 3 { 4 } else { 3 };
        updated.cc[vintr][0] = flipped;
        tcsetattr(slave, libc::TCSANOW, &updated).expect("tcsetattr mutated");
        let flipped_back = tcgetattr(slave).expect("tcgetattr mutated");
        assert_eq!(flipped_back.cc[vintr][0], flipped);
        assert_eq!(flipped_back.c_iflag, updated.c_iflag);
        assert_eq!(flipped_back.c_lflag, updated.c_lflag);
        assert_eq!(flipped_back.ispeed, updated.ispeed);
        assert_eq!(flipped_back.ospeed, updated.ospeed);
    }
}
