//! `rpython/rlib/rsocket.py` — socket calls above `_rsocket_rffi`.
//!
//! `look_inside_function` rejects this module exactly. `_rsocket_rffi` stays
//! visible to the tracer.

use crate::rffi::{INT, SIGNED};

/// `CSocketError`. `errno` is the code `last_error` read.
pub struct CSocketError {
    pub errno: i32,
}

/// `last_error` — `CSocketError(geterrno())`.
pub fn last_error() -> CSocketError {
    CSocketError {
        errno: crate::_rsocket_rffi::geterrno(),
    }
}

#[cfg(unix)]
type Fd = INT;
#[cfg(windows)]
type Fd = usize;

/// `getsockopt_int` — one `int` socket option.
///
/// A WinSock boolean option may write a single byte. The value starts at
/// zero so the unread bytes stay zero.
#[majit_macros::dont_look_inside]
pub fn getsockopt_int(fd: Fd, level: INT, option: INT) -> Result<SIGNED, CSocketError> {
    let mut flag: INT = 0;
    #[cfg(unix)]
    let mut flagsize: libc::socklen_t = std::mem::size_of::<INT>() as libc::socklen_t;
    #[cfg(windows)]
    let mut flagsize: INT = std::mem::size_of::<INT>() as INT;
    let res = unsafe {
        crate::_rsocket_rffi::socketgetsockopt(
            fd,
            level,
            option,
            (&raw mut flag).cast(),
            &raw mut flagsize,
        )
    };
    if res < 0 {
        return Err(last_error());
    }
    Ok(flag as SIGNED)
}

/// `ntohs` — `_c.ntohs` as a host integer.
pub fn ntohs(x: crate::rffi::USHORT) -> i64 {
    unsafe { crate::_rsocket_rffi::ntohs(x) as i64 }
}

/// `ntohl` — `_c.ntohl` as a host integer.
pub fn ntohl(x: crate::rffi::UINT) -> i64 {
    unsafe { crate::_rsocket_rffi::ntohl(x) as i64 }
}

/// `htons` — `_c.htons` as a host integer.
pub fn htons(x: crate::rffi::USHORT) -> i64 {
    unsafe { crate::_rsocket_rffi::htons(x) as i64 }
}

/// `htonl` — `_c.htonl` as a host integer.
pub fn htonl(x: crate::rffi::UINT) -> i64 {
    unsafe { crate::_rsocket_rffi::htonl(x) as i64 }
}

/// `get_socket_family` — `sa_family` from `getsockname`.
#[majit_macros::dont_look_inside]
pub fn get_socket_family(fd: Fd) -> Result<SIGNED, CSocketError> {
    #[cfg(unix)]
    let mut addr: libc::sockaddr = unsafe { std::mem::zeroed() };
    #[cfg(windows)]
    let mut addr: crate::_rsocket_rffi::sockaddr = unsafe { std::mem::zeroed() };
    #[cfg(unix)]
    let mut addrlen: libc::socklen_t = std::mem::size_of::<libc::sockaddr>() as libc::socklen_t;
    #[cfg(windows)]
    let mut addrlen: INT = std::mem::size_of::<crate::_rsocket_rffi::sockaddr>() as INT;
    let res =
        unsafe { crate::_rsocket_rffi::socketgetsockname(fd, &raw mut addr, &raw mut addrlen) };
    // The length comes back in the same slot the call was given.
    let _addrlen = addrlen;
    let result = addr.sa_family as SIGNED;
    if res < 0 {
        return Err(last_error());
    }
    Ok(result)
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn stream_socket_option_and_family() {
        unsafe {
            let fd = crate::_rsocket_rffi::socket(libc::AF_INET, libc::SOCK_STREAM, 0);
            assert!(fd >= 0, "socket errno {}", crate::rposix::get_saved_errno());
            let ty = getsockopt_int(fd, libc::SOL_SOCKET, libc::SO_TYPE)
                .unwrap_or_else(|error| panic!("getsockopt errno {}", error.errno));
            assert_eq!(ty, libc::SOCK_STREAM as SIGNED);
            let family = get_socket_family(fd)
                .unwrap_or_else(|error| panic!("getsockname errno {}", error.errno));
            assert_eq!(family, libc::AF_INET as SIGNED);
            assert_eq!(crate::_rsocket_rffi::socketclose(fd), 0);

            let inet6 = crate::_rsocket_rffi::socket(libc::AF_INET6, libc::SOCK_STREAM, 0);
            assert!(
                inet6 >= 0,
                "inet6 errno {}",
                crate::rposix::get_saved_errno()
            );
            let family6 = get_socket_family(inet6)
                .unwrap_or_else(|error| panic!("inet6 getsockname errno {}", error.errno));
            assert_eq!(family6, libc::AF_INET6 as SIGNED);
            assert_eq!(crate::_rsocket_rffi::socketclose(inet6), 0);

            let unix = crate::_rsocket_rffi::socket(libc::AF_UNIX, libc::SOCK_STREAM, 0);
            assert!(unix >= 0, "unix errno {}", crate::rposix::get_saved_errno());
            let family_u = get_socket_family(unix)
                .unwrap_or_else(|error| panic!("unix getsockname errno {}", error.errno));
            assert_eq!(family_u, libc::AF_UNIX as SIGNED);
            assert_eq!(crate::_rsocket_rffi::socketclose(unix), 0);
            let missing = getsockopt_int(-1, libc::SOL_SOCKET, libc::SO_TYPE).unwrap_err();
            assert_eq!(missing.errno, libc::EBADF);
        }
    }

    #[test]
    fn network_byte_order_matches_libc() {
        assert_eq!(htons(1), i64::from(libc::htons(1)));
        assert_eq!(ntohs(0x0201), i64::from(libc::ntohs(0x0201)));
        assert_eq!(htonl(0x0102_0304), i64::from(libc::htonl(0x0102_0304)));
        assert_eq!(ntohl(0x0403_0201), i64::from(libc::ntohl(0x0403_0201)));
    }
}
