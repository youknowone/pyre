//! `rpython/rlib/rsocket.py` — socket calls above `_rsocket_rffi`.
//!
//! `look_inside_function` rejects this module exactly. `_rsocket_rffi` stays
//! visible to the tracer.

use crate::rffi::{INT, SIGNED};

/// `CSocketError`. `errno` is the code `last_error` read.
#[derive(Debug)]
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

/// `Defaults.timeout`. `-1.0` blocks. Every thread reads the same cell.
static DEFAULT_TIMEOUT_BITS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new((-1.0f64).to_bits());

/// `getdefaulttimeout`.
pub fn getdefaulttimeout() -> f64 {
    f64::from_bits(DEFAULT_TIMEOUT_BITS.load(std::sync::atomic::Ordering::Relaxed))
}

/// `setdefaulttimeout`. A negative value is stored as `-1.0`.
pub fn setdefaulttimeout(timeout: f64) {
    let timeout = if timeout < 0.0 { -1.0 } else { timeout };
    DEFAULT_TIMEOUT_BITS.store(timeout.to_bits(), std::sync::atomic::Ordering::Relaxed);
}

/// `RSocketError`. `message` is `get_msg`.
#[derive(Debug)]
pub struct RSocketError {
    pub message: &'static str,
}

fn rsocket_error(message: &'static str) -> RSocketError {
    RSocketError { message }
}

/// `inet_aton`. Four address bytes, network order.
///
/// Windows has no `inet_aton`. `inet_addr` reports failure as `INADDR_NONE`,
/// which is also the broadcast address, so that spelling is answered first.
#[cfg(unix)]
pub fn inet_aton(ip: &std::ffi::CStr) -> Result<[u8; 4], RSocketError> {
    let mut addr: libc::in_addr = unsafe { std::mem::zeroed() };
    let ok = unsafe { crate::_rsocket_rffi::inet_aton(ip.as_ptr(), &raw mut addr) };
    if ok == 0 {
        return Err(rsocket_error(
            "illegal IP address string passed to inet_aton",
        ));
    }
    Ok(addr.s_addr.to_ne_bytes())
}

#[cfg(windows)]
pub fn inet_aton(ip: &std::ffi::CStr) -> Result<[u8; 4], RSocketError> {
    if ip.to_bytes() == b"255.255.255.255" {
        return Ok([0xff; 4]);
    }
    let packed = unsafe { crate::_rsocket_rffi::inet_addr(ip.as_ptr()) };
    if packed == crate::rffi::UINT::MAX {
        return Err(rsocket_error(
            "illegal IP address string passed to inet_aton",
        ));
    }
    Ok((packed as u32).to_ne_bytes())
}

/// `inet_ntoa`. The packed buffer is `sizeof(in_addr)` bytes.
pub fn inet_ntoa(packed: &[u8]) -> Result<String, RSocketError> {
    #[cfg(unix)]
    let width = std::mem::size_of::<libc::in_addr>();
    #[cfg(windows)]
    let width = std::mem::size_of::<crate::_rsocket_rffi::in_addr>();
    if packed.len() != width {
        return Err(rsocket_error("packed IP wrong length for inet_ntoa"));
    }
    let mut bytes = [0u8; 4];
    bytes.copy_from_slice(packed);
    #[cfg(unix)]
    let addr = libc::in_addr {
        s_addr: u32::from_ne_bytes(bytes),
    };
    #[cfg(windows)]
    let addr = crate::_rsocket_rffi::in_addr {
        s_addr: u32::from_ne_bytes(bytes),
    };
    let text = unsafe { crate::_rsocket_rffi::inet_ntoa(addr) };
    if text.is_null() {
        return Err(rsocket_error("inet_ntoa failed"));
    }
    Ok(unsafe { std::ffi::CStr::from_ptr(text) }
        .to_string_lossy()
        .into_owned())
}

/// `gethostname`. The buffer is 1024 bytes. The result stops at the first NUL.
#[cfg(unix)]
pub fn gethostname() -> Result<Vec<u8>, CSocketError> {
    let mut buf = [0u8; 1024];
    let res = unsafe { crate::_rsocket_rffi::gethostname(buf.as_mut_ptr().cast(), buf.len()) };
    if res < 0 {
        return Err(last_error());
    }
    let end = buf.iter().position(|&byte| byte == 0).unwrap_or(buf.len());
    Ok(buf[..end].to_vec())
}

/// `sethostname`. `hostname` is the raw byte count passed to the syscall.
#[cfg(unix)]
pub fn sethostname(hostname: &[u8]) -> Result<(), CSocketError> {
    let ptr = if hostname.is_empty() {
        c"".as_ptr()
    } else {
        hostname.as_ptr().cast()
    };
    let res = unsafe { crate::_rsocket_rffi::sethostname(ptr, hostname.len()) };
    if res < 0 {
        return Err(last_error());
    }
    Ok(())
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

    #[test]
    fn default_timeout_blocks_until_set() {
        let saved = getdefaulttimeout();
        setdefaulttimeout(-1.0);
        assert_eq!(getdefaulttimeout(), -1.0);
        setdefaulttimeout(1.5);
        assert_eq!(getdefaulttimeout(), 1.5);
        setdefaulttimeout(-4.0);
        assert_eq!(getdefaulttimeout(), -1.0);
        setdefaulttimeout(saved);
    }

    #[test]
    fn inet_aton_and_ntoa_round_trip() {
        let packed = inet_aton(c"127.0.0.1").expect("loopback");
        assert_eq!(packed, [127, 0, 0, 1]);
        assert_eq!(inet_ntoa(&packed).expect("ntoa"), "127.0.0.1");
        let broadcast = inet_aton(c"255.255.255.255").expect("broadcast");
        assert_eq!(broadcast, [255, 255, 255, 255]);
        assert_eq!(
            inet_ntoa(&broadcast).expect("broadcast text"),
            "255.255.255.255"
        );
        assert_eq!(
            inet_aton(c"nope").expect_err("bad address").message,
            "illegal IP address string passed to inet_aton"
        );
        assert_eq!(
            inet_ntoa(&[1, 2, 3]).expect_err("short").message,
            "packed IP wrong length for inet_ntoa"
        );
    }

    #[test]
    fn gethostname_matches_libc() {
        let ours = gethostname().expect("gethostname");
        let mut buf = [0u8; 1024];
        let rc = unsafe { libc::gethostname(buf.as_mut_ptr().cast(), buf.len()) };
        assert_eq!(rc, 0);
        let end = buf.iter().position(|&byte| byte == 0).unwrap_or(buf.len());
        assert_eq!(ours, buf[..end]);
    }
}
