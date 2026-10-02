//! `rpython/rlib/_rsocket_rffi.py` — socket C calls.
//!
//! POSIX covers `select` / `poll` / `FD_*` and the `external` socket calls
//! (`socket`, `connect`, `send`, `recv`, the resolvers, `fcntl`, `dup`,
//! `socketpair`, `if_nameindex`). `recvmsg_implementation`,
//! `sendmsg_implementation`, `CMSG_SPACE_wrapper` and `CMSG_LEN_wrapper`
//! stay out: each one is a separate C source. Windows here is `FD_*`,
//! `select`, `getsockname`, `getsockopt`, the byte-order conversions,
//! `inet_addr`, `inet_ntoa`, `inet_pton`, `inet_ntop`, and `_WSAGetLastError`.

#![allow(non_snake_case, non_camel_case_types)]

#[cfg(target_os = "windows")]
#[link(name = "ws2_32")]
unsafe extern "system" {
    #[link_name = "WSAGetLastError"]
    pub fn _WSAGetLastError() -> i32;
}

#[cfg(unix)]
mod posix {
    use crate::rffi::{INT, RFFI_SAVE_ERRNO, UINT, USHORT};

    // `_rsocket_rffi.eci` includes. Linux-only headers (`netpacket/packet.h`,
    // `linux/netlink.h`) stay out with the packet-socket slice.
    crate::rffi::external_compilation_info! {
        const ECI = {
            includes: [
                "sys/types.h",
                "sys/socket.h",
                "sys/un.h",
                "poll.h",
                "sys/select.h",
                "sys/time.h",
                "netinet/in.h",
                "netinet/tcp.h",
                "unistd.h",
                "fcntl.h",
                "stdio.h",
                "netdb.h",
                "arpa/inet.h",
                "stdint.h",
                "errno.h",
                "limits.h",
                "net/if.h",
            ],
        };
    }

    /// `fd_set` (`COpaquePtr`). Callers pass a pointer at the struct.
    pub type fd_set = *mut libc::fd_set;

    pub const FD_SETSIZE: usize = libc::FD_SETSIZE;
    /// `MAX_FD_SIZE`. POSIX `select` indexes a bitmap by descriptor.
    pub const MAX_FD_SIZE: Option<i32> = Some(FD_SETSIZE as i32);

    /// `_rsocket_rffi.geterrno` on POSIX (`rposix.get_saved_errno`).
    pub fn geterrno() -> i32 {
        crate::rposix::get_saved_errno()
    }

    // libc private build-script cfgs (`gnu_time_bits64`, `gnu_file_offset_bits64`,
    // `musl_redir_time64`) select 32-bit redirects (`__select64`, `__fcntl_time64`,
    // `__getsockopt64`, `__setsockopt64`). Native targets are 64-bit, so those
    // `link_name`s are not copied.
    crate::rffi::llexternal!(
        #[cfg_attr(
            all(target_os = "macos", target_arch = "x86_64"),
            link_name = "select$1050"
        )]
        #[cfg_attr(
            all(target_os = "macos", target_arch = "x86"),
            link_name = "select$UNIX2003"
        )]
        #[cfg_attr(target_os = "netbsd", link_name = "__select50")]
        #[cfg_attr(target_os = "aix", link_name = "__fd_select")]
        pub select = "select",
        [INT, fd_set, fd_set, fd_set, *mut libc::timeval],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );
    crate::rffi::llexternal!(
        #[cfg_attr(
            all(target_os = "macos", target_arch = "x86"),
            link_name = "poll$UNIX2003"
        )]
        pub poll = "poll",
        [*mut libc::pollfd, libc::nfds_t, INT],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );

    // `external_c` `FD_CLR` / `FD_ISSET` / `FD_SET` / `FD_ZERO` is `macro=True`.
    // `libc`'s `FD_ISSET` returns `bool`; upstream's result is `rffi.INT`.
    unsafe fn fd_isset(fd: INT, set: fd_set) -> INT {
        unsafe { libc::FD_ISSET(fd, set) as INT }
    }

    crate::rffi::llexternal!(
        pub FD_CLR = "FD_CLR",
        [INT, fd_set],
        (),
        compilation_info = ECI,
        calling_conv = "c",
        macro = libc::FD_CLR
    );
    crate::rffi::llexternal!(
        pub FD_ISSET = "FD_ISSET",
        [INT, fd_set],
        INT,
        compilation_info = ECI,
        calling_conv = "c",
        macro = fd_isset
    );
    crate::rffi::llexternal!(
        pub FD_SET = "FD_SET",
        [INT, fd_set],
        (),
        compilation_info = ECI,
        calling_conv = "c",
        macro = libc::FD_SET
    );
    crate::rffi::llexternal!(
        pub FD_ZERO = "FD_ZERO",
        [fd_set],
        (),
        compilation_info = ECI,
        calling_conv = "c",
        macro = libc::FD_ZERO
    );

    // `socketclose_no_errno`: `close`, `releasegil=False`, no `save_err`.
    crate::rffi::llexternal!(
        #[cfg_attr(
            all(target_os = "macos", target_arch = "x86"),
            link_name = "close$NOCANCEL$UNIX2003"
        )]
        #[cfg_attr(
            all(target_os = "macos", target_arch = "x86_64"),
            link_name = "close$NOCANCEL"
        )]
        pub socketclose_no_errno = "close",
        [INT],
        INT,
        compilation_info = ECI,
        releasegil = false
    );

    crate::rffi::llexternal!(
        pub dup = "dup",
        [INT],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );
    crate::rffi::llexternal!(
        #[cfg_attr(target_os = "netbsd", link_name = "__socket30")]
        #[cfg_attr(target_os = "illumos", link_name = "__xnet_socket")]
        #[cfg_attr(target_os = "solaris", link_name = "__xnet7_socket")]
        #[cfg_attr(target_os = "espidf", link_name = "lwip_socket")]
        pub socket = "socket",
        [INT, INT, INT],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );
    // `socketclose`: `close`, `releasegil=False`, `save_err`. Distinct from
    // `socketclose_no_errno`, which does not record errno.
    crate::rffi::llexternal!(
        #[cfg_attr(
            all(target_os = "macos", target_arch = "x86"),
            link_name = "close$NOCANCEL$UNIX2003"
        )]
        #[cfg_attr(
            all(target_os = "macos", target_arch = "x86_64"),
            link_name = "close$NOCANCEL"
        )]
        pub socketclose = "close",
        [INT],
        INT,
        compilation_info = ECI,
        releasegil = false,
        save_err = RFFI_SAVE_ERRNO
    );
    crate::rffi::llexternal!(
        #[cfg_attr(
            all(target_os = "macos", target_arch = "x86"),
            link_name = "connect$UNIX2003"
        )]
        #[cfg_attr(
            any(target_os = "illumos", target_os = "solaris"),
            link_name = "__xnet_connect"
        )]
        #[cfg_attr(target_os = "espidf", link_name = "lwip_connect")]
        pub socketconnect = "connect",
        [INT, *const libc::sockaddr, libc::socklen_t],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );
    crate::rffi::llexternal!(
        #[cfg_attr(target_os = "espidf", link_name = "lwip_bind")]
        #[cfg_attr(
            all(target_os = "macos", target_arch = "x86"),
            link_name = "bind$UNIX2003"
        )]
        #[cfg_attr(
            any(target_os = "solaris", target_os = "illumos"),
            link_name = "__xnet_bind"
        )]
        pub socketbind = "bind",
        [INT, *const libc::sockaddr, libc::socklen_t],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );
    crate::rffi::llexternal!(
        #[cfg_attr(
            all(target_os = "macos", target_arch = "x86"),
            link_name = "listen$UNIX2003"
        )]
        #[cfg_attr(target_os = "espidf", link_name = "lwip_listen")]
        pub socketlisten = "listen",
        [INT, INT],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );
    crate::rffi::llexternal!(
        #[cfg_attr(
            all(target_os = "macos", target_arch = "x86"),
            link_name = "accept$UNIX2003"
        )]
        #[cfg_attr(target_os = "espidf", link_name = "lwip_accept")]
        #[cfg_attr(target_os = "aix", link_name = "naccept")]
        pub socketaccept = "accept",
        [INT, *mut libc::sockaddr, *mut libc::socklen_t],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );
    crate::rffi::llexternal!(
        #[cfg_attr(
            all(target_os = "macos", target_arch = "x86"),
            link_name = "getpeername$UNIX2003"
        )]
        #[cfg_attr(target_os = "espidf", link_name = "lwip_getpeername")]
        #[cfg_attr(target_os = "aix", link_name = "ngetpeername")]
        pub socketgetpeername = "getpeername",
        [INT, *mut libc::sockaddr, *mut libc::socklen_t],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );
    crate::rffi::llexternal!(
        #[cfg_attr(
            all(target_os = "macos", target_arch = "x86"),
            link_name = "getsockname$UNIX2003"
        )]
        #[cfg_attr(target_os = "espidf", link_name = "lwip_getsockname")]
        #[cfg_attr(target_os = "aix", link_name = "ngetsockname")]
        pub socketgetsockname = "getsockname",
        [INT, *mut libc::sockaddr, *mut libc::socklen_t],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );
    crate::rffi::llexternal!(
        #[cfg_attr(
            any(target_os = "illumos", target_os = "solaris"),
            link_name = "__xnet_getsockopt"
        )]
        #[cfg_attr(target_os = "espidf", link_name = "lwip_getsockopt")]
        pub socketgetsockopt = "getsockopt",
        [INT, INT, INT, *mut libc::c_void, *mut libc::socklen_t],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );
    // `htons` / `ntohs` / `htonl` / `ntohl`. Darwin and OpenBSD publish
    // these as macros; the libc crate exposes the same functions on every
    // unix target.
    crate::rffi::llexternal!(
        pub htons = "htons",
        [USHORT],
        USHORT,
        compilation_info = ECI,
        macro = libc::htons
    );
    crate::rffi::llexternal!(
        pub ntohs = "ntohs",
        [USHORT],
        USHORT,
        compilation_info = ECI,
        macro = libc::ntohs
    );
    crate::rffi::llexternal!(
        pub htonl = "htonl",
        [UINT],
        UINT,
        compilation_info = ECI,
        macro = libc::htonl
    );
    crate::rffi::llexternal!(
        pub ntohl = "ntohl",
        [UINT],
        UINT,
        compilation_info = ECI,
        macro = libc::ntohl
    );
    crate::rffi::llexternal!(
        #[cfg_attr(target_os = "espidf", link_name = "lwip_setsockopt")]
        pub socketsetsockopt = "setsockopt",
        [INT, INT, INT, *const libc::c_void, libc::socklen_t],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );
    crate::rffi::llexternal!(
        #[cfg_attr(
            all(target_os = "macos", target_arch = "x86"),
            link_name = "socketpair$UNIX2003"
        )]
        #[cfg_attr(
            any(target_os = "illumos", target_os = "solaris"),
            link_name = "__xnet_socketpair"
        )]
        pub socketpair = "socketpair",
        [INT, INT, INT, *mut INT],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );
    crate::rffi::llexternal!(
        #[cfg_attr(
            all(target_os = "macos", target_arch = "x86"),
            link_name = "send$UNIX2003"
        )]
        #[cfg_attr(target_os = "espidf", link_name = "lwip_send")]
        pub send = "send",
        [INT, *const libc::c_void, libc::size_t, INT],
        libc::ssize_t,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );
    crate::rffi::llexternal!(
        #[cfg_attr(
            all(target_os = "macos", target_arch = "x86"),
            link_name = "recv$UNIX2003"
        )]
        #[cfg_attr(target_os = "espidf", link_name = "lwip_recv")]
        pub socketrecv = "recv",
        [INT, *mut libc::c_void, libc::size_t, INT],
        libc::ssize_t,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );
    crate::rffi::llexternal!(
        #[cfg_attr(
            all(target_os = "macos", target_arch = "x86"),
            link_name = "sendto$UNIX2003"
        )]
        #[cfg_attr(
            any(target_os = "illumos", target_os = "solaris"),
            link_name = "__xnet_sendto"
        )]
        #[cfg_attr(target_os = "espidf", link_name = "lwip_sendto")]
        pub sendto = "sendto",
        [
            INT,
            *const libc::c_void,
            libc::size_t,
            INT,
            *const libc::sockaddr,
            libc::socklen_t
        ],
        libc::ssize_t,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );
    crate::rffi::llexternal!(
        #[cfg_attr(target_os = "espidf", link_name = "lwip_recvfrom")]
        #[cfg_attr(
            all(target_os = "macos", target_arch = "x86"),
            link_name = "recvfrom$UNIX2003"
        )]
        #[cfg_attr(target_os = "aix", link_name = "nrecvfrom")]
        pub recvfrom = "recvfrom",
        [
            INT,
            *mut libc::c_void,
            libc::size_t,
            INT,
            *mut libc::sockaddr,
            *mut libc::socklen_t
        ],
        libc::ssize_t,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );
    crate::rffi::llexternal!(
        #[cfg_attr(target_os = "espidf", link_name = "lwip_shutdown")]
        pub socketshutdown = "shutdown",
        [INT, INT],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );
    crate::rffi::llexternal!(
        pub gethostname = "gethostname",
        [*mut libc::c_char, libc::size_t],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );
    // `fcntl` is variadic. `_rsocket_rffi.fcntl` uses `natural_arity = 2`.
    crate::rffi::llexternal!(
        #[cfg_attr(
            all(target_os = "macos", target_arch = "x86"),
            link_name = "fcntl$UNIX2003"
        )]
        pub fcntl = "fcntl",
        [INT, INT, INT],
        INT,
        compilation_info = ECI,
        natural_arity = 2,
        save_err = RFFI_SAVE_ERRNO
    );
    crate::rffi::llexternal!(
        pub gai_strerror = "gai_strerror",
        [INT],
        *const libc::c_char,
        compilation_info = ECI
    );
    crate::rffi::llexternal!(
        #[cfg_attr(
            any(target_os = "illumos", target_os = "solaris"),
            link_name = "__xnet_getaddrinfo"
        )]
        #[cfg_attr(target_os = "espidf", link_name = "lwip_getaddrinfo")]
        pub getaddrinfo = "getaddrinfo",
        [
            *const libc::c_char,
            *const libc::c_char,
            *const libc::addrinfo,
            *mut *mut libc::addrinfo
        ],
        INT,
        compilation_info = ECI
    );
    crate::rffi::llexternal!(
        #[cfg_attr(target_os = "espidf", link_name = "lwip_freeaddrinfo")]
        pub freeaddrinfo = "freeaddrinfo",
        [*mut libc::addrinfo],
        (),
        compilation_info = ECI
    );
    crate::rffi::llexternal!(
        pub getnameinfo = "getnameinfo",
        [
            *const libc::sockaddr,
            libc::socklen_t,
            *mut libc::c_char,
            libc::socklen_t,
            *mut libc::c_char,
            libc::socklen_t,
            INT
        ],
        INT,
        compilation_info = ECI
    );
    crate::rffi::llexternal!(
        pub getservbyname = "getservbyname",
        [*const libc::c_char, *const libc::c_char],
        *mut libc::servent,
        compilation_info = ECI
    );
    crate::rffi::llexternal!(
        pub getservbyport = "getservbyport",
        [INT, *const libc::c_char],
        *mut libc::servent,
        compilation_info = ECI
    );
    crate::rffi::llexternal!(
        pub getprotobyname = "getprotobyname",
        [*const libc::c_char],
        *mut libc::protoent,
        compilation_info = ECI
    );
    crate::rffi::llexternal!(
        pub inet_aton = "inet_aton",
        [*const libc::c_char, *mut libc::in_addr],
        INT,
        compilation_info = ECI
    );
    crate::rffi::llexternal!(
        pub inet_ntoa = "inet_ntoa",
        [libc::in_addr],
        *mut libc::c_char,
        compilation_info = ECI
    );
    crate::rffi::llexternal!(
        pub inet_pton = "inet_pton",
        [INT, *const libc::c_char, *mut libc::c_void],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );
    crate::rffi::llexternal!(
        pub inet_ntop = "inet_ntop",
        [INT, *const libc::c_void, *mut libc::c_char, libc::socklen_t],
        *const libc::c_char,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );
    // `gethostbyaddr`'s length is `socklen_t`, matching the call pyre already
    // makes. The resolver records failure in `h_errno`, not `errno`.
    crate::rffi::llexternal!(
        pub gethostbyname = "gethostbyname",
        [*const libc::c_char],
        *mut libc::c_void,
        compilation_info = ECI
    );
    crate::rffi::llexternal!(
        pub gethostbyaddr = "gethostbyaddr",
        [*const libc::c_void, libc::socklen_t, INT],
        *mut libc::c_void,
        compilation_info = ECI
    );
    crate::rffi::llexternal!(
        pub if_nameindex = "if_nameindex",
        [],
        *mut libc::if_nameindex,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );
    crate::rffi::llexternal!(
        pub if_freenameindex = "if_freenameindex",
        [*mut libc::if_nameindex],
        (),
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );

    // `libc::sethostname`'s length is `c_int` on Apple, FreeBSD, DragonFly,
    // AIX and Solaris, and `size_t` on Linux and the other BSDs.
    #[cfg(any(
        target_vendor = "apple",
        target_os = "freebsd",
        target_os = "dragonfly",
        target_os = "aix",
        target_os = "solaris",
        target_os = "illumos",
    ))]
    crate::rffi::llexternal!(
        pub(super) c_sethostname = "sethostname",
        [*const libc::c_char, libc::c_int],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );
    #[cfg(not(any(
        target_vendor = "apple",
        target_os = "freebsd",
        target_os = "dragonfly",
        target_os = "aix",
        target_os = "solaris",
        target_os = "illumos",
    )))]
    crate::rffi::llexternal!(
        pub(super) c_sethostname = "sethostname",
        [*const libc::c_char, libc::size_t],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO
    );

    /// `sethostname`. `len` is the byte count, narrowed to `c_int` where
    /// `libc::sethostname` takes one.
    pub unsafe fn sethostname(name: *const libc::c_char, len: usize) -> libc::c_int {
        unsafe {
            #[cfg(any(
                target_vendor = "apple",
                target_os = "freebsd",
                target_os = "dragonfly",
                target_os = "aix",
                target_os = "solaris",
                target_os = "illumos",
            ))]
            {
                c_sethostname(name, len as libc::c_int)
            }
            #[cfg(not(any(
                target_vendor = "apple",
                target_os = "freebsd",
                target_os = "dragonfly",
                target_os = "aix",
                target_os = "solaris",
                target_os = "illumos",
            )))]
            {
                c_sethostname(name, len)
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn stream_socket_dup_fcntl_and_close() {
            unsafe {
                let fd = socket(libc::AF_INET, libc::SOCK_STREAM, 0);
                assert!(fd >= 0, "socket errno {}", crate::rposix::get_saved_errno());
                let flags = fcntl(fd, libc::F_GETFD, 0);
                assert!(
                    flags >= 0,
                    "fcntl errno {}",
                    crate::rposix::get_saved_errno()
                );
                let duped = dup(fd);
                assert!(duped >= 0, "dup errno {}", crate::rposix::get_saved_errno());
                assert_eq!(socketclose(duped), 0);
                assert_eq!(socketclose(fd), 0);
            }
        }

        #[test]
        fn inet_pton_loopback() {
            let text = c"127.0.0.1";
            let mut buf = [0u8; 4];
            let rc = unsafe { inet_pton(libc::AF_INET, text.as_ptr(), buf.as_mut_ptr().cast()) };
            assert_eq!(rc, 1);
            assert_eq!(buf, [127, 0, 0, 1]);
        }
    }
}

#[cfg(unix)]
pub use posix::*;

#[cfg(windows)]
mod winsock {
    use crate::rffi::{INT, RFFI_SAVE_WSALASTERROR, UINT, USHORT};

    /// WinSock `FD_SETSIZE`. `constants_w_defaults` uses 64 when the header
    /// does not override it, and the SDK default is 64.
    pub const FD_SETSIZE: usize = 64;
    /// `MAX_FD_SIZE` is `None` on Windows: the set is a count, not a bitmap.
    pub const MAX_FD_SIZE: Option<i32> = None;

    /// `struct fd_set` (`fd_count` plus `fd_array[FD_SETSIZE]`).
    #[repr(C)]
    pub struct fd_set {
        pub fd_count: u32,
        pub fd_array: [usize; FD_SETSIZE],
    }

    /// `struct timeval` (`tv_sec` / `tv_usec` are `long`).
    #[repr(C)]
    pub struct timeval {
        pub tv_sec: std::ffi::c_long,
        pub tv_usec: std::ffi::c_long,
    }

    /// `struct sockaddr` (`sockaddr`). `sa_family` is `ADDRESS_FAMILY`.
    #[repr(C)]
    pub struct sockaddr {
        pub sa_family: u16,
        pub sa_data: [i8; 14],
    }

    pub type fd_set_p = *mut fd_set;

    /// `_rsocket_rffi.geterrno` on Windows (`rwin32.GetLastError_saved`).
    /// `select` saves `WSAGetLastError` into that slot.
    pub fn geterrno() -> i32 {
        crate::rwin32::GetLastError_saved() as i32
    }

    crate::rffi::external_compilation_info! {
        const ECI = {
            includes: ["winsock2.h", "ws2tcpip.h"],
            libraries: ["ws2_32"],
        };
    }

    unsafe extern "system" {
        fn __WSAFDIsSet(fd: usize, set: fd_set_p) -> i32;
    }

    // `external_c` `FD_ZERO` / `FD_SET` / `FD_ISSET` / `FD_CLR` is `macro=True`.
    // The bodies match
    // the WinSock headers: scan `fd_array`, append under `FD_SETSIZE`.
    // `FD_*` take `rffi.INT` on Windows too (`external_c`). The WinSock
    // array stores `SOCKET` (`uintptr`), so the body sign-extends that int.
    unsafe fn fd_zero(set: fd_set_p) {
        unsafe { (*set).fd_count = 0 };
    }
    unsafe fn fd_set_insert(fd: INT, set: fd_set_p) {
        let fd = fd as usize;
        unsafe {
            let count = (*set).fd_count as usize;
            for slot in 0..count {
                if (*set).fd_array[slot] == fd {
                    return;
                }
            }
            if count < FD_SETSIZE {
                (*set).fd_array[count] = fd;
                (*set).fd_count += 1;
            }
        }
    }
    unsafe fn fd_isset(fd: INT, set: fd_set_p) -> INT {
        unsafe { __WSAFDIsSet(fd as usize, set) }
    }
    unsafe fn fd_clr(fd: INT, set: fd_set_p) {
        let fd = fd as usize;
        unsafe {
            let count = (*set).fd_count as usize;
            let mut kept = 0usize;
            for slot in 0..count {
                let cur = (*set).fd_array[slot];
                if cur != fd {
                    (*set).fd_array[kept] = cur;
                    kept += 1;
                }
            }
            (*set).fd_count = kept as u32;
        }
    }

    crate::rffi::llexternal!(
        pub FD_ZERO = "FD_ZERO",
        [fd_set_p],
        (),
        compilation_info = ECI,
        calling_conv = "c",
        macro = fd_zero
    );
    crate::rffi::llexternal!(
        pub FD_SET = "FD_SET",
        [INT, fd_set_p],
        (),
        compilation_info = ECI,
        calling_conv = "c",
        macro = fd_set_insert
    );
    crate::rffi::llexternal!(
        pub FD_ISSET = "FD_ISSET",
        [INT, fd_set_p],
        INT,
        compilation_info = ECI,
        calling_conv = "c",
        macro = fd_isset
    );
    crate::rffi::llexternal!(
        pub FD_CLR = "FD_CLR",
        [INT, fd_set_p],
        (),
        compilation_info = ECI,
        calling_conv = "c",
        macro = fd_clr
    );

    // `external('select', ..., save_err=RFFI_SAVE_WSALASTERROR)` with
    // `calling_conv='win'`.
    crate::rffi::llexternal!(
        pub select = "select",
        [INT, fd_set_p, fd_set_p, fd_set_p, *mut timeval],
        INT,
        compilation_info = ECI,
        calling_conv = "win",
        save_err = RFFI_SAVE_WSALASTERROR
    );
    // `socketgetsockname` / `socketgetsockopt` (`external`, `save_err=SAVE_ERR`).
    // The descriptor is `lltype.Unsigned` (`socketfd_type` on Windows).
    crate::rffi::llexternal!(
        pub socketgetsockname = "getsockname",
        [usize, *mut sockaddr, *mut INT],
        INT,
        compilation_info = ECI,
        calling_conv = "win",
        save_err = RFFI_SAVE_WSALASTERROR
    );
    crate::rffi::llexternal!(
        pub socketgetsockopt = "getsockopt",
        [usize, INT, INT, *mut core::ffi::c_void, *mut INT],
        INT,
        compilation_info = ECI,
        calling_conv = "win",
        save_err = RFFI_SAVE_WSALASTERROR
    );
    // `htons` / `ntohs` / `htonl` / `ntohl` (`external`, not the Darwin macro).
    crate::rffi::llexternal!(
        pub htons = "htons",
        [USHORT],
        USHORT,
        compilation_info = ECI,
        calling_conv = "win"
    );
    crate::rffi::llexternal!(
        pub ntohs = "ntohs",
        [USHORT],
        USHORT,
        compilation_info = ECI,
        calling_conv = "win"
    );
    crate::rffi::llexternal!(
        pub htonl = "htonl",
        [UINT],
        UINT,
        compilation_info = ECI,
        calling_conv = "win"
    );
    crate::rffi::llexternal!(
        pub ntohl = "ntohl",
        [UINT],
        UINT,
        compilation_info = ECI,
        calling_conv = "win"
    );

    /// `struct in_addr`. `S_addr` is the first field of `S_un`.
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct in_addr {
        pub s_addr: u32,
    }

    // `inet_addr` / `inet_ntoa`. Windows has no `inet_aton`.
    crate::rffi::llexternal!(
        pub inet_addr = "inet_addr",
        [*const std::ffi::c_char],
        UINT,
        compilation_info = ECI,
        calling_conv = "win"
    );
    crate::rffi::llexternal!(
        pub inet_ntoa = "inet_ntoa",
        [in_addr],
        *mut std::ffi::c_char,
        compilation_info = ECI,
        calling_conv = "win"
    );

    /// `AF_INET` / `AF_INET6` in WinSock.
    pub const AF_INET: INT = 2;
    pub const AF_INET6: INT = 23;

    crate::rffi::llexternal!(
        pub inet_pton = "inet_pton",
        [INT, *const std::ffi::c_char, *mut core::ffi::c_void],
        INT,
        compilation_info = ECI,
        calling_conv = "win",
        save_err = RFFI_SAVE_WSALASTERROR
    );
    crate::rffi::llexternal!(
        pub inet_ntop = "inet_ntop",
        [INT, *const core::ffi::c_void, *mut std::ffi::c_char, usize],
        *const std::ffi::c_char,
        compilation_info = ECI,
        calling_conv = "win",
        save_err = RFFI_SAVE_WSALASTERROR
    );
}

#[cfg(windows)]
pub use winsock::*;
