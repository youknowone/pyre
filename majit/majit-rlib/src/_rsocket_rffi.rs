//! `rpython/rlib/_rsocket_rffi.py` — socket C calls.
//!
//! POSIX covers `select` / `poll` / `FD_*` and the `external` socket calls
//! (`socket`, `connect`, `send`, `recv`, the resolvers, `fcntl`, `dup`,
//! `socketpair`, `if_nameindex`). `recvmsg_implementation`,
//! `sendmsg_implementation`, `CMSG_SPACE_wrapper` and `CMSG_LEN_wrapper`
//! stay out: each one is a separate C source. Windows here is `FD_*`,
//! `select`, and `_WSAGetLastError`.

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

    // `libc` renames `select` (`select$1050` / `select$UNIX2003`) and `poll`
    // (`poll$UNIX2003` on macOS x86). `macro = libc::<fn>` calls that declaration.
    crate::rffi::llexternal!(
        pub select = "select",
        [INT, fd_set, fd_set, fd_set, *mut libc::timeval],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO,
        macro = libc::select
    );
    crate::rffi::llexternal!(
        pub poll = "poll",
        [*mut libc::pollfd, libc::nfds_t, INT],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO,
        macro = libc::poll
    );

    // `external_c(..., macro=True)`: `FD_*` are macros. `libc`'s wrappers
    // return `bool` from `FD_ISSET`; upstream's result is `rffi.INT`.
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
        pub socketclose_no_errno = "close",
        [INT],
        INT,
        compilation_info = ECI,
        releasegil = false,
        macro = libc::close
    );

    // `libc` renames several of these on macOS (`socket` is plain;
    // `connect$UNIX2003`, `send$UNIX2003`, `recvfrom$UNIX2003`, …).
    // `macro = libc::<fn>` follows that declaration. Symbols the crate does
    // not declare (`inet_pton`, `inet_aton`, `gethostbyname`) stay real
    // externs.
    crate::rffi::llexternal!(
        pub dup = "dup",
        [INT],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO,
        macro = libc::dup
    );
    crate::rffi::llexternal!(
        pub socket = "socket",
        [INT, INT, INT],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO,
        macro = libc::socket
    );
    // `socketclose`: `close`, `releasegil=False`, `save_err`. Distinct from
    // `socketclose_no_errno`, which does not record errno.
    crate::rffi::llexternal!(
        pub socketclose = "close",
        [INT],
        INT,
        compilation_info = ECI,
        releasegil = false,
        save_err = RFFI_SAVE_ERRNO,
        macro = libc::close
    );
    crate::rffi::llexternal!(
        pub socketconnect = "connect",
        [INT, *const libc::sockaddr, libc::socklen_t],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO,
        macro = libc::connect
    );
    crate::rffi::llexternal!(
        pub socketbind = "bind",
        [INT, *const libc::sockaddr, libc::socklen_t],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO,
        macro = libc::bind
    );
    crate::rffi::llexternal!(
        pub socketlisten = "listen",
        [INT, INT],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO,
        macro = libc::listen
    );
    crate::rffi::llexternal!(
        pub socketaccept = "accept",
        [INT, *mut libc::sockaddr, *mut libc::socklen_t],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO,
        macro = libc::accept
    );
    crate::rffi::llexternal!(
        pub socketgetpeername = "getpeername",
        [INT, *mut libc::sockaddr, *mut libc::socklen_t],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO,
        macro = libc::getpeername
    );
    crate::rffi::llexternal!(
        pub socketgetsockname = "getsockname",
        [INT, *mut libc::sockaddr, *mut libc::socklen_t],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO,
        macro = libc::getsockname
    );
    crate::rffi::llexternal!(
        pub socketgetsockopt = "getsockopt",
        [INT, INT, INT, *mut libc::c_void, *mut libc::socklen_t],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO,
        macro = libc::getsockopt
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
        pub socketsetsockopt = "setsockopt",
        [INT, INT, INT, *const libc::c_void, libc::socklen_t],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO,
        macro = libc::setsockopt
    );
    crate::rffi::llexternal!(
        pub socketpair = "socketpair",
        [INT, INT, INT, *mut INT],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO,
        macro = libc::socketpair
    );
    crate::rffi::llexternal!(
        pub send = "send",
        [INT, *const libc::c_void, libc::size_t, INT],
        libc::ssize_t,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO,
        macro = libc::send
    );
    crate::rffi::llexternal!(
        pub socketrecv = "recv",
        [INT, *mut libc::c_void, libc::size_t, INT],
        libc::ssize_t,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO,
        macro = libc::recv
    );
    crate::rffi::llexternal!(
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
        save_err = RFFI_SAVE_ERRNO,
        macro = libc::sendto
    );
    crate::rffi::llexternal!(
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
        save_err = RFFI_SAVE_ERRNO,
        macro = libc::recvfrom
    );
    crate::rffi::llexternal!(
        pub socketshutdown = "shutdown",
        [INT, INT],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO,
        macro = libc::shutdown
    );
    crate::rffi::llexternal!(
        pub gethostname = "gethostname",
        [*mut libc::c_char, libc::size_t],
        INT,
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO,
        macro = libc::gethostname
    );
    // `fcntl` is variadic. `_rsocket_rffi.fcntl` uses `natural_arity = 2`.
    crate::rffi::llexternal!(
        pub fcntl = "fcntl",
        [INT, INT, INT],
        INT,
        compilation_info = ECI,
        natural_arity = 2,
        save_err = RFFI_SAVE_ERRNO,
        macro = libc::fcntl
    );
    crate::rffi::llexternal!(
        pub gai_strerror = "gai_strerror",
        [INT],
        *const libc::c_char,
        compilation_info = ECI,
        macro = libc::gai_strerror
    );
    crate::rffi::llexternal!(
        pub getaddrinfo = "getaddrinfo",
        [
            *const libc::c_char,
            *const libc::c_char,
            *const libc::addrinfo,
            *mut *mut libc::addrinfo
        ],
        INT,
        compilation_info = ECI,
        macro = libc::getaddrinfo
    );
    crate::rffi::llexternal!(
        pub freeaddrinfo = "freeaddrinfo",
        [*mut libc::addrinfo],
        (),
        compilation_info = ECI,
        macro = libc::freeaddrinfo
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
        compilation_info = ECI,
        macro = libc::getnameinfo
    );
    crate::rffi::llexternal!(
        pub getservbyname = "getservbyname",
        [*const libc::c_char, *const libc::c_char],
        *mut libc::servent,
        compilation_info = ECI,
        macro = libc::getservbyname
    );
    crate::rffi::llexternal!(
        pub getservbyport = "getservbyport",
        [INT, *const libc::c_char],
        *mut libc::servent,
        compilation_info = ECI,
        macro = libc::getservbyport
    );
    crate::rffi::llexternal!(
        pub getprotobyname = "getprotobyname",
        [*const libc::c_char],
        *mut libc::protoent,
        compilation_info = ECI,
        macro = libc::getprotobyname
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
        save_err = RFFI_SAVE_ERRNO,
        macro = libc::if_nameindex
    );
    crate::rffi::llexternal!(
        pub if_freenameindex = "if_freenameindex",
        [*mut libc::if_nameindex],
        (),
        compilation_info = ECI,
        save_err = RFFI_SAVE_ERRNO,
        macro = libc::if_freenameindex
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
        save_err = RFFI_SAVE_ERRNO,
        macro = libc::sethostname
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
        save_err = RFFI_SAVE_ERRNO,
        macro = libc::sethostname
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
            includes: ["winsock2.h"],
            libraries: ["ws2_32"],
        };
    }

    unsafe extern "system" {
        fn __WSAFDIsSet(fd: usize, set: fd_set_p) -> i32;
    }

    // `FD_*` are macros (`external_c(..., macro=True)`). The bodies match
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
}

#[cfg(windows)]
pub use winsock::*;
