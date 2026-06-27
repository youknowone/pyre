//! _socket module — PyPy: pypy/module/_socket/interp_socket.py.
//!
//! Carries the W_Socket class implementation plus the shared address
//! conversion / IDNA / error-mapping helpers.  `register_module` is the
//! single entry point invoked by `moduledef::init`; it populates the
//! module namespace with constants, error classes, module-level
//! functions and the `socket` type definition.

use crate::DictStorage;

// POSIX socket FFI declarations missing from libc 0.2.186.  These are
// universal symbols from <arpa/inet.h>, <netdb.h>, <unistd.h>; we
// declare them at module scope so both init_socket and the socket()
// instance methods can call them.
#[cfg(unix)]
unsafe extern "C" {
    fn inet_aton(cp: *const libc::c_char, inp: *mut libc::in_addr) -> libc::c_int;
    fn inet_ntoa(addr: libc::in_addr) -> *mut libc::c_char;
    fn inet_pton(af: libc::c_int, src: *const libc::c_char, dst: *mut libc::c_void) -> libc::c_int;
    fn inet_ntop(
        af: libc::c_int,
        src: *const libc::c_void,
        dst: *mut libc::c_char,
        size: libc::socklen_t,
    ) -> *const libc::c_char;
    fn gethostname(name: *mut libc::c_char, len: libc::size_t) -> libc::c_int;
    fn gethostbyname(name: *const libc::c_char) -> *mut HostentRaw;
    fn gethostbyaddr(
        addr: *const libc::c_void,
        len: libc::socklen_t,
        family: libc::c_int,
    ) -> *mut HostentRaw;
    fn getservbyname(name: *const libc::c_char, proto: *const libc::c_char) -> *mut ServentRaw;
    fn getservbyport(port: libc::c_int, proto: *const libc::c_char) -> *mut ServentRaw;
}

/// Minimal mirror of `struct hostent` — we only read `h_addr_list[0]`
/// and `h_length`, so the rest can stay opaque.
#[cfg(unix)]
#[repr(C)]
#[allow(non_snake_case, dead_code)]
struct HostentRaw {
    h_name: *const libc::c_char,
    h_aliases: *mut *mut libc::c_char,
    h_addrtype: libc::c_int,
    h_length: libc::c_int,
    h_addr_list: *mut *mut libc::c_char,
}

/// Minimal mirror of `struct servent` — we read `s_name` and `s_port`.
#[cfg(unix)]
#[repr(C)]
#[allow(non_snake_case, dead_code)]
struct ServentRaw {
    s_name: *const libc::c_char,
    s_aliases: *mut *mut libc::c_char,
    s_port: libc::c_int,
    s_proto: *const libc::c_char,
}

/// _socket module — PyPy: pypy/module/_socket/.
///
/// **Slice S1: constants + name resolution helpers.**
///
/// Provides the AF_* / SOCK_* / IPPROTO_* / SOL_* / SO_* / SHUT_* /
/// AI_* / NI_* / IPV4-IPV6 constants plus the small "lookup" helpers
/// gethostname / sethostname / inet_aton / inet_ntoa / inet_pton /
/// inet_ntop / htons / htonl / ntohs / ntohl / getservbyname /
/// getservbyport / gethostbyname.
///
/// Does NOT yet provide the `socket` class itself — that requires
/// per-instance heap state (the OwnedFd + family/type/proto triple) and
/// is the next slice (S2).  Until then `import socket` succeeds and the
/// constants/helpers above are usable, but `socket.socket(...)` raises
/// the C-extension stub error.

/// `interp_socket.py:1066-1084 converted_error` — turn an rsocket
/// `SocketError` subclass into the matching python-level exception.
///
/// `applevelerrcls` matches the field defined on each rsocket error
/// class (`rpython/rlib/rsocket.py:1316/1360/1372/1383`):
///   "error"    → builtin `OSError`
///   "gaierror" → `_socket.gaierror` (OSError subclass)
///   "herror"   → `_socket.herror`   (OSError subclass)
///   "timeout"  → builtin `TimeoutError` (per `get_error()` line 1062-3,
///                NOT the `_socket.timeout` attribute, which is a
///                separate OSError subclass exposed for `isinstance` use)
///
/// When `errno` is `Some`, builds the exception with `(errno, message)`
/// like `SocketErrorWithErrno` (`interp_socket.py:1074-1075`); otherwise
/// only `(message,)` like the plain SocketError (`:1077-1078`).
/// `interp_socket.py:102-123 idna_converter` — turn a hostname argument
/// into a `Vec<u8>` suitable for passing to a DNS resolver.
///
/// Accepts str / bytes / bytearray.  For str: tries ASCII first; on
/// UnicodeEncodeError falls back to `.encode('idna')`.  Embedded null
/// bytes raise TypeError (matching `:120-122`).  Other input types
/// raise TypeError.
///
/// pyre's `idna` codec presently passes through as UTF-8 instead of
/// emitting punycode, so non-ASCII hostnames still pass through this
/// helper without raising but produce incorrect DNS queries — that is
/// an `encodings/idna` gap, not a `_socket` parity issue.
#[cfg(unix)]
fn socket_idna_converter(w_host: pyre_object::PyObjectRef) -> Result<Vec<u8>, crate::PyError> {
    if w_host.is_null() {
        return Err(crate::PyError::type_error(
            "string or unicode text buffer expected, not None",
        ));
    }
    let bytes: Vec<u8> = unsafe {
        if pyre_object::is_str(w_host) {
            let s = pyre_object::w_str_get_value(w_host);
            if s.is_ascii() {
                s.as_bytes().to_vec()
            } else {
                let method = crate::baseobjspace::getattr_str(w_host, "encode")?;
                let codec = pyre_object::w_str_new("idna");
                let encoded = crate::call_function(method, &[codec]);
                if encoded.is_null() {
                    return Err(crate::PyError::type_error("idna encoding failed"));
                }
                if !pyre_object::bytesobject::is_bytes_like(encoded) {
                    return Err(crate::PyError::type_error(
                        "idna encode did not return bytes",
                    ));
                }
                pyre_object::w_bytes_data(encoded).to_vec()
            }
        } else if pyre_object::bytesobject::is_bytes_like(w_host) {
            pyre_object::w_bytes_data(w_host).to_vec()
        } else {
            return Err(crate::PyError::type_error(
                "string or unicode text buffer expected",
            ));
        }
    };
    if bytes.contains(&0) {
        return Err(crate::PyError::type_error(
            "host name must not contain null character",
        ));
    }
    Ok(bytes)
}

#[cfg(unix)]
fn socket_converted_error(
    applevelerrcls: &str,
    errno: Option<i32>,
    message: &str,
) -> crate::PyError {
    let cls = match applevelerrcls {
        "timeout" => crate::builtins::lookup_exc_class("TimeoutError"),
        "gaierror" => crate::builtins::lookup_exc_class("_socket.gaierror"),
        "herror" => crate::builtins::lookup_exc_class("_socket.herror"),
        _ => crate::builtins::lookup_exc_class("OSError"),
    }
    .or_else(|| crate::builtins::lookup_exc_class("OSError"))
    .expect("OSError must be installed");

    let mut args = vec![cls];
    if let Some(e) = errno {
        args.push(pyre_object::w_int_new(e as i64));
    }
    args.push(pyre_object::w_str_new(message));

    let exc = crate::builtins::exc_exception_new(&args)
        .expect("exc_exception_new is infallible for str/int args");

    let mut err = crate::PyError::os_error(message);
    err.exc_object = exc;
    err
}

/// baseobjspace.py `writebuf_w` — the writable byte slice backing a
/// scatter/gather buffer argument.  PyPy accepts any object exporting a
/// writable buffer; pyre's writable byte stores are `bytearray` and a
/// `memoryview` over one, so those are resolved here and anything else is
/// rejected as `writebuf_w` does.
#[cfg(unix)]
fn socket_writebuf(obj: pyre_object::PyObjectRef) -> Result<&'static mut [u8], crate::PyError> {
    if unsafe { pyre_object::bytearrayobject::is_bytearray(obj) } {
        return Ok(unsafe { pyre_object::bytearrayobject::w_bytearray_data_mut(obj) });
    }
    if unsafe { pyre_object::interp_array::is_array(obj) } {
        // `space.writebuf_w` accepts any writable buffer exporter; an
        // `array.array` exposes its element bytes as one writable window
        // regardless of typecode (recv writes raw bytes into them).
        return Ok(unsafe { pyre_object::interp_array::w_array_vec_mut(obj).as_mut_slice() });
    }
    if unsafe { pyre_object::memoryview::is_w_memoryview(obj) } {
        // `space.buffer_w` rejects a released view before exposing its storage.
        unsafe { crate::builtins::memoryview_check_released(obj) }?;
        // A read-write buffer is required; a read-only view cannot back recv_into.
        if unsafe { pyre_object::memoryview::w_memoryview_readonly(obj) } {
            return Err(crate::PyError::type_error(
                "a read-write bytes-like object is required, not 'memoryview'",
            ));
        }
        // Only C-contiguous views are accepted; a strided slice (`m[::2]`,
        // `m[::-1]`) would need a scatter writer pyre does not have.  A
        // contiguous N-D view (`memoryview(ba).cast('B', shape=(2, 2))`)
        // exposes its window as one flat byte range, so it qualifies even
        // though its outermost stride is a row stride, not the itemsize.
        if !unsafe { crate::builtins::memoryview_contiguity(obj).0 } {
            return Err(crate::PyError::type_error(
                "a read-write bytes-like object is required, not 'memoryview'",
            ));
        }
        let backing = unsafe { pyre_object::memoryview::w_memoryview_backing(obj) };
        // A writable view's backing is a `bytearray` or an `array.array`; both
        // expose a mutable byte store.  Honour the view window: write only into
        // `[offset, offset+length)` of the backing, not the whole buffer.
        let full: &mut [u8] = if unsafe { pyre_object::bytearrayobject::is_bytearray(backing) } {
            unsafe { pyre_object::bytearrayobject::w_bytearray_data_mut(backing) }
        } else if unsafe { pyre_object::interp_array::is_array(backing) } {
            unsafe { pyre_object::interp_array::w_array_vec_mut(backing).as_mut_slice() }
        } else {
            return Err(crate::PyError::type_error("cannot modify read-only memory"));
        };
        let off = unsafe { pyre_object::memoryview::w_memoryview_offset(obj) } as usize;
        let len = unsafe { pyre_object::memoryview::w_memoryview_length(obj) } as usize;
        // The backing may have been resized after the view was taken; reject a
        // window that no longer fits rather than panic.
        if off.checked_add(len).is_none_or(|end| end > full.len()) {
            return Err(crate::PyError::value_error(
                "memoryview buffer is no longer valid",
            ));
        }
        return Ok(&mut full[off..off + len]);
    }
    Err(crate::PyError::type_error(
        "a writable bytes-like object is required",
    ))
}

pub fn register_module(ns: &mut DictStorage) {
    // `_rsocket_rffi.py:140-220 constant_names` + `:234-262
    // constants_w_defaults` — populated through the libc crate where
    // available, hardcoded for platform-specific constants the crate
    // does not expose.  Mirrors PyPy's
    // `for constant, value in rsocket.constants.iteritems(): wrap(value)`
    // loop in `_socket/moduledef.py:48-50`.
    #[cfg(unix)]
    {
        macro_rules! cst {
            ($name:literal, $val:expr) => {
                crate::dict_storage_store(ns, $name, pyre_object::w_int_new($val as i64));
            };
        }
        // ── Address families ──
        cst!("AF_UNSPEC", libc::AF_UNSPEC);
        cst!("AF_UNIX", libc::AF_UNIX);
        cst!("AF_INET", libc::AF_INET);
        cst!("AF_INET6", libc::AF_INET6);
        cst!("AF_ROUTE", libc::AF_ROUTE);
        #[cfg(any(target_os = "linux", target_os = "android"))]
        {
            cst!("AF_PACKET", libc::AF_PACKET);
            cst!("AF_NETLINK", libc::AF_NETLINK);
            cst!("AF_VSOCK", libc::AF_VSOCK);
        }
        // ── Socket types ──
        cst!("SOCK_STREAM", libc::SOCK_STREAM);
        cst!("SOCK_DGRAM", libc::SOCK_DGRAM);
        cst!("SOCK_RAW", libc::SOCK_RAW);
        cst!("SOCK_RDM", libc::SOCK_RDM);
        cst!("SOCK_SEQPACKET", libc::SOCK_SEQPACKET);
        #[cfg(any(target_os = "linux", target_os = "android"))]
        {
            cst!("SOCK_CLOEXEC", libc::SOCK_CLOEXEC);
            cst!("SOCK_NONBLOCK", libc::SOCK_NONBLOCK);
        }
        // ── Protocols ──
        cst!("IPPROTO_IP", libc::IPPROTO_IP);
        cst!("IPPROTO_HOPOPTS", libc::IPPROTO_HOPOPTS);
        cst!("IPPROTO_ICMP", libc::IPPROTO_ICMP);
        cst!("IPPROTO_IGMP", libc::IPPROTO_IGMP);
        cst!("IPPROTO_IPIP", libc::IPPROTO_IPIP);
        cst!("IPPROTO_TCP", libc::IPPROTO_TCP);
        cst!("IPPROTO_EGP", libc::IPPROTO_EGP);
        cst!("IPPROTO_PUP", libc::IPPROTO_PUP);
        cst!("IPPROTO_UDP", libc::IPPROTO_UDP);
        cst!("IPPROTO_IDP", libc::IPPROTO_IDP);
        cst!("IPPROTO_TP", libc::IPPROTO_TP);
        cst!("IPPROTO_IPV6", libc::IPPROTO_IPV6);
        cst!("IPPROTO_ROUTING", libc::IPPROTO_ROUTING);
        cst!("IPPROTO_FRAGMENT", libc::IPPROTO_FRAGMENT);
        cst!("IPPROTO_ESP", libc::IPPROTO_ESP);
        cst!("IPPROTO_AH", libc::IPPROTO_AH);
        cst!("IPPROTO_ICMPV6", libc::IPPROTO_ICMPV6);
        cst!("IPPROTO_NONE", libc::IPPROTO_NONE);
        cst!("IPPROTO_DSTOPTS", libc::IPPROTO_DSTOPTS);
        cst!("IPPROTO_PIM", libc::IPPROTO_PIM);
        cst!("IPPROTO_SCTP", libc::IPPROTO_SCTP);
        cst!("IPPROTO_RAW", libc::IPPROTO_RAW);
        cst!("IPPROTO_MAX", libc::IPPROTO_MAX);
        cst!("IPPROTO_GRE", libc::IPPROTO_GRE);
        cst!("IPPROTO_RSVP", libc::IPPROTO_RSVP);
        // `_rsocket_rffi.py:234-241 constants_w_defaults` — SOL_IP/TCP/UDP
        // and IPPROTO_* duplicates kept for PyPy compatibility.
        cst!("SOL_IP", 0);
        cst!("SOL_TCP", 6);
        cst!("SOL_UDP", 17);
        // ── INADDR_* (host byte order) ──
        cst!("INADDR_ANY", libc::INADDR_ANY);
        cst!("INADDR_LOOPBACK", libc::INADDR_LOOPBACK);
        cst!("INADDR_BROADCAST", libc::INADDR_BROADCAST);
        cst!("INADDR_NONE", libc::INADDR_NONE);
        cst!("INADDR_ALLHOSTS_GROUP", 0xe0000001u32);
        cst!("INADDR_UNSPEC_GROUP", 0xe0000000u32);
        cst!("INADDR_MAX_LOCAL_GROUP", 0xe00000ffu32);
        cst!("IPPORT_RESERVED", 1024);
        cst!("IPPORT_USERRESERVED", 5000);
        // ── SOL_* / SO_* (socket level) ──
        cst!("SOL_SOCKET", libc::SOL_SOCKET);
        cst!("SO_REUSEADDR", libc::SO_REUSEADDR);
        cst!("SO_REUSEPORT", libc::SO_REUSEPORT);
        cst!("SO_KEEPALIVE", libc::SO_KEEPALIVE);
        cst!("SO_BROADCAST", libc::SO_BROADCAST);
        cst!("SO_DEBUG", libc::SO_DEBUG);
        cst!("SO_DONTROUTE", libc::SO_DONTROUTE);
        cst!("SO_LINGER", libc::SO_LINGER);
        cst!("SO_OOBINLINE", libc::SO_OOBINLINE);
        cst!("SO_RCVBUF", libc::SO_RCVBUF);
        cst!("SO_SNDBUF", libc::SO_SNDBUF);
        cst!("SO_RCVTIMEO", libc::SO_RCVTIMEO);
        cst!("SO_SNDTIMEO", libc::SO_SNDTIMEO);
        cst!("SO_RCVLOWAT", libc::SO_RCVLOWAT);
        cst!("SO_SNDLOWAT", libc::SO_SNDLOWAT);
        cst!("SO_ERROR", libc::SO_ERROR);
        cst!("SO_TYPE", libc::SO_TYPE);
        cst!("SO_ACCEPTCONN", libc::SO_ACCEPTCONN);
        #[cfg(any(target_os = "linux", target_os = "android"))]
        {
            cst!("SO_DOMAIN", libc::SO_DOMAIN);
            cst!("SO_PROTOCOL", libc::SO_PROTOCOL);
            cst!("SO_PEERCRED", libc::SO_PEERCRED);
            cst!("SO_PASSCRED", libc::SO_PASSCRED);
            cst!("SO_PEERSEC", libc::SO_PEERSEC);
            cst!("SO_PASSSEC", libc::SO_PASSSEC);
        }
        // ── TCP-level ──
        cst!("TCP_NODELAY", libc::TCP_NODELAY);
        cst!("TCP_MAXSEG", libc::TCP_MAXSEG);
        #[cfg(any(target_os = "linux", target_os = "android"))]
        {
            cst!("TCP_KEEPIDLE", libc::TCP_KEEPIDLE);
            cst!("TCP_KEEPINTVL", libc::TCP_KEEPINTVL);
            cst!("TCP_KEEPCNT", libc::TCP_KEEPCNT);
            cst!("TCP_CORK", libc::TCP_CORK);
            cst!("TCP_DEFER_ACCEPT", libc::TCP_DEFER_ACCEPT);
            cst!("TCP_INFO", libc::TCP_INFO);
            cst!("TCP_LINGER2", libc::TCP_LINGER2);
            cst!("TCP_QUICKACK", libc::TCP_QUICKACK);
            cst!("TCP_SYNCNT", libc::TCP_SYNCNT);
            cst!("TCP_WINDOW_CLAMP", libc::TCP_WINDOW_CLAMP);
            cst!("TCP_USER_TIMEOUT", libc::TCP_USER_TIMEOUT);
            cst!("TCP_CONGESTION", libc::TCP_CONGESTION);
            cst!("TCP_FASTOPEN", libc::TCP_FASTOPEN);
            cst!("TCP_NOTSENT_LOWAT", libc::TCP_NOTSENT_LOWAT);
        }
        #[cfg(target_os = "macos")]
        {
            cst!("TCP_KEEPALIVE", libc::TCP_KEEPALIVE);
        }
        // ── IP-level ──
        cst!("IP_TTL", libc::IP_TTL);
        cst!("IP_TOS", libc::IP_TOS);
        cst!("IP_MULTICAST_TTL", libc::IP_MULTICAST_TTL);
        cst!("IP_MULTICAST_LOOP", libc::IP_MULTICAST_LOOP);
        cst!("IP_MULTICAST_IF", libc::IP_MULTICAST_IF);
        cst!("IP_ADD_MEMBERSHIP", libc::IP_ADD_MEMBERSHIP);
        cst!("IP_DROP_MEMBERSHIP", libc::IP_DROP_MEMBERSHIP);
        cst!("IP_HDRINCL", libc::IP_HDRINCL);
        // IP_OPTIONS / IP_RECVOPTS / IP_RECVRETOPTS / IP_RETOPTS are
        // POSIX but not exposed by the libc crate on linux/macos;
        // `_rsocket_rffi.py:170-172` lists them, but
        // `platform.DefinedConstantInteger` drops them when the header
        // does not define them.  Same behaviour here — not exposed.
        cst!("IP_DEFAULT_MULTICAST_LOOP", 1);
        cst!("IP_DEFAULT_MULTICAST_TTL", 1);
        cst!("IP_MAX_MEMBERSHIPS", 20);
        // ── IPv6 ──
        cst!("IPV6_V6ONLY", libc::IPV6_V6ONLY);
        cst!("IPV6_MULTICAST_HOPS", libc::IPV6_MULTICAST_HOPS);
        cst!("IPV6_MULTICAST_LOOP", libc::IPV6_MULTICAST_LOOP);
        cst!("IPV6_MULTICAST_IF", libc::IPV6_MULTICAST_IF);
        cst!("IPV6_UNICAST_HOPS", libc::IPV6_UNICAST_HOPS);
        cst!("IPV6_CHECKSUM", libc::IPV6_CHECKSUM);
        // `<netinet/in.h>` IPV6_JOIN_GROUP=20 / IPV6_LEAVE_GROUP=21 on Linux;
        // libc crate omits the symbols on linux-gnu though the kernel headers
        // define them.  Apple / BSD expose them with the BSD numbering (12 /
        // 13) — keep using `libc::*` there for header parity.
        #[cfg(any(target_os = "linux", target_os = "android"))]
        {
            cst!("IPV6_JOIN_GROUP", 20);
            cst!("IPV6_LEAVE_GROUP", 21);
        }
        #[cfg(not(any(target_os = "linux", target_os = "android")))]
        {
            cst!("IPV6_JOIN_GROUP", libc::IPV6_JOIN_GROUP);
            cst!("IPV6_LEAVE_GROUP", libc::IPV6_LEAVE_GROUP);
        }
        cst!("IPV6_RECVTCLASS", libc::IPV6_RECVTCLASS);
        cst!("IPV6_TCLASS", libc::IPV6_TCLASS);
        cst!("IPV6_RECVPKTINFO", libc::IPV6_RECVPKTINFO);
        cst!("IPV6_PKTINFO", libc::IPV6_PKTINFO);
        cst!("IPV6_RECVHOPLIMIT", libc::IPV6_RECVHOPLIMIT);
        cst!("IPV6_HOPLIMIT", libc::IPV6_HOPLIMIT);
        #[cfg(any(target_os = "linux", target_os = "android"))]
        {
            cst!("IPV6_DSTOPTS", libc::IPV6_DSTOPTS);
            cst!("IPV6_HOPOPTS", libc::IPV6_HOPOPTS);
            cst!("IPV6_NEXTHOP", libc::IPV6_NEXTHOP);
            cst!("IPV6_RECVDSTOPTS", libc::IPV6_RECVDSTOPTS);
            cst!("IPV6_RECVHOPOPTS", libc::IPV6_RECVHOPOPTS);
            cst!("IPV6_RECVRTHDR", libc::IPV6_RECVRTHDR);
            cst!("IPV6_RTHDR", libc::IPV6_RTHDR);
            cst!("IPV6_RTHDRDSTOPTS", libc::IPV6_RTHDRDSTOPTS);
            // `<netinet/in.h>` IPV6_RTHDR_TYPE_0=0; symbol omitted from
            // libc crate on linux-gnu but the kernel header defines it.
            cst!("IPV6_RTHDR_TYPE_0", 0);
        }
        // ── shutdown how ──
        cst!("SHUT_RD", libc::SHUT_RD);
        cst!("SHUT_WR", libc::SHUT_WR);
        cst!("SHUT_RDWR", libc::SHUT_RDWR);
        // ── Message flags ──
        cst!("MSG_OOB", libc::MSG_OOB);
        cst!("MSG_PEEK", libc::MSG_PEEK);
        cst!("MSG_DONTROUTE", libc::MSG_DONTROUTE);
        cst!("MSG_DONTWAIT", libc::MSG_DONTWAIT);
        cst!("MSG_WAITALL", libc::MSG_WAITALL);
        cst!("MSG_CTRUNC", libc::MSG_CTRUNC);
        cst!("MSG_TRUNC", libc::MSG_TRUNC);
        cst!("MSG_EOR", libc::MSG_EOR);
        #[cfg(any(target_os = "linux", target_os = "android"))]
        cst!("MSG_ERRQUEUE", libc::MSG_ERRQUEUE);
        // ── Address-info flags ──
        cst!("AI_PASSIVE", libc::AI_PASSIVE);
        cst!("AI_CANONNAME", libc::AI_CANONNAME);
        cst!("AI_NUMERICHOST", libc::AI_NUMERICHOST);
        cst!("AI_NUMERICSERV", libc::AI_NUMERICSERV);
        cst!("AI_ADDRCONFIG", libc::AI_ADDRCONFIG);
        cst!("AI_V4MAPPED", libc::AI_V4MAPPED);
        cst!("AI_ALL", libc::AI_ALL);
        #[cfg(target_os = "macos")]
        {
            cst!("AI_DEFAULT", libc::AI_DEFAULT);
            cst!("AI_MASK", libc::AI_MASK);
            cst!("AI_V4MAPPED_CFG", libc::AI_V4MAPPED_CFG);
        }
        // ── Name-info flags ──
        cst!("NI_NUMERICHOST", libc::NI_NUMERICHOST);
        cst!("NI_NUMERICSERV", libc::NI_NUMERICSERV);
        cst!("NI_NOFQDN", libc::NI_NOFQDN);
        cst!("NI_NAMEREQD", libc::NI_NAMEREQD);
        cst!("NI_DGRAM", libc::NI_DGRAM);
        cst!("NI_MAXHOST", libc::NI_MAXHOST);
        // POSIX <netdb.h> NI_MAXSERV = 32; libc crate omits it on linux-gnu
        cst!("NI_MAXSERV", 32);
        // ── EAI_* (gai_strerror codes) ──
        cst!("EAI_AGAIN", libc::EAI_AGAIN);
        cst!("EAI_BADFLAGS", libc::EAI_BADFLAGS);
        cst!("EAI_FAIL", libc::EAI_FAIL);
        cst!("EAI_FAMILY", libc::EAI_FAMILY);
        cst!("EAI_MEMORY", libc::EAI_MEMORY);
        cst!("EAI_NODATA", libc::EAI_NODATA);
        cst!("EAI_NONAME", libc::EAI_NONAME);
        cst!("EAI_OVERFLOW", libc::EAI_OVERFLOW);
        cst!("EAI_SERVICE", libc::EAI_SERVICE);
        cst!("EAI_SOCKTYPE", libc::EAI_SOCKTYPE);
        cst!("EAI_SYSTEM", libc::EAI_SYSTEM);
        // EAI_ADDRFAMILY / EAI_BADHINTS / EAI_PROTOCOL / EAI_MAX exist
        // on macOS at the system-header level but the libc crate does
        // not export them; PyPy filters them out via
        // `platform.DefinedConstantInteger` on platforms where they are
        // absent, so we mirror that and skip.
        // ── SCM_* (ancillary data types) ──
        cst!("SCM_RIGHTS", libc::SCM_RIGHTS);
        #[cfg(any(target_os = "linux", target_os = "android"))]
        cst!("SCM_CREDENTIALS", libc::SCM_CREDENTIALS);
        // ── socket-level cap ──
        cst!("SOMAXCONN", libc::SOMAXCONN);
    }

    // ── htons / htonl / ntohs / ntohl ──
    crate::dict_storage_store(
        ns,
        "htons",
        crate::make_builtin_function_with_arity(
            "htons",
            |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error("htons() missing argument"));
                }
                let x = (unsafe { pyre_object::w_int_get_value(args[0]) }) as u16;
                Ok(pyre_object::w_int_new(x.to_be() as i64))
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "ntohs",
        crate::make_builtin_function_with_arity(
            "ntohs",
            |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error("ntohs() missing argument"));
                }
                let x = (unsafe { pyre_object::w_int_get_value(args[0]) }) as u16;
                Ok(pyre_object::w_int_new(u16::from_be(x) as i64))
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "htonl",
        crate::make_builtin_function_with_arity(
            "htonl",
            |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error("htonl() missing argument"));
                }
                let x = (unsafe { pyre_object::w_int_get_value(args[0]) }) as u32;
                Ok(pyre_object::w_int_new(x.to_be() as i64))
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "ntohl",
        crate::make_builtin_function_with_arity(
            "ntohl",
            |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error("ntohl() missing argument"));
                }
                let x = (unsafe { pyre_object::w_int_get_value(args[0]) }) as u32;
                Ok(pyre_object::w_int_new(u32::from_be(x) as i64))
            },
            1,
        ),
    );

    // ── inet_aton / inet_ntoa ──
    #[cfg(unix)]
    {
        crate::dict_storage_store(
            ns,
            "inet_aton",
            crate::make_builtin_function_with_arity(
                "inet_aton",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("inet_aton() missing argument"));
                    }
                    let s = unsafe {
                        if !pyre_object::is_str(args[0]) {
                            return Err(crate::PyError::type_error(
                                "inet_aton: arg must be a string",
                            ));
                        }
                        pyre_object::w_str_get_value(args[0]).to_string()
                    };
                    let c = std::ffi::CString::new(s.as_bytes())
                        .map_err(|_| crate::PyError::value_error("embedded null in argument"))?;
                    let mut addr: libc::in_addr = unsafe { std::mem::zeroed() };
                    let r = unsafe { inet_aton(c.as_ptr(), &mut addr) };
                    if r == 0 {
                        return Err(crate::PyError::os_error(
                            "illegal IP address string passed to inet_aton",
                        ));
                    }
                    let bytes = addr.s_addr.to_ne_bytes();
                    Ok(pyre_object::bytesobject::w_bytes_from_bytes(&bytes))
                },
                1,
            ),
        );
        crate::dict_storage_store(
            ns,
            "inet_ntoa",
            crate::make_builtin_function_with_arity(
                "inet_ntoa",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("inet_ntoa() missing argument"));
                    }
                    let data = unsafe {
                        if !pyre_object::bytesobject::is_bytes_like(args[0]) {
                            return Err(crate::PyError::type_error(
                                "inet_ntoa: argument must be bytes-like",
                            ));
                        }
                        pyre_object::bytesobject::bytes_like_data(args[0])
                    };
                    if data.len() != 4 {
                        return Err(crate::PyError::os_error(
                            "packed IP wrong length for inet_ntoa",
                        ));
                    }
                    let addr = libc::in_addr {
                        s_addr: u32::from_ne_bytes([data[0], data[1], data[2], data[3]]),
                    };
                    let p = unsafe { inet_ntoa(addr) };
                    if p.is_null() {
                        return Err(crate::PyError::os_error("inet_ntoa failed"));
                    }
                    let cs = unsafe { std::ffi::CStr::from_ptr(p) };
                    Ok(pyre_object::w_str_new(&cs.to_string_lossy()))
                },
                1,
            ),
        );

        // inet_pton(af, ip) → bytes
        crate::dict_storage_store(
            ns,
            "inet_pton",
            crate::make_builtin_function_with_arity(
                "inet_pton",
                |args| {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error(
                            "inet_pton() requires 2 arguments",
                        ));
                    }
                    let af = (unsafe { pyre_object::w_int_get_value(args[0]) }) as libc::c_int;
                    let ip = unsafe {
                        if !pyre_object::is_str(args[1]) {
                            return Err(crate::PyError::type_error(
                                "inet_pton: address must be a string",
                            ));
                        }
                        pyre_object::w_str_get_value(args[1]).to_string()
                    };
                    let c_ip = std::ffi::CString::new(ip.as_bytes())
                        .map_err(|_| crate::PyError::value_error("embedded null"))?;
                    let mut buf = [0u8; 16];
                    let r = unsafe {
                        inet_pton(af, c_ip.as_ptr(), buf.as_mut_ptr() as *mut libc::c_void)
                    };
                    if r != 1 {
                        return Err(crate::PyError::os_error(
                            "illegal IP address string passed to inet_pton",
                        ));
                    }
                    let n = match af {
                        x if x == libc::AF_INET => 4,
                        x if x == libc::AF_INET6 => 16,
                        _ => {
                            return Err(crate::PyError::value_error("unknown address family"));
                        }
                    };
                    Ok(pyre_object::bytesobject::w_bytes_from_bytes(&buf[..n]))
                },
                2,
            ),
        );

        // inet_ntop(af, packed) → str
        crate::dict_storage_store(
            ns,
            "inet_ntop",
            crate::make_builtin_function_with_arity(
                "inet_ntop",
                |args| {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error(
                            "inet_ntop() requires 2 arguments",
                        ));
                    }
                    let af = (unsafe { pyre_object::w_int_get_value(args[0]) }) as libc::c_int;
                    let data = unsafe {
                        if !pyre_object::bytesobject::is_bytes_like(args[1]) {
                            return Err(crate::PyError::type_error(
                                "inet_ntop: argument must be bytes-like",
                            ));
                        }
                        pyre_object::bytesobject::bytes_like_data(args[1])
                    };
                    let expected = match af {
                        x if x == libc::AF_INET => 4,
                        x if x == libc::AF_INET6 => 16,
                        _ => {
                            return Err(crate::PyError::value_error("unknown address family"));
                        }
                    };
                    if data.len() != expected {
                        return Err(crate::PyError::value_error(
                            "invalid length of packed IP address string",
                        ));
                    }
                    let mut buf = [0u8; 64];
                    let r = unsafe {
                        inet_ntop(
                            af,
                            data.as_ptr() as *const libc::c_void,
                            buf.as_mut_ptr() as *mut libc::c_char,
                            buf.len() as libc::socklen_t,
                        )
                    };
                    if r.is_null() {
                        return Err(crate::PyError::os_error("inet_ntop failed"));
                    }
                    let s = unsafe { std::ffi::CStr::from_ptr(r) };
                    Ok(pyre_object::w_str_new(&s.to_string_lossy()))
                },
                2,
            ),
        );

        // gethostname() → str
        crate::dict_storage_store(
            ns,
            "gethostname",
            crate::make_builtin_function_with_arity(
                "gethostname",
                |_| {
                    let mut buf = [0u8; 256];
                    let r =
                        unsafe { gethostname(buf.as_mut_ptr() as *mut libc::c_char, buf.len()) };
                    if r != 0 {
                        return Err(crate::PyError::os_error_with_errno(
                            std::io::Error::last_os_error().raw_os_error().unwrap_or(0),
                            "gethostname",
                        ));
                    }
                    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
                    Ok(pyre_object::w_str_new(&String::from_utf8_lossy(
                        &buf[..end],
                    )))
                },
                0,
            ),
        );

        // sethostname(name) → None  (host_env::socket-backed)
        #[cfg(feature = "host_env")]
        crate::dict_storage_store(
            ns,
            "sethostname",
            crate::make_builtin_function_with_arity(
                "sethostname",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error(
                            "sethostname() requires 1 argument",
                        ));
                    }
                    let name = unsafe {
                        if !pyre_object::is_str(args[0]) {
                            return Err(crate::PyError::type_error(
                                "sethostname: name must be a string",
                            ));
                        }
                        pyre_object::w_str_get_value(args[0]).to_string()
                    };
                    rustpython_host_env::socket::sethostname(&name).map_err(|e| {
                        crate::PyError::os_error_with_errno(
                            e.raw_os_error().unwrap_or(0),
                            format!("sethostname: {e}"),
                        )
                    })?;
                    Ok(pyre_object::w_none())
                },
                1,
            ),
        );

        // gethostbyname(name) → ip_string.  `interp_func.py:32-44` —
        // host argument runs through encode_idna (→ idna_converter)
        // before the rsocket call.
        crate::dict_storage_store(
            ns,
            "gethostbyname",
            crate::make_builtin_function_with_arity(
                "gethostbyname",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error(
                            "gethostbyname() missing argument",
                        ));
                    }
                    let host_bytes = socket_idna_converter(args[0])?;
                    let c = std::ffi::CString::new(host_bytes.clone())
                        .map_err(|_| crate::PyError::value_error("embedded null"))?;
                    let he = unsafe { gethostbyname(c.as_ptr()) };
                    if he.is_null() {
                        let host_repr = String::from_utf8_lossy(&host_bytes).into_owned();
                        return Err(socket_converted_error(
                            "gaierror",
                            None,
                            &format!("gethostbyname failed for {host_repr}"),
                        ));
                    }
                    unsafe {
                        let h = &*he;
                        if h.h_length != 4 || (*h.h_addr_list).is_null() {
                            return Err(socket_converted_error(
                                "gaierror",
                                None,
                                "gethostbyname: no IPv4 address",
                            ));
                        }
                        let addr_ptr = *h.h_addr_list;
                        let addr = libc::in_addr {
                            s_addr: *(addr_ptr as *const u32),
                        };
                        let p = inet_ntoa(addr);
                        Ok(pyre_object::w_str_new(
                            &std::ffi::CStr::from_ptr(p).to_string_lossy(),
                        ))
                    }
                },
                1,
            ),
        );

        // gethostbyname_ex(name) → (name, aliases, addresses)
        // `interp_func.py:53-65` — same lookup as gethostbyname but
        // returns the full hostent triple.
        crate::dict_storage_store(
            ns,
            "gethostbyname_ex",
            crate::make_builtin_function_with_arity(
                "gethostbyname_ex",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error(
                            "gethostbyname_ex() missing argument",
                        ));
                    }
                    let host_bytes = socket_idna_converter(args[0])?;
                    let c = std::ffi::CString::new(host_bytes.clone())
                        .map_err(|_| crate::PyError::value_error("embedded null"))?;
                    let he = unsafe { gethostbyname(c.as_ptr()) };
                    if he.is_null() {
                        let host_repr = String::from_utf8_lossy(&host_bytes).into_owned();
                        return Err(socket_converted_error(
                            "gaierror",
                            None,
                            &format!("gethostbyname_ex failed for {host_repr}"),
                        ));
                    }
                    unpack_hostent(he)
                },
                1,
            ),
        );

        // gethostbyaddr(addr) → (name, aliases, addresses)
        // `interp_func.py:67-79` — reverse lookup; `addr` is an
        // IPv4/IPv6 string we resolve through inet_pton, then feed
        // to gethostbyaddr.
        crate::dict_storage_store(
            ns,
            "gethostbyaddr",
            crate::make_builtin_function_with_arity(
                "gethostbyaddr",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error(
                            "gethostbyaddr() missing argument",
                        ));
                    }
                    let host_bytes = socket_idna_converter(args[0])?;
                    let c = std::ffi::CString::new(host_bytes.clone())
                        .map_err(|_| crate::PyError::value_error("embedded null"))?;
                    // Try IPv4 first, then IPv6, then fall back to
                    // gethostbyname → hostent.h_addr to obtain a raw
                    // bytestring for gethostbyaddr.
                    let mut buf4 = [0u8; 4];
                    let r4 = unsafe {
                        inet_pton(
                            libc::AF_INET,
                            c.as_ptr(),
                            buf4.as_mut_ptr() as *mut libc::c_void,
                        )
                    };
                    let (family, addr_ptr, addr_len) = if r4 == 1 {
                        (
                            libc::AF_INET,
                            buf4.as_ptr() as *const libc::c_void,
                            4 as libc::socklen_t,
                        )
                    } else {
                        let mut buf6 = [0u8; 16];
                        let r6 = unsafe {
                            inet_pton(
                                libc::AF_INET6,
                                c.as_ptr(),
                                buf6.as_mut_ptr() as *mut libc::c_void,
                            )
                        };
                        if r6 == 1 {
                            // Borrowed pointer: we copy into a stable
                            // buffer below so the lifetime crosses the
                            // FFI call safely.
                            let mut owned: [u8; 16] = buf6;
                            let he = unsafe {
                                gethostbyaddr(
                                    owned.as_mut_ptr() as *mut libc::c_void,
                                    16 as libc::socklen_t,
                                    libc::AF_INET6,
                                )
                            };
                            if he.is_null() {
                                let host_repr = String::from_utf8_lossy(&host_bytes).into_owned();
                                return Err(socket_converted_error(
                                    "herror",
                                    None,
                                    &format!("gethostbyaddr failed for {host_repr}"),
                                ));
                            }
                            return unpack_hostent(he);
                        }
                        // Fall back: name → hostent → first IPv4 addr
                        let he = unsafe { gethostbyname(c.as_ptr()) };
                        if he.is_null() {
                            let host_repr = String::from_utf8_lossy(&host_bytes).into_owned();
                            return Err(socket_converted_error(
                                "herror",
                                None,
                                &format!("gethostbyaddr failed for {host_repr}"),
                            ));
                        }
                        unsafe {
                            let h = &*he;
                            if (*h.h_addr_list).is_null() {
                                return Err(socket_converted_error(
                                    "herror",
                                    None,
                                    "gethostbyaddr: empty address list",
                                ));
                            }
                            (
                                h.h_addrtype as libc::c_int,
                                *h.h_addr_list as *const libc::c_void,
                                h.h_length as libc::socklen_t,
                            )
                        }
                    };
                    let he = unsafe { gethostbyaddr(addr_ptr, addr_len, family) };
                    if he.is_null() {
                        let host_repr = String::from_utf8_lossy(&host_bytes).into_owned();
                        return Err(socket_converted_error(
                            "herror",
                            None,
                            &format!("gethostbyaddr failed for {host_repr}"),
                        ));
                    }
                    unpack_hostent(he)
                },
                1,
            ),
        );

        // getservbyname(name[, proto]) → port
        crate::dict_storage_store(
            ns,
            "getservbyname",
            crate::make_builtin_function("getservbyname", |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error(
                        "getservbyname() missing argument",
                    ));
                }
                let name = unsafe {
                    if !pyre_object::is_str(args[0]) {
                        return Err(crate::PyError::type_error(
                            "getservbyname: name must be a string",
                        ));
                    }
                    pyre_object::w_str_get_value(args[0]).to_string()
                };
                let c_name = std::ffi::CString::new(name.as_bytes())
                    .map_err(|_| crate::PyError::value_error("embedded null"))?;
                let proto_c: Option<std::ffi::CString> =
                    if args.len() >= 2 && unsafe { pyre_object::is_str(args[1]) } {
                        let p = unsafe { pyre_object::w_str_get_value(args[1]).to_string() };
                        Some(
                            std::ffi::CString::new(p.as_bytes())
                                .map_err(|_| crate::PyError::value_error("embedded null"))?,
                        )
                    } else {
                        None
                    };
                let p = unsafe {
                    getservbyname(
                        c_name.as_ptr(),
                        proto_c
                            .as_ref()
                            .map(|c| c.as_ptr())
                            .unwrap_or(std::ptr::null()),
                    )
                };
                if p.is_null() {
                    return Err(socket_converted_error(
                        "error",
                        None,
                        &format!("service/proto not found: {name}"),
                    ));
                }
                let port = unsafe { u16::from_be((*p).s_port as u16) };
                Ok(pyre_object::w_int_new(port as i64))
            }),
        );

        // getservbyport(port[, proto]) → name
        crate::dict_storage_store(
            ns,
            "getservbyport",
            crate::make_builtin_function("getservbyport", |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error(
                        "getservbyport() missing argument",
                    ));
                }
                let port = (unsafe { pyre_object::w_int_get_value(args[0]) }) as u16;
                let proto_c: Option<std::ffi::CString> =
                    if args.len() >= 2 && unsafe { pyre_object::is_str(args[1]) } {
                        let p = unsafe { pyre_object::w_str_get_value(args[1]).to_string() };
                        Some(
                            std::ffi::CString::new(p.as_bytes())
                                .map_err(|_| crate::PyError::value_error("embedded null"))?,
                        )
                    } else {
                        None
                    };
                let p = unsafe {
                    getservbyport(
                        port.to_be() as libc::c_int,
                        proto_c
                            .as_ref()
                            .map(|c| c.as_ptr())
                            .unwrap_or(std::ptr::null()),
                    )
                };
                if p.is_null() {
                    return Err(socket_converted_error(
                        "error",
                        None,
                        &format!("port/proto not found: {port}"),
                    ));
                }
                let name = unsafe {
                    std::ffi::CStr::from_ptr((*p).s_name)
                        .to_string_lossy()
                        .into_owned()
                };
                Ok(pyre_object::w_str_new(&name))
            }),
        );
    }

    // `interp_socket.py:1041-1063 SocketAPI`:
    //   error    = w_OSError                       (alias)
    //   herror   = new_exception_class("_socket.herror",   w_OSError)
    //   gaierror = new_exception_class("_socket.gaierror", w_OSError)
    //   timeout  = new_exception_class("_socket.timeout",  w_OSError)
    let w_os_error = crate::builtins::lookup_exc_class("OSError")
        .expect("OSError must be installed before _socket init");
    crate::dict_storage_store(ns, "error", w_os_error);
    crate::dict_storage_store(
        ns,
        "herror",
        crate::builtins::make_exc_type(
            "_socket.herror",
            crate::builtins::exc_exception_new,
            w_os_error,
        ),
    );
    crate::dict_storage_store(
        ns,
        "gaierror",
        crate::builtins::make_exc_type(
            "_socket.gaierror",
            crate::builtins::exc_exception_new,
            w_os_error,
        ),
    );
    crate::dict_storage_store(
        ns,
        "timeout",
        crate::builtins::make_exc_type(
            "_socket.timeout",
            crate::builtins::exc_exception_new,
            w_os_error,
        ),
    );

    // Default timeout (None) — modulus has a getter/setter; we just stash
    // a None so attribute lookups succeed.
    crate::dict_storage_store(ns, "_default_timeout", pyre_object::w_none());

    // `_rsocket_rffi.py:1155 constants['has_ipv6'] = True` — exposed by
    // PyPy's moduledef.py constants loop as a module-level boolean.
    crate::dict_storage_store(ns, "has_ipv6", pyre_object::boolobject::w_bool_from(true));

    // ── module-level getdefaulttimeout / setdefaulttimeout ──
    // `interp_func.py:378-397` — None means "blocking", float means
    // "timeout in seconds".  Stored as a process-wide cell.
    crate::dict_storage_store(
        ns,
        "getdefaulttimeout",
        crate::make_builtin_function_with_arity(
            "getdefaulttimeout",
            |_| Ok(get_default_socket_timeout()),
            0,
        ),
    );
    crate::dict_storage_store(
        ns,
        "setdefaulttimeout",
        crate::make_builtin_function_with_arity(
            "setdefaulttimeout",
            |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error(
                        "setdefaulttimeout() missing argument",
                    ));
                }
                let v = args[0];
                if unsafe { pyre_object::is_none(v) } {
                    set_default_socket_timeout(None);
                    return Ok(pyre_object::w_none());
                }
                let secs = unsafe {
                    if pyre_object::is_int(v) {
                        pyre_object::w_int_get_value(v) as f64
                    } else if pyre_object::is_float(v) {
                        pyre_object::floatobject::w_float_get_value(v)
                    } else {
                        return Err(crate::PyError::type_error(
                            "setdefaulttimeout: value must be a float or None",
                        ));
                    }
                };
                if secs < 0.0 || !secs.is_finite() {
                    return Err(crate::PyError::value_error("Timeout value out of range"));
                }
                set_default_socket_timeout(Some(secs));
                Ok(pyre_object::w_none())
            },
            1,
        ),
    );

    // ── module-level close(fd) ──
    // `interp_socket.py:close(fd)` — raw libc close, used for fd
    // cleanup when callers obtain a bare fd via .detach().
    #[cfg(unix)]
    crate::dict_storage_store(
        ns,
        "close",
        crate::make_builtin_function_with_arity(
            "close",
            |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error("close() missing fd"));
                }
                if !unsafe { pyre_object::is_int(args[0]) } {
                    return Err(crate::PyError::type_error("close: fd must be an integer"));
                }
                let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as libc::c_int;
                let r = unsafe { libc::close(fd) };
                if r != 0 {
                    return Err(socket_io_err(std::io::Error::last_os_error()));
                }
                Ok(pyre_object::w_none())
            },
            1,
        ),
    );

    // ── getprotobyname(name) ──
    // `interp_func.py:125-134` — returns the IPPROTO_* number for a
    // protocol name.  libc getprotobyname returns NULL on lookup
    // failure; we surface that as OSError to match `converted_error`.
    #[cfg(unix)]
    crate::dict_storage_store(
        ns,
        "getprotobyname",
        crate::make_builtin_function_with_arity(
            "getprotobyname",
            |args| {
                if args.is_empty() || !unsafe { pyre_object::is_str(args[0]) } {
                    return Err(crate::PyError::type_error(
                        "getprotobyname: name must be a string",
                    ));
                }
                let name = unsafe { pyre_object::w_str_get_value(args[0]).to_string() };
                let c_name = std::ffi::CString::new(name.as_bytes())
                    .map_err(|_| crate::PyError::value_error("embedded null in name"))?;
                let pe = unsafe { libc::getprotobyname(c_name.as_ptr()) };
                if pe.is_null() {
                    return Err(socket_converted_error("error", None, "protocol not found"));
                }
                let proto = unsafe { (*pe).p_proto };
                Ok(pyre_object::w_int_new(proto as i64))
            },
            1,
        ),
    );

    // ── if_nameindex / if_nametoindex / if_indextoname ──
    // `interp_socket.py:if_nameindex|if_nametoindex|if_indextoname`
    // — direct wrappers around libc's network-interface accessors.
    #[cfg(unix)]
    {
        crate::dict_storage_store(
            ns,
            "if_nameindex",
            crate::make_builtin_function_with_arity(
                "if_nameindex",
                |_| {
                    let head = unsafe { libc::if_nameindex() };
                    if head.is_null() {
                        return Err(socket_io_err(std::io::Error::last_os_error()));
                    }
                    let mut items = Vec::new();
                    let mut p = head;
                    unsafe {
                        while (*p).if_index != 0 && !(*p).if_name.is_null() {
                            let name = std::ffi::CStr::from_ptr((*p).if_name)
                                .to_string_lossy()
                                .into_owned();
                            items.push(pyre_object::w_tuple_new(vec![
                                pyre_object::w_int_new((*p).if_index as i64),
                                pyre_object::w_str_new(&name),
                            ]));
                            p = p.add(1);
                        }
                        libc::if_freenameindex(head);
                    }
                    Ok(pyre_object::w_list_new(items))
                },
                0,
            ),
        );
        crate::dict_storage_store(
            ns,
            "if_nametoindex",
            crate::make_builtin_function_with_arity(
                "if_nametoindex",
                |args| {
                    if args.is_empty() || !unsafe { pyre_object::is_str(args[0]) } {
                        return Err(crate::PyError::type_error(
                            "if_nametoindex: name must be a string",
                        ));
                    }
                    let name = unsafe { pyre_object::w_str_get_value(args[0]).to_string() };
                    let c_name = std::ffi::CString::new(name.as_bytes())
                        .map_err(|_| crate::PyError::value_error("embedded null in name"))?;
                    let idx = unsafe { libc::if_nametoindex(c_name.as_ptr()) };
                    if idx == 0 {
                        return Err(socket_io_err(std::io::Error::last_os_error()));
                    }
                    Ok(pyre_object::w_int_new(idx as i64))
                },
                1,
            ),
        );
        crate::dict_storage_store(
            ns,
            "if_indextoname",
            crate::make_builtin_function_with_arity(
                "if_indextoname",
                |args| {
                    if args.is_empty() || !unsafe { pyre_object::is_int(args[0]) } {
                        return Err(crate::PyError::type_error(
                            "if_indextoname: index must be an integer",
                        ));
                    }
                    let idx = unsafe { pyre_object::w_int_get_value(args[0]) } as libc::c_uint;
                    let mut buf = [0u8; libc::IF_NAMESIZE];
                    let p =
                        unsafe { libc::if_indextoname(idx, buf.as_mut_ptr() as *mut libc::c_char) };
                    if p.is_null() {
                        return Err(socket_io_err(std::io::Error::last_os_error()));
                    }
                    let s = unsafe { std::ffi::CStr::from_ptr(p) };
                    Ok(pyre_object::w_str_new(&s.to_string_lossy()))
                },
                1,
            ),
        );
    }

    // ── CMSG_SPACE / CMSG_LEN ──
    // `interp_func.py:341-376` — POSIX macros, exposed only when the
    // host libc has them.  rust's `libc` crate provides both on every
    // unix target we ship, so we register them under the same cfg.
    #[cfg(unix)]
    {
        crate::dict_storage_store(
            ns,
            "CMSG_SPACE",
            crate::make_builtin_function_with_arity(
                "CMSG_SPACE",
                |args| {
                    if args.is_empty() || !unsafe { pyre_object::is_int(args[0]) } {
                        return Err(crate::PyError::type_error(
                            "CMSG_SPACE: size must be an integer",
                        ));
                    }
                    let raw = unsafe { pyre_object::w_int_get_value(args[0]) };
                    if raw < 0 {
                        return Err(crate::PyError::overflow_error(
                            "CMSG_SPACE() argument out of range",
                        ));
                    }
                    let n = unsafe { libc::CMSG_SPACE(raw as libc::c_uint) };
                    if n == 0 {
                        return Err(crate::PyError::overflow_error(
                            "CMSG_SPACE() argument out of range",
                        ));
                    }
                    Ok(pyre_object::w_int_new(n as i64))
                },
                1,
            ),
        );
        crate::dict_storage_store(
            ns,
            "CMSG_LEN",
            crate::make_builtin_function_with_arity(
                "CMSG_LEN",
                |args| {
                    if args.is_empty() || !unsafe { pyre_object::is_int(args[0]) } {
                        return Err(crate::PyError::type_error(
                            "CMSG_LEN: length must be an integer",
                        ));
                    }
                    let raw = unsafe { pyre_object::w_int_get_value(args[0]) };
                    if raw < 0 {
                        return Err(crate::PyError::overflow_error(
                            "CMSG_LEN() argument out of range",
                        ));
                    }
                    let n = unsafe { libc::CMSG_LEN(raw as libc::c_uint) };
                    if n == 0 {
                        return Err(crate::PyError::overflow_error(
                            "CMSG_LEN() argument out of range",
                        ));
                    }
                    Ok(pyre_object::w_int_new(n as i64))
                },
                1,
            ),
        );
    }

    // ── getaddrinfo / getnameinfo ──
    // `interp_func.py:294-339` (getaddrinfo) and `:137-156`
    // (getnameinfo) — directly wrap libc's getaddrinfo / getnameinfo
    // and walk the addrinfo linked list.
    #[cfg(unix)]
    init_socket_getaddrinfo(ns);

    // ── socket class (slice S2) ──
    #[cfg(unix)]
    {
        let socket_tp = socket_type();
        // Expose the type itself as `socket` AND `SocketType` so the
        // stdlib's `class socket(_socket.socket):` pattern works.
        crate::dict_storage_store(ns, "socket", socket_tp);
        crate::dict_storage_store(ns, "SocketType", socket_tp);

        // socketpair(family=AF_UNIX, type=SOCK_STREAM, proto=0)
        crate::dict_storage_store(
            ns,
            "socketpair",
            crate::make_builtin_function("socketpair", |args| {
                for (idx, label) in [(0, "family"), (1, "type"), (2, "proto")] {
                    if args.len() > idx && !unsafe { pyre_object::is_int(args[idx]) } {
                        return Err(crate::PyError::type_error(format!(
                            "socketpair: {label} must be an integer"
                        )));
                    }
                }
                let family = if args.is_empty() {
                    libc::AF_UNIX
                } else {
                    unsafe { pyre_object::w_int_get_value(args[0]) as libc::c_int }
                };
                let ty = if args.len() < 2 {
                    libc::SOCK_STREAM
                } else {
                    unsafe { pyre_object::w_int_get_value(args[1]) as libc::c_int }
                };
                let proto = if args.len() < 3 {
                    0
                } else {
                    unsafe { pyre_object::w_int_get_value(args[2]) as libc::c_int }
                };
                let mut fds = [0 as libc::c_int; 2];
                let r = unsafe { libc::socketpair(family, ty, proto, fds.as_mut_ptr()) };
                if r != 0 {
                    return Err(socket_io_err(std::io::Error::last_os_error()));
                }
                // `rsocket.py:socketpair(inheritable=False)` — every
                // socket pyre creates from the module starts with
                // FD_CLOEXEC set, matching CPython's PEP 446 default.
                unsafe {
                    libc::fcntl(fds[0], libc::F_SETFD, libc::FD_CLOEXEC);
                    libc::fcntl(fds[1], libc::F_SETFD, libc::FD_CLOEXEC);
                }
                Ok(pyre_object::w_tuple_new(vec![
                    socket_from_fd(fds[0], family, ty, proto),
                    socket_from_fd(fds[1], family, ty, proto),
                ]))
            }),
        );

        // dup(fd) → new fd.  Per `rsocket.py:dup()` the duplicated
        // descriptor sets FD_CLOEXEC (rsocket goes through dup3+CLOEXEC
        // on Linux; we use the portable fcntl path).
        crate::dict_storage_store(
            ns,
            "dup",
            crate::make_builtin_function_with_arity(
                "dup",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("dup() missing argument"));
                    }
                    if !unsafe { pyre_object::is_int(args[0]) } {
                        return Err(crate::PyError::type_error("dup: fd must be an integer"));
                    }
                    let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as libc::c_int;
                    let n = unsafe { libc::dup(fd) };
                    if n < 0 {
                        return Err(socket_io_err(std::io::Error::last_os_error()));
                    }
                    unsafe {
                        libc::fcntl(n, libc::F_SETFD, libc::FD_CLOEXEC);
                    }
                    Ok(pyre_object::w_int_new(n as i64))
                },
                1,
            ),
        );

        // fromfd(fd, family, type, proto=0) — `interp_func.py:75
        // fromfd_w`: dup() the supplied fd (so the caller still owns the
        // original) and wrap it in a fresh `_socket.socket`.  CPython
        // requires the dup so close() on the returned socket leaves the
        // input descriptor intact.
        crate::dict_storage_store(
            ns,
            "fromfd",
            crate::make_builtin_function("fromfd", |args| {
                if args.len() < 3 {
                    return Err(crate::PyError::type_error(
                        "fromfd() requires fd, family and type",
                    ));
                }
                for (idx, label) in [(0, "fd"), (1, "family"), (2, "type")] {
                    if !unsafe { pyre_object::is_int(args[idx]) } {
                        return Err(crate::PyError::type_error(format!(
                            "fromfd: {label} must be an integer"
                        )));
                    }
                }
                let fd = (unsafe { pyre_object::w_int_get_value(args[0]) }) as libc::c_int;
                let family = (unsafe { pyre_object::w_int_get_value(args[1]) }) as libc::c_int;
                let ty = (unsafe { pyre_object::w_int_get_value(args[2]) }) as libc::c_int;
                let proto = if args.len() >= 4 {
                    if !unsafe { pyre_object::is_int(args[3]) } {
                        return Err(crate::PyError::type_error(
                            "fromfd: proto must be an integer",
                        ));
                    }
                    (unsafe { pyre_object::w_int_get_value(args[3]) }) as libc::c_int
                } else {
                    0
                };
                let new_fd = unsafe { libc::dup(fd) };
                if new_fd < 0 {
                    return Err(socket_io_err(std::io::Error::last_os_error()));
                }
                unsafe {
                    libc::fcntl(new_fd, libc::F_SETFD, libc::FD_CLOEXEC);
                }
                Ok(socket_from_fd(new_fd, family, ty, proto))
            }),
        );
    }
}

// ── hostent → (name, aliases, addrs) ──
// `interp_func.py:46-51 common_wrapgethost` — packs a libc hostent
// into the 3-tuple shape used by gethostbyname_ex / gethostbyaddr.
#[cfg(unix)]
fn unpack_hostent(he: *mut HostentRaw) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    unsafe {
        let h = &*he;
        let name = if h.h_name.is_null() {
            String::new()
        } else {
            std::ffi::CStr::from_ptr(h.h_name)
                .to_string_lossy()
                .into_owned()
        };
        let mut aliases = Vec::new();
        if !h.h_aliases.is_null() {
            let mut p = h.h_aliases;
            while !(*p).is_null() {
                aliases.push(pyre_object::w_str_new(
                    &std::ffi::CStr::from_ptr(*p).to_string_lossy(),
                ));
                p = p.add(1);
            }
        }
        let mut addrs = Vec::new();
        if !h.h_addr_list.is_null() {
            let mut p = h.h_addr_list;
            while !(*p).is_null() {
                let addr_str = if h.h_addrtype == libc::AF_INET && h.h_length == 4 {
                    let addr = libc::in_addr {
                        s_addr: *(*p as *const u32),
                    };
                    let s = inet_ntoa(addr);
                    std::ffi::CStr::from_ptr(s).to_string_lossy().into_owned()
                } else if h.h_addrtype == libc::AF_INET6 && h.h_length == 16 {
                    let mut buf = [0u8; 64];
                    let q = inet_ntop(
                        libc::AF_INET6,
                        *p as *const libc::c_void,
                        buf.as_mut_ptr() as *mut libc::c_char,
                        buf.len() as libc::socklen_t,
                    );
                    if q.is_null() {
                        String::new()
                    } else {
                        std::ffi::CStr::from_ptr(q).to_string_lossy().into_owned()
                    }
                } else {
                    String::new()
                };
                addrs.push(pyre_object::w_str_new(&addr_str));
                p = p.add(1);
            }
        }
        Ok(pyre_object::w_tuple_new(vec![
            pyre_object::w_str_new(&name),
            pyre_object::w_list_new(aliases),
            pyre_object::w_list_new(addrs),
        ]))
    }
}

// ── default socket timeout cell ──
// `rsocket.py:setdefaulttimeout|getdefaulttimeout` — process-wide
// default for socket() construction.  None == blocking; Some(secs)
// == timeout in seconds.

thread_local! {
    static DEFAULT_SOCKET_TIMEOUT: std::cell::Cell<Option<f64>> =
        const { std::cell::Cell::new(None) };
}

fn get_default_socket_timeout() -> pyre_object::PyObjectRef {
    match DEFAULT_SOCKET_TIMEOUT.with(|c| c.get()) {
        None => pyre_object::w_none(),
        Some(s) => pyre_object::floatobject::w_float_new(s),
    }
}

fn set_default_socket_timeout(v: Option<f64>) {
    DEFAULT_SOCKET_TIMEOUT.with(|c| c.set(v));
}

// ── getaddrinfo / getnameinfo wiring ──
//
// PyPy's `interp_func.py:294-339` walks libc's `addrinfo` linked
// list and packs each entry into a 5-tuple `(family, socktype,
// proto, canonname, sockaddr)`.  `getnameinfo` is the symmetric
// path used by stdlib socket.getnameinfo.

#[cfg(unix)]
fn init_socket_getaddrinfo(ns: &mut DictStorage) {
    crate::dict_storage_store(
        ns,
        "getaddrinfo",
        crate::make_builtin_function("getaddrinfo", |args| {
            if args.len() < 2 {
                return Err(crate::PyError::type_error(
                    "getaddrinfo() missing host or port",
                ));
            }
            // host: None | str
            let host_obj = args[0];
            let host: Option<std::ffi::CString> = unsafe {
                if pyre_object::is_none(host_obj) {
                    None
                } else if pyre_object::is_str(host_obj) {
                    let s = pyre_object::w_str_get_value(host_obj).to_string();
                    Some(
                        std::ffi::CString::new(s.as_bytes())
                            .map_err(|_| crate::PyError::value_error("embedded null in host"))?,
                    )
                } else {
                    return Err(crate::PyError::type_error(
                        "getaddrinfo() argument 1 must be string or None",
                    ));
                }
            };
            // port: None | int | str
            let port_obj = args[1];
            let port: Option<std::ffi::CString> = unsafe {
                if pyre_object::is_none(port_obj) {
                    None
                } else if pyre_object::is_int(port_obj) {
                    let v = pyre_object::w_int_get_value(port_obj);
                    Some(std::ffi::CString::new(format!("{v}")).unwrap())
                } else if pyre_object::is_str(port_obj) {
                    let s = pyre_object::w_str_get_value(port_obj).to_string();
                    Some(
                        std::ffi::CString::new(s.as_bytes())
                            .map_err(|_| crate::PyError::value_error("embedded null in port"))?,
                    )
                } else {
                    return Err(crate::PyError::type_error(
                        "getaddrinfo() argument 2 must be integer or string",
                    ));
                }
            };

            let int_arg =
                |idx: usize, default: libc::c_int| -> Result<libc::c_int, crate::PyError> {
                    if args.len() > idx {
                        if !unsafe { pyre_object::is_int(args[idx]) } {
                            return Err(crate::PyError::type_error(
                                "getaddrinfo: family/type/proto/flags must be integers",
                            ));
                        }
                        Ok(unsafe { pyre_object::w_int_get_value(args[idx]) } as libc::c_int)
                    } else {
                        Ok(default)
                    }
                };
            let family = int_arg(2, libc::AF_UNSPEC)?;
            let socktype = int_arg(3, 0)?;
            let proto = int_arg(4, 0)?;
            let flags = int_arg(5, 0)?;

            let mut hints: libc::addrinfo = unsafe { std::mem::zeroed() };
            hints.ai_family = family;
            hints.ai_socktype = socktype;
            hints.ai_protocol = proto;
            hints.ai_flags = flags;

            let mut res: *mut libc::addrinfo = std::ptr::null_mut();
            let host_ptr = host
                .as_ref()
                .map(|c| c.as_ptr())
                .unwrap_or(std::ptr::null());
            let port_ptr = port
                .as_ref()
                .map(|c| c.as_ptr())
                .unwrap_or(std::ptr::null());
            let rc = unsafe { libc::getaddrinfo(host_ptr, port_ptr, &hints, &mut res) };
            if rc != 0 {
                let msg = unsafe {
                    std::ffi::CStr::from_ptr(libc::gai_strerror(rc))
                        .to_string_lossy()
                        .into_owned()
                };
                return Err(socket_converted_error("gaierror", Some(rc), &msg));
            }

            let mut items = Vec::new();
            let mut cur = res;
            unsafe {
                while !cur.is_null() {
                    let ai = &*cur;
                    let canon = if ai.ai_canonname.is_null() {
                        String::new()
                    } else {
                        std::ffi::CStr::from_ptr(ai.ai_canonname)
                            .to_string_lossy()
                            .into_owned()
                    };
                    // Copy sockaddr into our sockaddr_storage so we can
                    // reuse unpack_inet_addr.
                    let mut storage: libc::sockaddr_storage = std::mem::zeroed();
                    let copy_len = (ai.ai_addrlen as usize)
                        .min(core::mem::size_of::<libc::sockaddr_storage>());
                    std::ptr::copy_nonoverlapping(
                        ai.ai_addr as *const u8,
                        &mut storage as *mut _ as *mut u8,
                        copy_len,
                    );
                    let addr = unpack_inet_addr(&storage);
                    items.push(pyre_object::w_tuple_new(vec![
                        pyre_object::w_int_new(ai.ai_family as i64),
                        pyre_object::w_int_new(ai.ai_socktype as i64),
                        pyre_object::w_int_new(ai.ai_protocol as i64),
                        pyre_object::w_str_new(&canon),
                        addr,
                    ]));
                    cur = ai.ai_next;
                }
                libc::freeaddrinfo(res);
            }
            Ok(pyre_object::w_list_new(items))
        }),
    );

    crate::dict_storage_store(
        ns,
        "getnameinfo",
        crate::make_builtin_function_with_arity(
            "getnameinfo",
            |args| {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error(
                        "getnameinfo() requires (sockaddr, flags)",
                    ));
                }
                if !unsafe { pyre_object::is_tuple(args[0]) } {
                    return Err(crate::PyError::type_error(
                        "getnameinfo: sockaddr must be a tuple",
                    ));
                }
                if !unsafe { pyre_object::is_int(args[1]) } {
                    return Err(crate::PyError::type_error(
                        "getnameinfo: flags must be an integer",
                    ));
                }
                let flags = unsafe { pyre_object::w_int_get_value(args[1]) } as libc::c_int;
                // Resolve sockaddr via getaddrinfo(AF_UNSPEC, SOCK_DGRAM,
                // AI_NUMERICHOST) so we get a real sockaddr_storage,
                // matching `interp_func.py:142-152`.
                let host_obj = unsafe { pyre_object::w_tuple_getitem(args[0], 0) }
                    .ok_or_else(|| crate::PyError::value_error("sockaddr: missing host"))?;
                let port_obj = unsafe { pyre_object::w_tuple_getitem(args[0], 1) }
                    .ok_or_else(|| crate::PyError::value_error("sockaddr: missing port"))?;
                if !unsafe { pyre_object::is_str(host_obj) } {
                    return Err(crate::PyError::type_error(
                        "getnameinfo: sockaddr[0] must be a string",
                    ));
                }
                if !unsafe { pyre_object::is_int(port_obj) } {
                    return Err(crate::PyError::type_error(
                        "getnameinfo: sockaddr[1] must be an integer",
                    ));
                }
                let host = unsafe { pyre_object::w_str_get_value(host_obj).to_string() };
                let port_v = unsafe { pyre_object::w_int_get_value(port_obj) };

                let c_host = std::ffi::CString::new(host.as_bytes())
                    .map_err(|_| crate::PyError::value_error("embedded null in host"))?;
                let c_port = std::ffi::CString::new(format!("{port_v}")).unwrap();

                let mut hints: libc::addrinfo = unsafe { std::mem::zeroed() };
                hints.ai_family = libc::AF_UNSPEC;
                hints.ai_socktype = libc::SOCK_DGRAM;
                hints.ai_flags = libc::AI_NUMERICHOST;
                let mut res: *mut libc::addrinfo = std::ptr::null_mut();
                let rc = unsafe {
                    libc::getaddrinfo(c_host.as_ptr(), c_port.as_ptr(), &hints, &mut res)
                };
                if rc != 0 {
                    let msg = unsafe {
                        std::ffi::CStr::from_ptr(libc::gai_strerror(rc))
                            .to_string_lossy()
                            .into_owned()
                    };
                    return Err(socket_converted_error("gaierror", Some(rc), &msg));
                }
                let head = res;
                let ai = unsafe { &*head };
                if !ai.ai_next.is_null() {
                    unsafe { libc::freeaddrinfo(head) };
                    return Err(socket_converted_error(
                        "error",
                        None,
                        "sockaddr resolved to multiple addresses",
                    ));
                }
                let mut host_buf = [0 as libc::c_char; libc::NI_MAXHOST as usize];
                let mut serv_buf = [0 as libc::c_char; 32];
                let nrc = unsafe {
                    libc::getnameinfo(
                        ai.ai_addr,
                        ai.ai_addrlen,
                        host_buf.as_mut_ptr(),
                        host_buf.len() as libc::socklen_t,
                        serv_buf.as_mut_ptr(),
                        serv_buf.len() as libc::socklen_t,
                        flags,
                    )
                };
                unsafe { libc::freeaddrinfo(head) };
                if nrc != 0 {
                    let msg = unsafe {
                        std::ffi::CStr::from_ptr(libc::gai_strerror(nrc))
                            .to_string_lossy()
                            .into_owned()
                    };
                    return Err(socket_converted_error("gaierror", Some(nrc), &msg));
                }
                let host_s = unsafe {
                    std::ffi::CStr::from_ptr(host_buf.as_ptr())
                        .to_string_lossy()
                        .into_owned()
                };
                let serv_s = unsafe {
                    std::ffi::CStr::from_ptr(serv_buf.as_ptr())
                        .to_string_lossy()
                        .into_owned()
                };
                Ok(pyre_object::w_tuple_new(vec![
                    pyre_object::w_str_new(&host_s),
                    pyre_object::w_str_new(&serv_s),
                ]))
            },
            2,
        ),
    );
}

// ── _socket socket() class implementation ─────────────────────────────
//
// Instance state lives in the instance dict under reserved keys
// `_fd` (int) / `_family` (int) / `_type` (int) / `_proto` (int) /
// `_timeout` (float or None).  Methods read/write via baseobjspace.

#[cfg(unix)]
thread_local! {
    static SOCKET_TYPE_OBJ: std::cell::OnceCell<pyre_object::PyObjectRef> =
        const { std::cell::OnceCell::new() };
}

#[cfg(unix)]
fn socket_type() -> pyre_object::PyObjectRef {
    SOCKET_TYPE_OBJ.with(|c| {
        *c.get_or_init(|| {
            let tp = crate::typedef::make_builtin_type("socket", init_socket_type);
            unsafe { pyre_object::typeobject::w_type_set_hasdict(tp, true) };
            tp
        })
    })
}

#[cfg(unix)]
fn socket_io_err(e: std::io::Error) -> crate::PyError {
    let errno = e.raw_os_error().unwrap_or(0);
    // `rsocket.py` carries the C `strerror` text; build the OSError in
    // its `(errno, strerror)` form so `e.errno` and `str(e)` match.
    let strerror = unsafe {
        let p = libc::strerror(errno);
        if p.is_null() {
            format!("Unknown error {errno}")
        } else {
            std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    };
    crate::PyError::os_error_errno_strerror(errno, strerror)
}

#[cfg(unix)]
fn socket_get_attr_i64(obj: pyre_object::PyObjectRef, key: &str) -> i64 {
    let d = crate::baseobjspace::getdict(obj);
    if d.is_null() {
        return -1;
    }
    if let Some(v) = unsafe { pyre_object::w_dict_getitem_str(d, key) } {
        if unsafe { pyre_object::is_int(v) } {
            return unsafe { pyre_object::w_int_get_value(v) };
        }
    }
    -1
}

#[cfg(unix)]
fn socket_set_attr(obj: pyre_object::PyObjectRef, key: &str, v: pyre_object::PyObjectRef) {
    let d = crate::baseobjspace::getdict(obj);
    if d.is_null() {
        return;
    }
    unsafe {
        pyre_object::w_dict_setitem_str(d, key, v);
    }
}

/// `rsocket.py:RSocket.settimeout` — apply a timeout value to a live fd.
///
/// `timeout < 0` (the "None" sentinel) clears `O_NONBLOCK` so the socket
/// blocks indefinitely.  `timeout == 0` flips `O_NONBLOCK` on for
/// non-blocking mode.  `timeout > 0` clears `O_NONBLOCK` and writes the
/// duration to `SO_RCVTIMEO` + `SO_SNDTIMEO` so the kernel returns
/// `EAGAIN`/`EWOULDBLOCK` after the elapsed time.
///
/// Until this helper landed, `settimeout` only stashed the value in the
/// instance dict and `recv`/`send` blocked indefinitely regardless.
#[cfg(unix)]
fn socket_apply_timeout(fd: libc::c_int, timeout: f64) -> Result<(), crate::PyError> {
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFL, 0) };
    if flags < 0 {
        return Err(socket_io_err(std::io::Error::last_os_error()));
    }
    let want_nonblock = timeout == 0.0;
    // Bit-clear without unary `!` so the static analyzer accepts the
    // helper (the analyzer rejects bitwise-not on signed `c_int`).
    let new_flags = if want_nonblock {
        flags | libc::O_NONBLOCK
    } else if (flags & libc::O_NONBLOCK) != 0 {
        flags - libc::O_NONBLOCK
    } else {
        flags
    };
    if new_flags != flags {
        let r = unsafe { libc::fcntl(fd, libc::F_SETFL, new_flags) };
        if r < 0 {
            return Err(socket_io_err(std::io::Error::last_os_error()));
        }
    }
    let tv = if timeout > 0.0 {
        let sec = timeout.trunc() as libc::time_t;
        let usec = ((timeout - timeout.trunc()) * 1_000_000.0).round() as libc::suseconds_t;
        libc::timeval {
            tv_sec: sec,
            tv_usec: usec,
        }
    } else {
        libc::timeval {
            tv_sec: 0,
            tv_usec: 0,
        }
    };
    for opt in [libc::SO_RCVTIMEO, libc::SO_SNDTIMEO] {
        let r = unsafe {
            libc::setsockopt(
                fd,
                libc::SOL_SOCKET,
                opt,
                &tv as *const _ as *const libc::c_void,
                core::mem::size_of::<libc::timeval>() as libc::socklen_t,
            )
        };
        if r != 0 {
            return Err(socket_io_err(std::io::Error::last_os_error()));
        }
    }
    Ok(())
}

#[cfg(unix)]
fn socket_fd(obj: pyre_object::PyObjectRef) -> Result<libc::c_int, crate::PyError> {
    let fd = socket_get_attr_i64(obj, "_fd") as libc::c_int;
    if fd < 0 {
        return Err(crate::PyError::os_error("Bad file descriptor"));
    }
    Ok(fd)
}

#[cfg(unix)]
fn socket_from_fd(
    fd: libc::c_int,
    family: libc::c_int,
    ty: libc::c_int,
    proto: libc::c_int,
) -> pyre_object::PyObjectRef {
    let obj = pyre_object::w_instance_new(socket_type());
    socket_set_attr(obj, "_fd", pyre_object::w_int_new(fd as i64));
    socket_set_attr(obj, "_family", pyre_object::w_int_new(family as i64));
    socket_set_attr(obj, "_type", pyre_object::w_int_new(ty as i64));
    socket_set_attr(obj, "_proto", pyre_object::w_int_new(proto as i64));
    socket_set_attr(obj, "_timeout", pyre_object::w_none());
    // `interp_socket.py:977 usecount = 1` — start the refcount at 1 so
    // `_drop` followed by no `_reuse` closes the underlying fd exactly
    // once.
    socket_set_attr(obj, "_usecount", pyre_object::w_int_new(1));
    obj
}

/// `rsocket.py:1694 get_socket_family` — the family of an existing fd,
/// read from `getsockname`'s returned `sa_family`.
#[cfg(unix)]
fn socket_detect_family(fd: libc::c_int) -> Result<libc::c_int, crate::PyError> {
    let mut addr: libc::sockaddr_storage = unsafe { std::mem::zeroed() };
    let mut len = std::mem::size_of::<libc::sockaddr_storage>() as libc::socklen_t;
    let res =
        unsafe { libc::getsockname(fd, &mut addr as *mut _ as *mut libc::sockaddr, &mut len) };
    if res < 0 {
        return Err(socket_io_err(std::io::Error::last_os_error()));
    }
    Ok(addr.ss_family as libc::c_int)
}

/// `rsocket.py:1678 getsockopt_int` — a single int socket option.
#[cfg(unix)]
fn socket_getsockopt_int(
    fd: libc::c_int,
    level: libc::c_int,
    option: libc::c_int,
) -> Result<libc::c_int, crate::PyError> {
    let mut val: libc::c_int = 0;
    let mut len = std::mem::size_of::<libc::c_int>() as libc::socklen_t;
    let res = unsafe {
        libc::getsockopt(
            fd,
            level,
            option,
            &mut val as *mut _ as *mut libc::c_void,
            &mut len,
        )
    };
    if res < 0 {
        return Err(socket_io_err(std::io::Error::last_os_error()));
    }
    Ok(val)
}

/// `interp_socket.py:93 get_so_protocol` — the protocol of an existing
/// fd via `SO_PROTOCOL`, or `-1` on platforms without it (`HAS_SO_PROTOCOL`).
#[cfg(all(unix, any(target_os = "linux", target_os = "android")))]
fn socket_get_so_protocol(fd: libc::c_int) -> Result<libc::c_int, crate::PyError> {
    socket_getsockopt_int(fd, libc::SOL_SOCKET, libc::SO_PROTOCOL)
}
#[cfg(all(unix, not(any(target_os = "linux", target_os = "android"))))]
fn socket_get_so_protocol(_fd: libc::c_int) -> Result<libc::c_int, crate::PyError> {
    Ok(-1)
}

// ── address pack/unpack helpers ──
//
// Python passes IPv4 addresses as (host, port) tuples and IPv6 as
// (host, port, flowinfo, scopeid).  These helpers convert to/from
// `sockaddr_storage`.

#[cfg(unix)]
fn pack_inet_addr(
    family: libc::c_int,
    addr: pyre_object::PyObjectRef,
) -> Result<(libc::sockaddr_storage, libc::socklen_t), crate::PyError> {
    let mut storage: libc::sockaddr_storage = unsafe { std::mem::zeroed() };
    // AF_UNIX is special: rsocket.py:RSocket.bind/connect accept a bare
    // bytes/str path (or a 1-tuple wrapping the path).  Pull the path
    // out before touching tuple[1], which only the AF_INET/AF_INET6
    // forms guarantee.
    if family == libc::AF_UNIX {
        let path_obj = if unsafe { pyre_object::is_tuple(addr) } {
            unsafe { pyre_object::w_tuple_getitem(addr, 0) }
                .ok_or_else(|| crate::PyError::value_error("address: missing path"))?
        } else {
            addr
        };
        let path_bytes_vec: Vec<u8> = unsafe {
            if pyre_object::is_str(path_obj) {
                pyre_object::w_str_get_value(path_obj)
                    .to_string()
                    .into_bytes()
            } else if pyre_object::bytesobject::is_bytes_like(path_obj) {
                pyre_object::bytesobject::bytes_like_data(path_obj).to_vec()
            } else {
                return Err(crate::PyError::type_error(
                    "AF_UNIX address must be a string or bytes path",
                ));
            }
        };
        let sun = unsafe { &mut *(&mut storage as *mut _ as *mut libc::sockaddr_un) };
        sun.sun_family = libc::AF_UNIX as libc::sa_family_t;
        if path_bytes_vec.len() >= sun.sun_path.len() {
            return Err(crate::PyError::os_error("AF_UNIX path too long"));
        }
        for (i, &b) in path_bytes_vec.iter().enumerate() {
            sun.sun_path[i] = b as libc::c_char;
        }
        return Ok((
            storage,
            (core::mem::size_of::<libc::sa_family_t>() + path_bytes_vec.len() + 1)
                as libc::socklen_t,
        ));
    }

    if !unsafe { pyre_object::is_tuple(addr) } {
        return Err(crate::PyError::type_error(
            "AF_INET address must be a (host, port) tuple",
        ));
    }
    let len = unsafe { pyre_object::w_tuple_len(addr) };
    if family == libc::AF_INET && len < 2 {
        return Err(crate::PyError::type_error(
            "AF_INET address must be a (host, port) tuple",
        ));
    }
    let host_obj = unsafe { pyre_object::w_tuple_getitem(addr, 0) }
        .ok_or_else(|| crate::PyError::value_error("address: missing host"))?;
    let port_obj = unsafe { pyre_object::w_tuple_getitem(addr, 1) }
        .ok_or_else(|| crate::PyError::value_error("address: missing port"))?;
    let host = unsafe {
        if !pyre_object::is_str(host_obj) {
            return Err(crate::PyError::type_error("address host must be a string"));
        }
        pyre_object::w_str_get_value(host_obj).to_string()
    };
    if !unsafe { pyre_object::is_int(port_obj) } {
        return Err(crate::PyError::type_error(
            "address port must be an integer",
        ));
    }
    let port_raw = unsafe { pyre_object::w_int_get_value(port_obj) };
    if !(0..=0xFFFF).contains(&port_raw) {
        return Err(crate::PyError::overflow_error("port must be 0-65535"));
    }
    let port = (port_raw as u16).to_be();

    let c_host = std::ffi::CString::new(host.as_bytes())
        .map_err(|_| crate::PyError::value_error("embedded null in host"))?;
    if family == libc::AF_INET {
        let sin = unsafe { &mut *(&mut storage as *mut _ as *mut libc::sockaddr_in) };
        sin.sin_family = libc::AF_INET as libc::sa_family_t;
        sin.sin_port = port;
        // inet_pton handles both "0.0.0.0" and dotted-quad.
        let r = unsafe {
            inet_pton(
                libc::AF_INET,
                c_host.as_ptr(),
                &mut sin.sin_addr as *mut _ as *mut libc::c_void,
            )
        };
        if r != 1 {
            // Fall back to gethostbyname for hostnames.
            let he = unsafe { gethostbyname(c_host.as_ptr()) };
            if he.is_null() {
                return Err(crate::PyError::os_error(format!(
                    "name or service not known: {host}"
                )));
            }
            unsafe {
                let h = &*he;
                let addr_ptr = *h.h_addr_list;
                sin.sin_addr.s_addr = *(addr_ptr as *const u32);
            }
        }
        Ok((
            storage,
            core::mem::size_of::<libc::sockaddr_in>() as libc::socklen_t,
        ))
    } else if family == libc::AF_INET6 {
        let sin6 = unsafe { &mut *(&mut storage as *mut _ as *mut libc::sockaddr_in6) };
        sin6.sin6_family = libc::AF_INET6 as libc::sa_family_t;
        sin6.sin6_port = port;
        let mut buf = [0u8; 16];
        let r = unsafe {
            inet_pton(
                libc::AF_INET6,
                c_host.as_ptr(),
                buf.as_mut_ptr() as *mut libc::c_void,
            )
        };
        if r != 1 {
            return Err(crate::PyError::os_error(format!(
                "invalid IPv6 address: {host}"
            )));
        }
        sin6.sin6_addr.s6_addr = buf;
        if len >= 3 {
            if let Some(v) = unsafe { pyre_object::w_tuple_getitem(addr, 2) } {
                sin6.sin6_flowinfo = unsafe { pyre_object::w_int_get_value(v) } as u32;
            }
        }
        if len >= 4 {
            if let Some(v) = unsafe { pyre_object::w_tuple_getitem(addr, 3) } {
                sin6.sin6_scope_id = unsafe { pyre_object::w_int_get_value(v) } as u32;
            }
        }
        Ok((
            storage,
            core::mem::size_of::<libc::sockaddr_in6>() as libc::socklen_t,
        ))
    } else {
        Err(crate::PyError::os_error(format!(
            "unsupported address family: {family}"
        )))
    }
}

#[cfg(unix)]
fn unpack_inet_addr(storage: &libc::sockaddr_storage) -> pyre_object::PyObjectRef {
    let family = storage.ss_family as libc::c_int;
    if family == libc::AF_INET {
        let sin = unsafe { &*(storage as *const _ as *const libc::sockaddr_in) };
        let mut buf = [0u8; 64];
        let p = unsafe {
            inet_ntop(
                libc::AF_INET,
                &sin.sin_addr as *const _ as *const libc::c_void,
                buf.as_mut_ptr() as *mut libc::c_char,
                buf.len() as libc::socklen_t,
            )
        };
        let host = if p.is_null() {
            String::new()
        } else {
            unsafe { std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned() }
        };
        let port = u16::from_be(sin.sin_port) as i64;
        pyre_object::w_tuple_new(vec![
            pyre_object::w_str_new(&host),
            pyre_object::w_int_new(port),
        ])
    } else if family == libc::AF_INET6 {
        let sin6 = unsafe { &*(storage as *const _ as *const libc::sockaddr_in6) };
        let mut buf = [0u8; 64];
        let p = unsafe {
            inet_ntop(
                libc::AF_INET6,
                &sin6.sin6_addr as *const _ as *const libc::c_void,
                buf.as_mut_ptr() as *mut libc::c_char,
                buf.len() as libc::socklen_t,
            )
        };
        let host = if p.is_null() {
            String::new()
        } else {
            unsafe { std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned() }
        };
        let port = u16::from_be(sin6.sin6_port) as i64;
        pyre_object::w_tuple_new(vec![
            pyre_object::w_str_new(&host),
            pyre_object::w_int_new(port),
            pyre_object::w_int_new(sin6.sin6_flowinfo as i64),
            pyre_object::w_int_new(sin6.sin6_scope_id as i64),
        ])
    } else if family == libc::AF_UNIX {
        let sun = unsafe { &*(storage as *const _ as *const libc::sockaddr_un) };
        let end = sun
            .sun_path
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(sun.sun_path.len());
        let bytes: Vec<u8> = sun.sun_path[..end].iter().map(|&b| b as u8).collect();
        pyre_object::w_str_new(&String::from_utf8_lossy(&bytes))
    } else {
        pyre_object::w_tuple_new(vec![])
    }
}

#[cfg(unix)]
fn init_socket_type(ns: &mut DictStorage) {
    // The `socket` callable: socket(family=AF_INET, type=SOCK_STREAM, proto=0, fileno=None)
    // CPython lets you pass a pre-existing fd via fileno=; we honor that
    // by wrapping the fd directly instead of calling socket(2).
    crate::dict_storage_store(
        ns,
        "__new__",
        crate::make_builtin_function("__new__", |args| {
            // args = (cls, family, type, proto, fileno).  The cls slot is
            // present when the type is invoked as `socket(...)`; keyword
            // arguments arrive as a trailing `__pyre_kw__` dict.
            let after_cls = if !args.is_empty() && !unsafe { pyre_object::is_int(args[0]) } {
                &args[1..]
            } else {
                args
            };
            let (pos, kwargs) = crate::builtins::split_builtin_kwargs(after_cls);
            // `descr_init`'s `@unwrap_spec` signature rejects unknown
            // keywords and a parameter supplied both by position and name.
            crate::builtins::kwarg_reject_unknown(
                kwargs,
                &["family", "type", "proto", "fileno"],
                "socket",
            )?;
            crate::builtins::kwarg_reject_duplicate(kwargs, "socket", "family", !pos.is_empty())?;
            crate::builtins::kwarg_reject_duplicate(kwargs, "socket", "type", pos.len() >= 2)?;
            crate::builtins::kwarg_reject_duplicate(kwargs, "socket", "proto", pos.len() >= 3)?;
            crate::builtins::kwarg_reject_duplicate(kwargs, "socket", "fileno", pos.len() >= 4)?;
            // `interp_socket.py:216 descr_init(family=-1, type=-1, proto=-1,
            // w_fileno=None)` — each parameter comes from its positional
            // slot, then its keyword; family/type/proto keep the sentinel
            // -1 (resolved below from the module defaults or the fd).
            let family_obj = pos
                .first()
                .copied()
                .or_else(|| crate::builtins::kwarg_get(kwargs, "family"));
            let type_obj = pos
                .get(1)
                .copied()
                .or_else(|| crate::builtins::kwarg_get(kwargs, "type"));
            let proto_obj = pos
                .get(2)
                .copied()
                .or_else(|| crate::builtins::kwarg_get(kwargs, "proto"));
            let fileno_obj = pos
                .get(3)
                .copied()
                .or_else(|| crate::builtins::kwarg_get(kwargs, "fileno"));
            // `@unwrap_spec(family=int, type=int, proto=int)` — a present
            // argument goes through the gateway int converter (`__index__` /
            // `__int__`, OverflowError if it does not fit), defaulting to the
            // -1 sentinel when omitted.
            let int_arg =
                |obj: Option<pyre_object::PyObjectRef>| -> Result<libc::c_int, crate::PyError> {
                match obj {
                    Some(o) => Ok(crate::baseobjspace::int_w(o)? as libc::c_int),
                    None => Ok(-1),
                }
            };
            let mut family = int_arg(family_obj)?;
            let mut ty = int_arg(type_obj)?;
            let mut proto = int_arg(proto_obj)?;
            let has_fileno = match fileno_obj {
                Some(o) => !unsafe { pyre_object::is_none(o) },
                None => false,
            };
            if !has_fileno {
                // `interp_socket.py:219-225` — without a fileno the
                // sentinels resolve to AF_INET / SOCK_STREAM / 0.
                if family == -1 {
                    family = libc::AF_INET;
                }
                if ty == -1 {
                    ty = libc::SOCK_STREAM;
                }
                if proto == -1 {
                    proto = 0;
                }
                let fd = unsafe { libc::socket(family, ty, proto) };
                if fd < 0 {
                    return Err(socket_io_err(std::io::Error::last_os_error()));
                }
                // `rsocket.py:RSocket.__init__` sets FD_CLOEXEC on every
                // newly created socket (PEP 446).
                unsafe {
                    libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC);
                }
                return Ok(socket_from_fd(fd, family, ty, proto));
            }
            // `interp_socket.py:253-265` — wrap an existing fd.  A float
            // fileno is a TypeError, a negative fd a ValueError, and any
            // -1 family/type/proto is derived from the descriptor itself.
            let fileno_obj = fileno_obj.unwrap();
            if unsafe { pyre_object::is_float(fileno_obj) } {
                return Err(crate::PyError::type_error(
                    "integer argument expected, got float",
                ));
            }
            // `interp_socket.py:255` — `space.int_w(w_fileno)` accepts ints,
            // longs, and objects with `__int__` / `__index__`.
            let fd = crate::baseobjspace::int_w(fileno_obj)?;
            if fd < 0 {
                return Err(crate::PyError::value_error("negative file descriptor"));
            }
            let fd = fd as libc::c_int;
            if family == -1 {
                family = socket_detect_family(fd)?;
            }
            if ty == -1 {
                ty = socket_getsockopt_int(fd, libc::SOL_SOCKET, libc::SO_TYPE)?;
            }
            if proto == -1 {
                proto = socket_get_so_protocol(fd)?;
            }
            Ok(socket_from_fd(fd, family, ty, proto))
        }),
    );

    // `interp_socket.py:1157-1160` — `family`/`type`/`proto`/`timeout`
    // are GetSetProperty data descriptors (plain attribute access, not
    // callables).  The getter receives `(descriptor, instance)`, so the
    // socket object is `args[1]`.
    crate::dict_storage_store(
        ns,
        "family",
        crate::typedef::make_getset_descriptor_named(
            crate::make_builtin_function_with_arity(
                "family",
                |args| Ok(pyre_object::w_int_new(socket_get_attr_i64(args[1], "_family"))),
                2,
            ),
            "family",
        ),
    );
    crate::dict_storage_store(
        ns,
        "type",
        crate::typedef::make_getset_descriptor_named(
            crate::make_builtin_function_with_arity(
                "type",
                |args| Ok(pyre_object::w_int_new(socket_get_attr_i64(args[1], "_type"))),
                2,
            ),
            "type",
        ),
    );
    crate::dict_storage_store(
        ns,
        "proto",
        crate::typedef::make_getset_descriptor_named(
            crate::make_builtin_function_with_arity(
                "proto",
                |args| Ok(pyre_object::w_int_new(socket_get_attr_i64(args[1], "_proto"))),
                2,
            ),
            "proto",
        ),
    );
    // `interp_socket.py:454 gettimeout_w` — `timeout` is the stored
    // `_timeout` object (float, or `None` when disabled).
    crate::dict_storage_store(
        ns,
        "timeout",
        crate::typedef::make_getset_descriptor_named(
            crate::make_builtin_function_with_arity(
                "timeout",
                |args| {
                    let d = crate::baseobjspace::getdict(args[1]);
                    if d.is_null() {
                        return Ok(pyre_object::w_none());
                    }
                    Ok(unsafe { pyre_object::w_dict_getitem_str(d, "_timeout") }
                        .unwrap_or(pyre_object::w_none()))
                },
                2,
            ),
            "timeout",
        ),
    );

    crate::dict_storage_store(
        ns,
        "fileno",
        crate::make_builtin_function_with_arity(
            "fileno",
            |args| {
                let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                Ok(pyre_object::w_int_new(socket_get_attr_i64(obj, "_fd")))
            },
            1,
        ),
    );

    crate::dict_storage_store(
        ns,
        "close",
        crate::make_builtin_function_with_arity(
            "close",
            |args| {
                let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                let fd = socket_get_attr_i64(obj, "_fd") as libc::c_int;
                if fd >= 0 {
                    let _ = unsafe { libc::close(fd) };
                    socket_set_attr(obj, "_fd", pyre_object::w_int_new(-1));
                }
                Ok(pyre_object::w_none())
            },
            1,
        ),
    );

    // detach() → returns the fd and forgets it.
    crate::dict_storage_store(
        ns,
        "detach",
        crate::make_builtin_function_with_arity(
            "detach",
            |args| {
                let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                let fd = socket_get_attr_i64(obj, "_fd");
                socket_set_attr(obj, "_fd", pyre_object::w_int_new(-1));
                Ok(pyre_object::w_int_new(fd))
            },
            1,
        ),
    );

    // `interp_socket.py:978-996 _reuse_w / _drop_w` — refcount methods
    // the app-level `socket._socketobject` wrapper uses to share one
    // underlying fd across `socket.makefile()` file-like aliases.
    // `_reuse` increments the usecount; `_drop` decrements and closes
    // when it reaches zero.
    crate::dict_storage_store(
        ns,
        "_reuse",
        crate::make_builtin_function_with_arity(
            "_reuse",
            |args| {
                let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                let n = socket_get_attr_i64(obj, "_usecount");
                let n = if n < 0 { 1 } else { n };
                socket_set_attr(obj, "_usecount", pyre_object::w_int_new(n + 1));
                Ok(pyre_object::w_none())
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "_drop",
        crate::make_builtin_function_with_arity(
            "_drop",
            |args| {
                let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                let n = socket_get_attr_i64(obj, "_usecount");
                let n = if n < 0 { 1 } else { n };
                let next = n - 1;
                socket_set_attr(obj, "_usecount", pyre_object::w_int_new(next));
                if next <= 0 {
                    let fd = socket_get_attr_i64(obj, "_fd") as libc::c_int;
                    if fd >= 0 {
                        let _ = unsafe { libc::close(fd) };
                        socket_set_attr(obj, "_fd", pyre_object::w_int_new(-1));
                    }
                }
                Ok(pyre_object::w_none())
            },
            1,
        ),
    );

    // bind(addr) — addr is (host, port) for AF_INET / (host, port, flowinfo,
    // scopeid) for AF_INET6 / path string for AF_UNIX.
    crate::dict_storage_store(
        ns,
        "bind",
        crate::make_builtin_function_with_arity(
            "bind",
            |args| {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error("bind() missing address"));
                }
                let obj = args[0];
                let fd = socket_fd(obj)?;
                let family = socket_get_attr_i64(obj, "_family") as libc::c_int;
                let (storage, slen) = pack_inet_addr(family, args[1])?;
                let r =
                    unsafe { libc::bind(fd, &storage as *const _ as *const libc::sockaddr, slen) };
                if r != 0 {
                    return Err(socket_io_err(std::io::Error::last_os_error()));
                }
                Ok(pyre_object::w_none())
            },
            2,
        ),
    );

    crate::dict_storage_store(
        ns,
        "listen",
        crate::make_builtin_function("listen", |args| {
            let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
            let fd = socket_fd(obj)?;
            let backlog = if args.len() >= 2 {
                (unsafe { pyre_object::w_int_get_value(args[1]) }) as libc::c_int
            } else {
                128
            };
            let r = unsafe { libc::listen(fd, backlog) };
            if r != 0 {
                return Err(socket_io_err(std::io::Error::last_os_error()));
            }
            Ok(pyre_object::w_none())
        }),
    );

    crate::dict_storage_store(
        ns,
        "accept",
        crate::make_builtin_function_with_arity(
            "accept",
            |args| {
                let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                let fd = socket_fd(obj)?;
                let family = socket_get_attr_i64(obj, "_family") as libc::c_int;
                let ty = socket_get_attr_i64(obj, "_type") as libc::c_int;
                let proto = socket_get_attr_i64(obj, "_proto") as libc::c_int;
                let mut storage: libc::sockaddr_storage = unsafe { std::mem::zeroed() };
                let mut slen = core::mem::size_of::<libc::sockaddr_storage>() as libc::socklen_t;
                let cfd = loop {
                    let r = unsafe {
                        libc::accept(fd, &mut storage as *mut _ as *mut libc::sockaddr, &mut slen)
                    };
                    if r >= 0 {
                        break r;
                    }
                    let err = std::io::Error::last_os_error();
                    if err.raw_os_error() != Some(libc::EINTR) {
                        return Err(socket_io_err(err));
                    }
                    // EINTR: deliver a pending signal, then retry
                    // (`converted_error` eintr_retry).
                    crate::module::signal::interp_signal::checksignals_now()?;
                };
                // `rsocket.py:RSocket._accept` returns the new fd with
                // FD_CLOEXEC set (rsocket uses accept4(SOCK_CLOEXEC) on
                // Linux; we use the portable fcntl path).
                unsafe {
                    libc::fcntl(cfd, libc::F_SETFD, libc::FD_CLOEXEC);
                }
                let new_sock = socket_from_fd(cfd, family, ty, proto);
                let addr = unpack_inet_addr(&storage);
                Ok(pyre_object::w_tuple_new(vec![new_sock, addr]))
            },
            1,
        ),
    );

    // `interp_socket.py:1090 socketmethodnames _accept` — primitive
    // returning `(fd, addr)`.  CPython's app-level `socket.py:262 def
    // accept` wraps this to construct the new socket object;
    // pyre's `accept` above bundles both steps for callers that
    // bypass the stdlib wrapper.
    crate::dict_storage_store(
        ns,
        "_accept",
        crate::make_builtin_function_with_arity(
            "_accept",
            |args| {
                let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                let fd = socket_fd(obj)?;
                let mut storage: libc::sockaddr_storage = unsafe { std::mem::zeroed() };
                let mut slen = core::mem::size_of::<libc::sockaddr_storage>() as libc::socklen_t;
                let cfd = loop {
                    let r = unsafe {
                        libc::accept(fd, &mut storage as *mut _ as *mut libc::sockaddr, &mut slen)
                    };
                    if r >= 0 {
                        break r;
                    }
                    let err = std::io::Error::last_os_error();
                    if err.raw_os_error() != Some(libc::EINTR) {
                        return Err(socket_io_err(err));
                    }
                    // EINTR: deliver a pending signal, then retry
                    // (`converted_error` eintr_retry).
                    crate::module::signal::interp_signal::checksignals_now()?;
                };
                unsafe {
                    libc::fcntl(cfd, libc::F_SETFD, libc::FD_CLOEXEC);
                }
                let addr = unpack_inet_addr(&storage);
                Ok(pyre_object::w_tuple_new(vec![
                    pyre_object::w_int_new(cfd as i64),
                    addr,
                ]))
            },
            1,
        ),
    );

    crate::dict_storage_store(
        ns,
        "connect",
        crate::make_builtin_function_with_arity(
            "connect",
            |args| {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error("connect() missing address"));
                }
                let obj = args[0];
                let fd = socket_fd(obj)?;
                let family = socket_get_attr_i64(obj, "_family") as libc::c_int;
                let (storage, slen) = pack_inet_addr(family, args[1])?;
                loop {
                    let r = unsafe {
                        libc::connect(fd, &storage as *const _ as *const libc::sockaddr, slen)
                    };
                    if r == 0 {
                        break;
                    }
                    let err = std::io::Error::last_os_error();
                    if err.raw_os_error() != Some(libc::EINTR) {
                        return Err(socket_io_err(err));
                    }
                    // EINTR: deliver a pending signal, then retry
                    // (`converted_error` eintr_retry).
                    crate::module::signal::interp_signal::checksignals_now()?;
                }
                Ok(pyre_object::w_none())
            },
            2,
        ),
    );

    // connect_ex(address) → errno (no exception on error)
    // `interp_socket.py:376-392` — `try: connect; except` equivalent
    // that returns the errno integer instead of raising OSError.
    crate::dict_storage_store(
        ns,
        "connect_ex",
        crate::make_builtin_function_with_arity(
            "connect_ex",
            |args| {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error("connect_ex() missing address"));
                }
                let obj = args[0];
                let fd = socket_fd(obj)?;
                let family = socket_get_attr_i64(obj, "_family") as libc::c_int;
                let (storage, slen) = pack_inet_addr(family, args[1])?;
                // `interp_socket.py:387-391` — retry while the call is
                // interrupted (EINTR), otherwise return the errno.
                let err = loop {
                    let r = unsafe {
                        libc::connect(fd, &storage as *const _ as *const libc::sockaddr, slen)
                    };
                    if r == 0 {
                        break 0;
                    }
                    let e = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
                    if e != libc::EINTR {
                        break e;
                    }
                    // `interp_socket.py:391` — deliver a pending signal, then
                    // retry the connect.
                    crate::module::signal::interp_signal::checksignals_now()?;
                };
                Ok(pyre_object::w_int_new(err as i64))
            },
            2,
        ),
    );

    crate::dict_storage_store(
        ns,
        "send",
        crate::make_builtin_function("send", |args| {
            if args.len() < 2 {
                return Err(crate::PyError::type_error("send() missing buffer"));
            }
            let obj = args[0];
            let fd = socket_fd(obj)?;
            let buf = unsafe {
                if !pyre_object::bytesobject::is_bytes_like(args[1]) {
                    return Err(crate::PyError::type_error(
                        "send: buffer must be bytes-like",
                    ));
                }
                pyre_object::bytesobject::bytes_like_data(args[1])
            };
            let flags = if args.len() >= 3 {
                (unsafe { pyre_object::w_int_get_value(args[2]) }) as libc::c_int
            } else {
                0
            };
            let n = loop {
                let r = unsafe {
                    libc::send(fd, buf.as_ptr() as *const libc::c_void, buf.len(), flags)
                };
                if r >= 0 {
                    break r;
                }
                let err = std::io::Error::last_os_error();
                if err.raw_os_error() != Some(libc::EINTR) {
                    return Err(socket_io_err(err));
                }
                // EINTR: deliver a pending signal, then retry
                // (`converted_error` eintr_retry).
                crate::module::signal::interp_signal::checksignals_now()?;
            };
            Ok(pyre_object::w_int_new(n as i64))
        }),
    );

    crate::dict_storage_store(
        ns,
        "sendall",
        crate::make_builtin_function("sendall", |args| {
            if args.len() < 2 {
                return Err(crate::PyError::type_error("sendall() missing buffer"));
            }
            let obj = args[0];
            let fd = socket_fd(obj)?;
            let buf = unsafe {
                if !pyre_object::bytesobject::is_bytes_like(args[1]) {
                    return Err(crate::PyError::type_error(
                        "sendall: buffer must be bytes-like",
                    ));
                }
                pyre_object::bytesobject::bytes_like_data(args[1]).to_vec()
            };
            let flags = if args.len() >= 3 {
                (unsafe { pyre_object::w_int_get_value(args[2]) }) as libc::c_int
            } else {
                0
            };
            let mut off = 0usize;
            while off < buf.len() {
                let n = unsafe {
                    libc::send(
                        fd,
                        buf[off..].as_ptr() as *const libc::c_void,
                        buf.len() - off,
                        flags,
                    )
                };
                if n < 0 {
                    let err = std::io::Error::last_os_error();
                    if err.raw_os_error() == Some(libc::EINTR) {
                        // `rsocket.py:1132` signal_checker — deliver a pending
                        // signal, then retry the remaining bytes.
                        crate::module::signal::interp_signal::checksignals_now()?;
                        continue;
                    }
                    return Err(socket_io_err(err));
                }
                off += n as usize;
            }
            Ok(pyre_object::w_none())
        }),
    );

    crate::dict_storage_store(
        ns,
        "recv",
        crate::make_builtin_function("recv", |args| {
            if args.len() < 2 {
                return Err(crate::PyError::type_error("recv() missing size"));
            }
            if !unsafe { pyre_object::is_int(args[1]) } {
                return Err(crate::PyError::type_error("recv: size must be an integer"));
            }
            let raw = unsafe { pyre_object::w_int_get_value(args[1]) };
            if raw < 0 {
                return Err(crate::PyError::value_error("negative buffersize in recv"));
            }
            let obj = args[0];
            let fd = socket_fd(obj)?;
            let n = raw as usize;
            let flags = if args.len() >= 3 {
                if !unsafe { pyre_object::is_int(args[2]) } {
                    return Err(crate::PyError::type_error("recv: flags must be an integer"));
                }
                (unsafe { pyre_object::w_int_get_value(args[2]) }) as libc::c_int
            } else {
                0
            };
            let mut buf = vec![0u8; n];
            let got = loop {
                let r = unsafe { libc::recv(fd, buf.as_mut_ptr() as *mut libc::c_void, n, flags) };
                if r >= 0 {
                    break r;
                }
                let err = std::io::Error::last_os_error();
                if err.raw_os_error() != Some(libc::EINTR) {
                    return Err(socket_io_err(err));
                }
                // EINTR: deliver a pending signal, then retry
                // (`converted_error` eintr_retry).
                crate::module::signal::interp_signal::checksignals_now()?;
            };
            buf.truncate(got as usize);
            Ok(pyre_object::bytesobject::w_bytes_from_bytes(&buf))
        }),
    );

    crate::dict_storage_store(
        ns,
        "sendto",
        crate::make_builtin_function("sendto", |args| {
            // sendto(buffer, [flags,] address)
            if args.len() < 3 {
                return Err(crate::PyError::type_error(
                    "sendto() needs buffer + address",
                ));
            }
            let obj = args[0];
            let fd = socket_fd(obj)?;
            let buf = unsafe {
                if !pyre_object::bytesobject::is_bytes_like(args[1]) {
                    return Err(crate::PyError::type_error(
                        "sendto: buffer must be bytes-like",
                    ));
                }
                pyre_object::bytesobject::bytes_like_data(args[1])
            };
            // 3-arg form: (buf, flags, addr).  4-arg form: (self, buf, flags, addr).
            // We always take self-as-args[0], so 3 args = (self, buf, addr) [no flags]
            // and 4 args = (self, buf, flags, addr).
            let (flags, addr_obj) = if args.len() == 3 {
                (0, args[2])
            } else {
                (
                    (unsafe { pyre_object::w_int_get_value(args[2]) }) as libc::c_int,
                    args[3],
                )
            };
            let family = socket_get_attr_i64(obj, "_family") as libc::c_int;
            let (storage, slen) = pack_inet_addr(family, addr_obj)?;
            let n = loop {
                let r = unsafe {
                    libc::sendto(
                        fd,
                        buf.as_ptr() as *const libc::c_void,
                        buf.len(),
                        flags,
                        &storage as *const _ as *const libc::sockaddr,
                        slen,
                    )
                };
                if r >= 0 {
                    break r;
                }
                let err = std::io::Error::last_os_error();
                if err.raw_os_error() != Some(libc::EINTR) {
                    return Err(socket_io_err(err));
                }
                // EINTR: deliver a pending signal, then retry
                // (`converted_error` eintr_retry).
                crate::module::signal::interp_signal::checksignals_now()?;
            };
            Ok(pyre_object::w_int_new(n as i64))
        }),
    );

    crate::dict_storage_store(
        ns,
        "recvfrom",
        crate::make_builtin_function("recvfrom", |args| {
            if args.len() < 2 {
                return Err(crate::PyError::type_error("recvfrom() missing size"));
            }
            if !unsafe { pyre_object::is_int(args[1]) } {
                return Err(crate::PyError::type_error(
                    "recvfrom: size must be an integer",
                ));
            }
            let raw = unsafe { pyre_object::w_int_get_value(args[1]) };
            if raw < 0 {
                return Err(crate::PyError::value_error(
                    "negative buffersize in recvfrom",
                ));
            }
            let obj = args[0];
            let fd = socket_fd(obj)?;
            let n = raw as usize;
            let flags = if args.len() >= 3 {
                if !unsafe { pyre_object::is_int(args[2]) } {
                    return Err(crate::PyError::type_error(
                        "recvfrom: flags must be an integer",
                    ));
                }
                (unsafe { pyre_object::w_int_get_value(args[2]) }) as libc::c_int
            } else {
                0
            };
            let mut buf = vec![0u8; n];
            let mut storage: libc::sockaddr_storage = unsafe { std::mem::zeroed() };
            let mut slen = core::mem::size_of::<libc::sockaddr_storage>() as libc::socklen_t;
            let got = loop {
                let r = unsafe {
                    libc::recvfrom(
                        fd,
                        buf.as_mut_ptr() as *mut libc::c_void,
                        n,
                        flags,
                        &mut storage as *mut _ as *mut libc::sockaddr,
                        &mut slen,
                    )
                };
                if r >= 0 {
                    break r;
                }
                let err = std::io::Error::last_os_error();
                if err.raw_os_error() != Some(libc::EINTR) {
                    return Err(socket_io_err(err));
                }
                // EINTR: deliver a pending signal, then retry
                // (`converted_error` eintr_retry).
                crate::module::signal::interp_signal::checksignals_now()?;
            };
            buf.truncate(got as usize);
            let addr = unpack_inet_addr(&storage);
            Ok(pyre_object::w_tuple_new(vec![
                pyre_object::bytesobject::w_bytes_from_bytes(&buf),
                addr,
            ]))
        }),
    );

    // recv_into(buffer, [nbytes, flags]) → nbytes_read
    // `interp_socket.py:831-863` — writes directly into a writable
    // bytes-like buffer.  nbytes==0 uses the full buffer length.
    crate::dict_storage_store(
        ns,
        "recv_into",
        crate::make_builtin_function("recv_into", |args| {
            if args.len() < 2 {
                return Err(crate::PyError::type_error("recv_into() missing buffer"));
            }
            let obj = args[0];
            let buf_obj = args[1];
            let slot = socket_writebuf(buf_obj)?;
            let buf_len = slot.len();
            let nbytes = if args.len() >= 3 {
                if !unsafe { pyre_object::is_int(args[2]) } {
                    return Err(crate::PyError::type_error(
                        "recv_into: nbytes must be an integer",
                    ));
                }
                let raw = unsafe { pyre_object::w_int_get_value(args[2]) };
                if raw < 0 {
                    return Err(crate::PyError::value_error(
                        "negative buffersize in recv_into",
                    ));
                }
                let n = raw as usize;
                if n == 0 { buf_len } else { n }
            } else {
                buf_len
            };
            if buf_len < nbytes {
                return Err(crate::PyError::value_error(
                    "buffer too small for requested bytes",
                ));
            }
            let flags = if args.len() >= 4 {
                if !unsafe { pyre_object::is_int(args[3]) } {
                    return Err(crate::PyError::type_error(
                        "recv_into: flags must be an integer",
                    ));
                }
                unsafe { pyre_object::w_int_get_value(args[3]) as libc::c_int }
            } else {
                0
            };
            let fd = socket_fd(obj)?;
            let got = loop {
                let r = unsafe {
                    libc::recv(fd, slot.as_mut_ptr() as *mut libc::c_void, nbytes, flags)
                };
                if r >= 0 {
                    break r;
                }
                let err = std::io::Error::last_os_error();
                if err.raw_os_error() != Some(libc::EINTR) {
                    return Err(socket_io_err(err));
                }
                // EINTR: deliver a pending signal, then retry
                // (`converted_error` eintr_retry).
                crate::module::signal::interp_signal::checksignals_now()?;
            };
            Ok(pyre_object::w_int_new(got as i64))
        }),
    );

    // recvfrom_into(buffer, [nbytes, flags]) → (nbytes, address)
    // `interp_socket.py:866-899` — recvfrom variant that fills a
    // caller-provided buffer rather than allocating a new bytes.
    crate::dict_storage_store(
        ns,
        "recvfrom_into",
        crate::make_builtin_function("recvfrom_into", |args| {
            if args.len() < 2 {
                return Err(crate::PyError::type_error("recvfrom_into() missing buffer"));
            }
            let obj = args[0];
            let buf_obj = args[1];
            let slot = socket_writebuf(buf_obj)?;
            let buf_len = slot.len();
            let nbytes = if args.len() >= 3 {
                if !unsafe { pyre_object::is_int(args[2]) } {
                    return Err(crate::PyError::type_error(
                        "recvfrom_into: nbytes must be an integer",
                    ));
                }
                let raw = unsafe { pyre_object::w_int_get_value(args[2]) };
                if raw < 0 {
                    return Err(crate::PyError::value_error(
                        "negative buffersize in recvfrom_into",
                    ));
                }
                let n = raw as usize;
                if n == 0 { buf_len } else { n }
            } else {
                buf_len
            };
            if nbytes > buf_len {
                return Err(crate::PyError::value_error(
                    "nbytes is greater than the length of the buffer",
                ));
            }
            let flags = if args.len() >= 4 {
                if !unsafe { pyre_object::is_int(args[3]) } {
                    return Err(crate::PyError::type_error(
                        "recvfrom_into: flags must be an integer",
                    ));
                }
                unsafe { pyre_object::w_int_get_value(args[3]) as libc::c_int }
            } else {
                0
            };
            let fd = socket_fd(obj)?;
            let mut storage: libc::sockaddr_storage = unsafe { std::mem::zeroed() };
            let mut slen = core::mem::size_of::<libc::sockaddr_storage>() as libc::socklen_t;
            let got = loop {
                let r = unsafe {
                    libc::recvfrom(
                        fd,
                        slot.as_mut_ptr() as *mut libc::c_void,
                        nbytes,
                        flags,
                        &mut storage as *mut _ as *mut libc::sockaddr,
                        &mut slen,
                    )
                };
                if r >= 0 {
                    break r;
                }
                let err = std::io::Error::last_os_error();
                if err.raw_os_error() != Some(libc::EINTR) {
                    return Err(socket_io_err(err));
                }
                // EINTR: deliver a pending signal, then retry
                // (`converted_error` eintr_retry).
                crate::module::signal::interp_signal::checksignals_now()?;
            };
            let addr = unpack_inet_addr(&storage);
            Ok(pyre_object::w_tuple_new(vec![
                pyre_object::w_int_new(got as i64),
                addr,
            ]))
        }),
    );

    // recvmsg(bufsize, [ancbufsize, flags]) → (data, ancdata, msg_flags, address)
    // `interp_socket.py:525-569` — receives normal + ancillary data
    // via libc::recvmsg.  ancdata is a list of (cmsg_level, cmsg_type,
    // cmsg_data:bytes) triples walked through CMSG_FIRSTHDR /
    // CMSG_NXTHDR / CMSG_DATA.
    crate::dict_storage_store(
        ns,
        "recvmsg",
        crate::make_builtin_function("recvmsg", |args| {
            if args.len() < 2 {
                return Err(crate::PyError::type_error("recvmsg() missing buffer size"));
            }
            if !unsafe { pyre_object::is_int(args[1]) } {
                return Err(crate::PyError::type_error(
                    "recvmsg: bufsize must be an integer",
                ));
            }
            let bufsize_raw = unsafe { pyre_object::w_int_get_value(args[1]) };
            if bufsize_raw < 0 {
                return Err(crate::PyError::value_error(
                    "negative buffer size in recvmsg()",
                ));
            }
            let bufsize = bufsize_raw as usize;
            let ancbufsize = if args.len() >= 3 {
                if !unsafe { pyre_object::is_int(args[2]) } {
                    return Err(crate::PyError::type_error(
                        "recvmsg: ancbufsize must be an integer",
                    ));
                }
                let raw = unsafe { pyre_object::w_int_get_value(args[2]) };
                if raw < 0 {
                    return Err(crate::PyError::value_error(
                        "invalid ancillary data buffer length",
                    ));
                }
                raw as usize
            } else {
                0
            };
            let flags = if args.len() >= 4 {
                if !unsafe { pyre_object::is_int(args[3]) } {
                    return Err(crate::PyError::type_error(
                        "recvmsg: flags must be an integer",
                    ));
                }
                unsafe { pyre_object::w_int_get_value(args[3]) as libc::c_int }
            } else {
                0
            };
            let fd = socket_fd(args[0])?;

            let mut data = vec![0u8; bufsize];
            let mut control = vec![0u8; ancbufsize];
            let mut storage: libc::sockaddr_storage = unsafe { std::mem::zeroed() };
            let (got, msg_flags) = loop {
                let mut iov = libc::iovec {
                    iov_base: data.as_mut_ptr() as *mut libc::c_void,
                    iov_len: bufsize,
                };
                let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
                msg.msg_name = &mut storage as *mut _ as *mut libc::c_void;
                msg.msg_namelen = core::mem::size_of::<libc::sockaddr_storage>() as libc::socklen_t;
                msg.msg_iov = &mut iov;
                msg.msg_iovlen = 1;
                if ancbufsize > 0 {
                    msg.msg_control = control.as_mut_ptr() as *mut libc::c_void;
                    msg.msg_controllen = ancbufsize as _;
                }
                let r = unsafe { libc::recvmsg(fd, &mut msg, flags) };
                if r >= 0 {
                    break (r, msg.msg_flags);
                }
                let err = std::io::Error::last_os_error();
                if err.raw_os_error() != Some(libc::EINTR) {
                    return Err(socket_io_err(err));
                }
                // EINTR: deliver a pending signal, then retry
                // (`converted_error` eintr_retry).
                crate::module::signal::interp_signal::checksignals_now()?;
            };
            data.truncate(got as usize);

            // Walk ancillary data.  Re-run msghdr with the final
            // controllen so CMSG_* macros see the trimmed buffer.
            let mut anc_items = Vec::new();
            if ancbufsize > 0 {
                let mut iov = libc::iovec {
                    iov_base: data.as_mut_ptr() as *mut libc::c_void,
                    iov_len: bufsize,
                };
                let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
                msg.msg_iov = &mut iov;
                msg.msg_iovlen = 1;
                msg.msg_control = control.as_mut_ptr() as *mut libc::c_void;
                msg.msg_controllen = ancbufsize as _;
                unsafe {
                    let mut cmsg = libc::CMSG_FIRSTHDR(&msg);
                    while !cmsg.is_null() {
                        let header = &*cmsg;
                        let hdr_size = libc::CMSG_LEN(0) as usize;
                        let total = header.cmsg_len as usize;
                        if total < hdr_size {
                            break;
                        }
                        let payload_len = total - hdr_size;
                        let payload_ptr = libc::CMSG_DATA(cmsg);
                        let payload = std::slice::from_raw_parts(payload_ptr, payload_len).to_vec();
                        anc_items.push(pyre_object::w_tuple_new(vec![
                            pyre_object::w_int_new(header.cmsg_level as i64),
                            pyre_object::w_int_new(header.cmsg_type as i64),
                            pyre_object::bytesobject::w_bytes_from_bytes(&payload),
                        ]));
                        cmsg = libc::CMSG_NXTHDR(&msg, cmsg);
                    }
                }
            }
            let addr = unpack_inet_addr(&storage);
            Ok(pyre_object::w_tuple_new(vec![
                pyre_object::bytesobject::w_bytes_from_bytes(&data),
                pyre_object::w_list_new(anc_items),
                pyre_object::w_int_new(msg_flags as i64),
                addr,
            ]))
        }),
    );

    // recvmsg_into(buffers, [ancbufsize, [flags]]) ->
    //   (nbytes, ancdata, msg_flags, address)
    // `interp_socket.py:572-652 recvmsg_into_w` — scatter-receive into
    // a list/tuple of writable buffers; each `writebuf_w` slice
    // contributes one iovec entry.
    crate::dict_storage_store(
        ns,
        "recvmsg_into",
        crate::make_builtin_function("recvmsg_into", |args| {
            if args.len() < 2 {
                return Err(crate::PyError::type_error("recvmsg_into() missing buffers"));
            }
            let seq = args[1];
            let (is_list, is_tuple) =
                unsafe { (pyre_object::is_list(seq), pyre_object::is_tuple(seq)) };
            if !is_list && !is_tuple {
                return Err(crate::PyError::type_error(
                    "recvmsg_into: buffers must be a list or tuple of writable buffers",
                ));
            }
            let nbufs = unsafe {
                if is_list {
                    pyre_object::w_list_len(seq)
                } else {
                    pyre_object::w_tuple_len(seq)
                }
            };
            let mut buffers: Vec<&'static mut [u8]> = Vec::with_capacity(nbufs);
            for i in 0..nbufs {
                let item = unsafe {
                    if is_list {
                        pyre_object::w_list_getitem(seq, i as i64)
                    } else {
                        pyre_object::w_tuple_getitem(seq, i as i64)
                    }
                }
                .ok_or_else(|| crate::PyError::type_error("recvmsg_into: buffer item missing"))?;
                buffers.push(socket_writebuf(item)?);
            }
            let ancbufsize = if args.len() >= 3 {
                if !unsafe { pyre_object::is_int(args[2]) } {
                    return Err(crate::PyError::type_error(
                        "recvmsg_into: ancbufsize must be an integer",
                    ));
                }
                let raw = unsafe { pyre_object::w_int_get_value(args[2]) };
                if raw < 0 {
                    return Err(crate::PyError::value_error(
                        "invalid ancillary data buffer length",
                    ));
                }
                raw as usize
            } else {
                0
            };
            let flags = if args.len() >= 4 {
                if !unsafe { pyre_object::is_int(args[3]) } {
                    return Err(crate::PyError::type_error(
                        "recvmsg_into: flags must be an integer",
                    ));
                }
                unsafe { pyre_object::w_int_get_value(args[3]) as libc::c_int }
            } else {
                0
            };
            let fd = socket_fd(args[0])?;

            let mut iovs: Vec<libc::iovec> = buffers
                .into_iter()
                .map(|slice| libc::iovec {
                    iov_base: slice.as_mut_ptr() as *mut libc::c_void,
                    iov_len: slice.len(),
                })
                .collect();
            let mut control = vec![0u8; ancbufsize];
            let mut storage: libc::sockaddr_storage = unsafe { std::mem::zeroed() };
            let (got, msg_flags, controllen) = loop {
                let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
                msg.msg_name = &mut storage as *mut _ as *mut libc::c_void;
                msg.msg_namelen = core::mem::size_of::<libc::sockaddr_storage>() as libc::socklen_t;
                msg.msg_iov = iovs.as_mut_ptr();
                msg.msg_iovlen = iovs.len() as _;
                if ancbufsize > 0 {
                    msg.msg_control = control.as_mut_ptr() as *mut libc::c_void;
                    msg.msg_controllen = ancbufsize as _;
                }
                let r = unsafe { libc::recvmsg(fd, &mut msg, flags) };
                if r >= 0 {
                    break (r, msg.msg_flags, msg.msg_controllen);
                }
                let err = std::io::Error::last_os_error();
                if err.raw_os_error() != Some(libc::EINTR) {
                    return Err(socket_io_err(err));
                }
                // EINTR: deliver a pending signal, then retry
                // (`converted_error` eintr_retry).
                crate::module::signal::interp_signal::checksignals_now()?;
            };

            let mut anc_items = Vec::new();
            if ancbufsize > 0 && controllen > 0 {
                let mut dummy_iov = libc::iovec {
                    iov_base: std::ptr::null_mut(),
                    iov_len: 0,
                };
                let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
                msg.msg_iov = &mut dummy_iov;
                msg.msg_iovlen = 1;
                msg.msg_control = control.as_mut_ptr() as *mut libc::c_void;
                msg.msg_controllen = controllen;
                unsafe {
                    let mut cmsg = libc::CMSG_FIRSTHDR(&msg);
                    while !cmsg.is_null() {
                        let header = &*cmsg;
                        let hdr_size = libc::CMSG_LEN(0) as usize;
                        let total = header.cmsg_len as usize;
                        if total < hdr_size {
                            break;
                        }
                        let payload_len = total - hdr_size;
                        let payload_ptr = libc::CMSG_DATA(cmsg);
                        let payload = std::slice::from_raw_parts(payload_ptr, payload_len).to_vec();
                        anc_items.push(pyre_object::w_tuple_new(vec![
                            pyre_object::w_int_new(header.cmsg_level as i64),
                            pyre_object::w_int_new(header.cmsg_type as i64),
                            pyre_object::bytesobject::w_bytes_from_bytes(&payload),
                        ]));
                        cmsg = libc::CMSG_NXTHDR(&msg, cmsg);
                    }
                }
            }
            let addr = unpack_inet_addr(&storage);
            Ok(pyre_object::w_tuple_new(vec![
                pyre_object::w_int_new(got as i64),
                pyre_object::w_list_new(anc_items),
                pyre_object::w_int_new(msg_flags as i64),
                addr,
            ]))
        }),
    );

    // sendmsg(data_iter[, ancillary[, flags[, address]]]) → bytes_sent
    // `interp_socket.py:711-773` — gather-write of multiple bytes-like
    // buffers plus optional ancillary control messages.  Each cmsg is
    // a (cmsg_level, cmsg_type, cmsg_data) 3-tuple; we lay them out
    // into a single control buffer via CMSG_SPACE / CMSG_NXTHDR.
    crate::dict_storage_store(
        ns,
        "sendmsg",
        crate::make_builtin_function("sendmsg", |args| {
            if args.len() < 2 {
                return Err(crate::PyError::type_error("sendmsg() missing data"));
            }
            let obj = args[0];
            let fd = socket_fd(obj)?;

            // Collect data buffers from args[1] (must be an iterable
            // of bytes-like).  We borrow the bytes-like data ref into
            // a Vec<&[u8]> so the iovec can point at it.
            if !unsafe { pyre_object::is_list(args[1]) || pyre_object::is_tuple(args[1]) } {
                return Err(crate::PyError::type_error(
                    "sendmsg: data must be a sequence of bytes-like objects",
                ));
            }
            let data_len = unsafe {
                if pyre_object::is_list(args[1]) {
                    pyre_object::w_list_len(args[1])
                } else {
                    pyre_object::w_tuple_len(args[1])
                }
            };
            let mut data_refs: Vec<&[u8]> = Vec::with_capacity(data_len);
            for i in 0..data_len {
                let item = unsafe {
                    if pyre_object::is_list(args[1]) {
                        pyre_object::w_list_getitem(args[1], i as i64)
                            .unwrap_or(pyre_object::PY_NULL)
                    } else {
                        pyre_object::w_tuple_getitem(args[1], i as i64)
                            .unwrap_or(pyre_object::PY_NULL)
                    }
                };
                if !unsafe { pyre_object::bytesobject::is_bytes_like(item) } {
                    return Err(crate::PyError::type_error(
                        "sendmsg: data items must be bytes-like",
                    ));
                }
                let slice = unsafe { pyre_object::bytesobject::bytes_like_data(item) };
                data_refs.push(slice);
            }
            let mut iovs: Vec<libc::iovec> = data_refs
                .iter()
                .map(|s| libc::iovec {
                    iov_base: s.as_ptr() as *mut libc::c_void,
                    iov_len: s.len(),
                })
                .collect();

            // Build ancillary control buffer from args[2] (optional).
            let mut cmsgs: Vec<(libc::c_int, libc::c_int, Vec<u8>)> = Vec::new();
            if args.len() >= 3 && !unsafe { pyre_object::is_none(args[2]) } {
                if !unsafe { pyre_object::is_list(args[2]) || pyre_object::is_tuple(args[2]) } {
                    return Err(crate::PyError::type_error(
                        "sendmsg: ancillary must be a sequence",
                    ));
                }
                let n = unsafe {
                    if pyre_object::is_list(args[2]) {
                        pyre_object::w_list_len(args[2])
                    } else {
                        pyre_object::w_tuple_len(args[2])
                    }
                };
                for i in 0..n {
                    let item = unsafe {
                        if pyre_object::is_list(args[2]) {
                            pyre_object::w_list_getitem(args[2], i as i64)
                                .unwrap_or(pyre_object::PY_NULL)
                        } else {
                            pyre_object::w_tuple_getitem(args[2], i as i64)
                                .unwrap_or(pyre_object::PY_NULL)
                        }
                    };
                    if !unsafe { pyre_object::is_tuple(item) }
                        || unsafe { pyre_object::w_tuple_len(item) } != 3
                    {
                        return Err(crate::PyError::type_error(
                            "sendmsg: ancillary items must be 3-tuples",
                        ));
                    }
                    let level_o = unsafe { pyre_object::w_tuple_getitem(item, 0) }
                        .ok_or_else(|| crate::PyError::value_error("ancillary level missing"))?;
                    let type_o = unsafe { pyre_object::w_tuple_getitem(item, 1) }
                        .ok_or_else(|| crate::PyError::value_error("ancillary type missing"))?;
                    let data_o = unsafe { pyre_object::w_tuple_getitem(item, 2) }
                        .ok_or_else(|| crate::PyError::value_error("ancillary data missing"))?;
                    if !unsafe { pyre_object::is_int(level_o) }
                        || !unsafe { pyre_object::is_int(type_o) }
                    {
                        return Err(crate::PyError::type_error(
                            "sendmsg: ancillary level/type must be integers",
                        ));
                    }
                    if !unsafe { pyre_object::bytesobject::is_bytes_like(data_o) } {
                        return Err(crate::PyError::type_error(
                            "sendmsg: ancillary data must be bytes-like",
                        ));
                    }
                    let level = unsafe { pyre_object::w_int_get_value(level_o) } as libc::c_int;
                    let ty = unsafe { pyre_object::w_int_get_value(type_o) } as libc::c_int;
                    let data =
                        unsafe { pyre_object::bytesobject::bytes_like_data(data_o).to_vec() };
                    cmsgs.push((level, ty, data));
                }
            }
            let flags = if args.len() >= 4 {
                if !unsafe { pyre_object::is_int(args[3]) } {
                    return Err(crate::PyError::type_error(
                        "sendmsg: flags must be an integer",
                    ));
                }
                unsafe { pyre_object::w_int_get_value(args[3]) as libc::c_int }
            } else {
                0
            };
            let (addr_storage, addr_len) =
                if args.len() >= 5 && !unsafe { pyre_object::is_none(args[4]) } {
                    let family = socket_get_attr_i64(obj, "_family") as libc::c_int;
                    let (s, l) = pack_inet_addr(family, args[4])?;
                    (Some(s), l)
                } else {
                    (None, 0)
                };

            // Lay out cmsgs into a single control buffer.
            let total_control: usize = cmsgs
                .iter()
                .map(|(_, _, d)| unsafe { libc::CMSG_SPACE(d.len() as libc::c_uint) as usize })
                .sum();
            let mut control = vec![0u8; total_control];
            let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
            msg.msg_iov = iovs.as_mut_ptr();
            msg.msg_iovlen = iovs.len() as _;
            if let Some(ref s) = addr_storage {
                msg.msg_name = s as *const _ as *mut libc::c_void;
                msg.msg_namelen = addr_len;
            }
            if total_control > 0 {
                msg.msg_control = control.as_mut_ptr() as *mut libc::c_void;
                msg.msg_controllen = total_control as _;
                unsafe {
                    let mut cur = libc::CMSG_FIRSTHDR(&msg);
                    for (level, ty, data) in &cmsgs {
                        if cur.is_null() {
                            break;
                        }
                        let cmsg_len = libc::CMSG_LEN(data.len() as libc::c_uint);
                        (*cur).cmsg_level = *level;
                        (*cur).cmsg_type = *ty;
                        (*cur).cmsg_len = cmsg_len as _;
                        std::ptr::copy_nonoverlapping(
                            data.as_ptr(),
                            libc::CMSG_DATA(cur),
                            data.len(),
                        );
                        cur = libc::CMSG_NXTHDR(&msg, cur);
                    }
                }
            }

            let sent = loop {
                let r = unsafe { libc::sendmsg(fd, &msg, flags) };
                if r >= 0 {
                    break r;
                }
                let err = std::io::Error::last_os_error();
                if err.raw_os_error() != Some(libc::EINTR) {
                    return Err(socket_io_err(err));
                }
                // EINTR: deliver a pending signal, then retry
                // (`converted_error` eintr_retry).
                crate::module::signal::interp_signal::checksignals_now()?;
            };
            Ok(pyre_object::w_int_new(sent as i64))
        }),
    );

    crate::dict_storage_store(
        ns,
        "shutdown",
        crate::make_builtin_function_with_arity(
            "shutdown",
            |args| {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error("shutdown() missing how"));
                }
                let fd = socket_fd(args[0])?;
                let how = (unsafe { pyre_object::w_int_get_value(args[1]) }) as libc::c_int;
                let r = unsafe { libc::shutdown(fd, how) };
                if r != 0 {
                    return Err(socket_io_err(std::io::Error::last_os_error()));
                }
                Ok(pyre_object::w_none())
            },
            2,
        ),
    );

    crate::dict_storage_store(
        ns,
        "getsockname",
        crate::make_builtin_function_with_arity(
            "getsockname",
            |args| {
                let fd = socket_fd(args.first().copied().unwrap_or(pyre_object::PY_NULL))?;
                let mut storage: libc::sockaddr_storage = unsafe { std::mem::zeroed() };
                let mut slen = core::mem::size_of::<libc::sockaddr_storage>() as libc::socklen_t;
                let r = unsafe {
                    libc::getsockname(fd, &mut storage as *mut _ as *mut libc::sockaddr, &mut slen)
                };
                if r != 0 {
                    return Err(socket_io_err(std::io::Error::last_os_error()));
                }
                Ok(unpack_inet_addr(&storage))
            },
            1,
        ),
    );

    crate::dict_storage_store(
        ns,
        "getpeername",
        crate::make_builtin_function_with_arity(
            "getpeername",
            |args| {
                let fd = socket_fd(args.first().copied().unwrap_or(pyre_object::PY_NULL))?;
                let mut storage: libc::sockaddr_storage = unsafe { std::mem::zeroed() };
                let mut slen = core::mem::size_of::<libc::sockaddr_storage>() as libc::socklen_t;
                let r = unsafe {
                    libc::getpeername(fd, &mut storage as *mut _ as *mut libc::sockaddr, &mut slen)
                };
                if r != 0 {
                    return Err(socket_io_err(std::io::Error::last_os_error()));
                }
                Ok(unpack_inet_addr(&storage))
            },
            1,
        ),
    );

    crate::dict_storage_store(
        ns,
        "setsockopt",
        crate::make_builtin_function("setsockopt", |args| {
            if args.len() < 4 {
                return Err(crate::PyError::type_error(
                    "setsockopt() requires self + level + name + value",
                ));
            }
            let fd = socket_fd(args[0])?;
            let level = (unsafe { pyre_object::w_int_get_value(args[1]) }) as libc::c_int;
            let name = (unsafe { pyre_object::w_int_get_value(args[2]) }) as libc::c_int;
            let val = args[3];
            let r = unsafe {
                if pyre_object::is_int(val) {
                    let v = pyre_object::w_int_get_value(val) as libc::c_int;
                    libc::setsockopt(
                        fd,
                        level,
                        name,
                        &v as *const _ as *const libc::c_void,
                        core::mem::size_of::<libc::c_int>() as libc::socklen_t,
                    )
                } else if pyre_object::bytesobject::is_bytes_like(val) {
                    let data = pyre_object::bytesobject::bytes_like_data(val);
                    libc::setsockopt(
                        fd,
                        level,
                        name,
                        data.as_ptr() as *const libc::c_void,
                        data.len() as libc::socklen_t,
                    )
                } else {
                    return Err(crate::PyError::type_error(
                        "setsockopt: value must be int or bytes-like",
                    ));
                }
            };
            if r != 0 {
                return Err(socket_io_err(std::io::Error::last_os_error()));
            }
            Ok(pyre_object::w_none())
        }),
    );

    crate::dict_storage_store(
        ns,
        "getsockopt",
        crate::make_builtin_function("getsockopt", |args| {
            if args.len() < 3 {
                return Err(crate::PyError::type_error(
                    "getsockopt() requires self + level + name [+ buflen]",
                ));
            }
            let fd = socket_fd(args[0])?;
            let level = (unsafe { pyre_object::w_int_get_value(args[1]) }) as libc::c_int;
            let name = (unsafe { pyre_object::w_int_get_value(args[2]) }) as libc::c_int;
            // `interp_socket.py:434-451 getsockopt_w` — `buflen == 0`
            // (including when omitted) reads an int option; otherwise the
            // length must be in `1..=1024` and a bytes buffer is returned.
            let buflen = if args.len() >= 4 {
                unsafe { pyre_object::w_int_get_value(args[3]) }
            } else {
                0
            };
            if buflen == 0 {
                let mut v: libc::c_int = 0;
                let mut sz = core::mem::size_of::<libc::c_int>() as libc::socklen_t;
                let r = unsafe {
                    libc::getsockopt(
                        fd,
                        level,
                        name,
                        &mut v as *mut _ as *mut libc::c_void,
                        &mut sz,
                    )
                };
                if r != 0 {
                    return Err(socket_io_err(std::io::Error::last_os_error()));
                }
                Ok(pyre_object::w_int_new(v as i64))
            } else {
                if buflen < 0 || buflen > 1024 {
                    return Err(crate::PyError::os_error("getsockopt buflen out of range"));
                }
                let buflen = buflen as usize;
                let mut buf = vec![0u8; buflen];
                let mut sz = buflen as libc::socklen_t;
                let r = unsafe {
                    libc::getsockopt(
                        fd,
                        level,
                        name,
                        buf.as_mut_ptr() as *mut libc::c_void,
                        &mut sz,
                    )
                };
                if r != 0 {
                    return Err(socket_io_err(std::io::Error::last_os_error()));
                }
                buf.truncate(sz as usize);
                Ok(pyre_object::bytesobject::w_bytes_from_bytes(&buf))
            }
        }),
    );

    // `interp_socket.py:777-797 setblocking_w` per PyPy docstring: True
    // is equivalent to `settimeout(None)`, False to `settimeout(0.0)`.
    // Routing through `socket_apply_timeout` keeps the SO_*TIMEO state
    // consistent with the timeout attribute and prevents a stale
    // SO_RCVTIMEO from surviving a `setblocking(True)` call.
    crate::dict_storage_store(
        ns,
        "setblocking",
        crate::make_builtin_function_with_arity(
            "setblocking",
            |args| {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error("setblocking() missing argument"));
                }
                let blocking = unsafe { pyre_object::w_int_get_value(args[1]) } != 0;
                let fd = socket_fd(args[0])?;
                let timeout = if blocking { -1.0 } else { 0.0 };
                socket_apply_timeout(fd, timeout)?;
                socket_set_attr(
                    args[0],
                    "_timeout",
                    if blocking {
                        pyre_object::w_none()
                    } else {
                        pyre_object::floatobject::w_float_new(0.0)
                    },
                );
                Ok(pyre_object::w_none())
            },
            2,
        ),
    );

    crate::dict_storage_store(
        ns,
        "getblocking",
        crate::make_builtin_function_with_arity(
            "getblocking",
            |args| {
                let fd = socket_fd(args.first().copied().unwrap_or(pyre_object::PY_NULL))?;
                let flags = unsafe { libc::fcntl(fd, libc::F_GETFL, 0) };
                if flags < 0 {
                    return Err(socket_io_err(std::io::Error::last_os_error()));
                }
                Ok(pyre_object::w_bool_from(flags & libc::O_NONBLOCK == 0))
            },
            1,
        ),
    );

    // `interp_socket.py:811-828 settimeout_w` then `rsocket.py:RSocket.
    // settimeout`: None → blocking (no O_NONBLOCK, no SO_*TIMEO); 0.0 →
    // non-blocking (O_NONBLOCK on); >0 → blocking + SO_RCVTIMEO +
    // SO_SNDTIMEO set to the duration; <0 → ValueError "Timeout value
    // out of range".
    crate::dict_storage_store(
        ns,
        "settimeout",
        crate::make_builtin_function_with_arity(
            "settimeout",
            |args| {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error("settimeout() missing argument"));
                }
                let obj = args[0];
                let w_t = args[1];
                let timeout: f64 = if unsafe { pyre_object::is_none(w_t) } {
                    -1.0
                } else {
                    let v = unsafe {
                        if pyre_object::is_float(w_t) {
                            pyre_object::floatobject::w_float_get_value(w_t)
                        } else if pyre_object::is_int(w_t) {
                            pyre_object::w_int_get_value(w_t) as f64
                        } else {
                            return Err(crate::PyError::type_error(
                                "settimeout: timeout must be a float or None",
                            ));
                        }
                    };
                    if v < 0.0 {
                        return Err(crate::PyError::value_error("Timeout value out of range"));
                    }
                    v
                };
                let fd = socket_fd(obj)?;
                socket_apply_timeout(fd, timeout)?;
                socket_set_attr(obj, "_timeout", w_t);
                Ok(pyre_object::w_none())
            },
            2,
        ),
    );

    crate::dict_storage_store(
        ns,
        "gettimeout",
        crate::make_builtin_function_with_arity(
            "gettimeout",
            |args| {
                let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                let d = crate::baseobjspace::getdict(obj);
                if d.is_null() {
                    return Ok(pyre_object::w_none());
                }
                Ok(unsafe { pyre_object::w_dict_getitem_str(d, "_timeout") }
                    .unwrap_or(pyre_object::w_none()))
            },
            1,
        ),
    );

    crate::dict_storage_store(
        ns,
        "__enter__",
        crate::make_builtin_function_with_arity(
            "__enter__",
            |args| Ok(args.first().copied().unwrap_or(pyre_object::w_none())),
            1,
        ),
    );

    crate::dict_storage_store(
        ns,
        "__exit__",
        crate::make_builtin_function("__exit__", |args| {
            if let Some(&obj) = args.first() {
                let fd = socket_get_attr_i64(obj, "_fd") as libc::c_int;
                if fd >= 0 {
                    let _ = unsafe { libc::close(fd) };
                    socket_set_attr(obj, "_fd", pyre_object::w_int_new(-1));
                }
            }
            Ok(pyre_object::w_bool_from(false))
        }),
    );

    // __repr__ — `interp_socket.py:304-312 descr_repr`.  Format
    // matches CPython: `<socket object, fd=N, family=F, type=T, proto=P>`.
    crate::dict_storage_store(
        ns,
        "__repr__",
        crate::make_builtin_function_with_arity(
            "__repr__",
            |args| {
                let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                let fd = socket_get_attr_i64(obj, "_fd");
                let family = socket_get_attr_i64(obj, "_family");
                let ty = socket_get_attr_i64(obj, "_type");
                let proto = socket_get_attr_i64(obj, "_proto");
                Ok(pyre_object::w_str_new(&format!(
                    "<socket object, fd={fd}, family={family}, type={ty}, proto={proto}>"
                )))
            },
            1,
        ),
    );

    // set_inheritable / get_inheritable — `interp_socket.py` wraps
    // the FD_CLOEXEC bit on `F_GETFD` / `F_SETFD`.
    crate::dict_storage_store(
        ns,
        "set_inheritable",
        crate::make_builtin_function_with_arity(
            "set_inheritable",
            |args| {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error(
                        "set_inheritable() missing argument",
                    ));
                }
                let fd = socket_fd(args[0])?;
                let want_inheritable = unsafe {
                    if pyre_object::is_bool(args[1]) {
                        pyre_object::boolobject::w_bool_get_value(args[1])
                    } else if pyre_object::is_int(args[1]) {
                        pyre_object::w_int_get_value(args[1]) != 0
                    } else {
                        return Err(crate::PyError::type_error(
                            "set_inheritable: value must be bool",
                        ));
                    }
                };
                let cur = unsafe { libc::fcntl(fd, libc::F_GETFD) };
                if cur < 0 {
                    return Err(socket_io_err(std::io::Error::last_os_error()));
                }
                let new = if want_inheritable {
                    cur & !libc::FD_CLOEXEC
                } else {
                    cur | libc::FD_CLOEXEC
                };
                if new != cur {
                    let r = unsafe { libc::fcntl(fd, libc::F_SETFD, new) };
                    if r < 0 {
                        return Err(socket_io_err(std::io::Error::last_os_error()));
                    }
                }
                Ok(pyre_object::w_none())
            },
            2,
        ),
    );
    crate::dict_storage_store(
        ns,
        "get_inheritable",
        crate::make_builtin_function_with_arity(
            "get_inheritable",
            |args| {
                let fd = socket_fd(args.first().copied().unwrap_or(pyre_object::PY_NULL))?;
                let r = unsafe { libc::fcntl(fd, libc::F_GETFD) };
                if r < 0 {
                    return Err(socket_io_err(std::io::Error::last_os_error()));
                }
                Ok(pyre_object::w_bool_from((r & libc::FD_CLOEXEC) == 0))
            },
            1,
        ),
    );
}

#[cfg(not(unix))]
fn socket_type() -> pyre_object::PyObjectRef {
    crate::typedef::w_object()
}
