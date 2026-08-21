//! _socket module — PyPy: pypy/module/_socket/interp_socket.py.
//!
//! Carries the W_Socket class implementation plus the shared address
//! conversion / IDNA / error-mapping helpers.  `register_module` is the
//! single entry point invoked by `moduledef::init`; it populates the
//! module namespace with constants, error classes, module-level
//! functions and the `socket` type definition.


/// The host socket layer, in the role `_rsocket_rffi.py` plays for rsocket:
/// one set of names over libc and WinSock, so the bodies below name a single
/// API.
#[cfg(any(unix, windows))]
use super::rsocket_rffi as rffi;

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

/// `interp_socket.py converted_error` — turn an rsocket
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
/// `interp_socket.py idna_converter` — turn a hostname argument
/// into a `Vec<u8>` suitable for passing to a DNS resolver.
///
/// Accepts str / bytes / bytearray.  For str: tries ASCII first; on
/// UnicodeEncodeError falls back to the `idna` codec.  Embedded null
/// bytes raise TypeError (matching `:120-122`).  Other input types
/// raise TypeError.
#[cfg(any(unix, windows))]
fn socket_idna_converter(w_host: pyre_object::PyObjectRef) -> Result<Vec<u8>, crate::PyError> {
    if w_host.is_null() {
        return Err(crate::PyError::type_error(
            "string or unicode text buffer expected, not None",
        ));
    }
    let bytes: Vec<u8> = unsafe {
        if pyre_object::is_str(w_host) {
            // The ASCII attempt reads the raw buffer, so a host that has no
            // utf-8 view — one carrying a lone surrogate — reaches the `idna`
            // fallback the same way any other non-ASCII host does.
            let s = pyre_object::unicodeobject::w_str_get_wtf8(w_host);
            if s.as_bytes().is_ascii() {
                s.as_bytes().to_vec()
            } else {
                // `space.encode_unicode_object(w_host, 'idna', None)`: the
                // codec runs on the string value itself rather than through
                // an `encode` attribute lookup, so a `str` subclass cannot
                // decide how its hostname reaches the resolver, and the
                // error a caller sees is the codec's own rejection.
                crate::type_methods::encode_object(w_host, "idna", "strict")?
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

#[cfg(any(unix, windows))]
fn socket_converted_error(
    applevelerrcls: &str,
    errno: Option<i32>,
    message: &str,
) -> crate::PyError {
    let cls = match applevelerrcls {
        "timeout" => crate::builtins::lookup_exc_class("TimeoutError"),
        "gaierror" => crate::builtins::lookup_exc_class("socket.gaierror"),
        "herror" => crate::builtins::lookup_exc_class("socket.herror"),
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

#[cfg(all(windows, feature = "host_env"))]
fn interface_io_error(error: std::io::Error) -> crate::PyError {
    use rustpython_host_env::os::ErrorExt;

    // The two scalar conversions fail through the C runtime `errno` the IP
    // Helper API sets, which `raw_os_error` leaves empty on purpose so that no
    // `winerror` is attached to it; `posix_errno` is what recovers it.  The
    // enumeration path instead fails through a Win32 status, and that one is
    // reported as the Win32 flavour it is.
    let Some(winerror) = error.raw_os_error() else {
        let errno = error.posix_errno();
        let message = match errno {
            libc::ENODEV => "No such device".to_string(),
            libc::ENXIO => "No such device or address".to_string(),
            _ => error.to_string(),
        };
        return crate::PyError::os_error_with_errno(errno, message);
    };
    crate::PyError::os_error_win32_syscall2(winerror, pyre_object::PY_NULL, pyre_object::PY_NULL)
}

/// One `space.acquire_writebuf` export held for the whole of a `recv_into`
/// family call, the way `with rwbuffer:` keeps it across `c_recv`.
///
/// The syscall runs with the GIL released, so without the export another
/// thread could resize the exporter and reallocate the storage the kernel is
/// writing through; the count is what makes that resize raise instead.  The
/// requested object and the concrete storage owner are both rooted, because
/// the release in `Drop` has to name the owner after whatever Python ran in
/// between.
#[cfg(any(unix, windows))]
struct SocketWritableBuffer {
    _roots: pyre_object::gc_roots::RootScope,
    owner_slot: usize,
    held: bool,
    address: *mut u8,
    length: usize,
}

#[cfg(any(unix, windows))]
impl SocketWritableBuffer {
    unsafe fn acquire(obj: pyre_object::PyObjectRef) -> Result<Self, crate::PyError> {
        let roots = pyre_object::gc_roots::push_roots();
        pyre_object::gc_roots::pin_root(obj);
        let obj_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let (data, owner) =
            socket_writebuf(pyre_object::gc_roots::shadow_stack_get(obj_slot))?;
        pyre_object::gc_roots::pin_root(owner);
        let owner_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let owner = pyre_object::gc_roots::shadow_stack_get(owner_slot);
        let held = crate::builtins::buffer_export_incref(owner);
        Ok(Self {
            _roots: roots,
            owner_slot,
            held,
            address: data.as_mut_ptr(),
            length: data.len(),
        })
    }

    unsafe fn as_mut_slice(&mut self) -> &mut [u8] {
        std::slice::from_raw_parts_mut(self.address, self.length)
    }
}

#[cfg(any(unix, windows))]
impl Drop for SocketWritableBuffer {
    fn drop(&mut self) {
        if self.held {
            let owner = pyre_object::gc_roots::shadow_stack_get(self.owner_slot);
            unsafe { crate::builtins::buffer_export_decref(owner) };
        }
    }
}

/// baseobjspace.py `writebuf_w` — the writable byte slice backing a
/// scatter/gather buffer argument, with the object the export count lives on.
/// PyPy accepts any object exporting a writable buffer; pyre's writable byte
/// stores are `bytearray` and a `memoryview` over one, so those are resolved
/// here and anything else is rejected as `writebuf_w` does.
#[cfg(any(unix, windows))]
fn socket_writebuf(
    obj: pyre_object::PyObjectRef,
) -> Result<(&'static mut [u8], pyre_object::PyObjectRef), crate::PyError> {
    if unsafe { pyre_object::bytearrayobject::is_bytearray(obj) } {
        return Ok((
            unsafe { pyre_object::bytearrayobject::w_bytearray_data_mut(obj) },
            obj,
        ));
    }
    if unsafe { pyre_object::interp_array::is_array(obj) } {
        // `space.writebuf_w` accepts any writable buffer exporter; an
        // `array.array` exposes its element bytes as one writable window
        // regardless of typecode (recv writes raw bytes into them).
        return Ok((
            unsafe { pyre_object::interp_array::w_array_vec_mut(obj).as_mut_slice() },
            obj,
        ));
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
        let view = unsafe { pyre_object::memoryview::w_memoryview_view(obj) };
        // The view itself carries the export, as it does for `readinto`. Its
        // backing is already held by the view's own construction, so nothing
        // can resize the storage underneath while this one is alive either.
        let owner = obj;
        // A writable view's backing is a `bytearray` or an `array.array`; both
        // expose a mutable byte store.  Honour the view window: write only into
        // `[offset, offset+length)` of the backing storage (itself already the
        // `Buffer::Sub` window for a zero-copy slice), not the whole buffer.
        let Some(full) = (unsafe { view.backing().as_bytes_mut() }) else {
            return Err(crate::PyError::type_error("cannot modify read-only memory"));
        };
        let off = unsafe { view.offset() } as usize;
        let len = unsafe { pyre_object::memoryview::w_memoryview_length(obj) } as usize;
        // The backing may have been resized after the view was taken; reject a
        // window that no longer fits rather than panic.
        if off.checked_add(len).is_none_or(|end| end > full.len()) {
            return Err(crate::PyError::value_error(
                "memoryview buffer is no longer valid",
            ));
        }
        return Ok((&mut full[off..off + len], owner));
    }
    Err(crate::PyError::type_error(
        "a writable bytes-like object is required",
    ))
}

pub fn register_module(ns: pyre_object::PyObjectRef) {
    // `_rsocket_rffi.py:1150 rwin32.get_wsa_error`'s companion: WinSock has to
    // be started before any of its entry points answers, so the module takes
    // that cost at import rather than leaving the first call to fail with
    // WSANOTINITIALISED.  The individual entry points repeat it, since nothing
    // guarantees `_socket` is the first importer.
    #[cfg(any(unix, windows))]
    rffi::init();

    // `_rsocket_rffi.py constant_names` + `:234-262
    // constants_w_defaults` — populated through the libc crate where
    // available, hardcoded for platform-specific constants the crate
    // does not expose.  Mirrors PyPy's
    // `for constant, value in rsocket.constants.iteritems(): wrap(value)`
    // loop in `_socket/moduledef.py:48-50`.
    #[cfg(unix)]
    {
        macro_rules! cst {
            ($name:literal, $val:expr) => {
                crate::module_ns_store(ns, $name, pyre_object::w_int_new($val as i64));
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
        // libc deprecates this: the kernel raised the value and libc will
        // follow upstream in a later release (rust-lang/libc#1896). Keep
        // reading it from libc rather than freezing a literal, so the new
        // value arrives with the dependency; `socketmodule.c` likewise takes
        // whatever the platform header defines.
        #[allow(deprecated)]
        {
            cst!("IPPROTO_MAX", libc::IPPROTO_MAX);
        }
        cst!("IPPROTO_GRE", libc::IPPROTO_GRE);
        cst!("IPPROTO_RSVP", libc::IPPROTO_RSVP);
        // `_rsocket_rffi.py constants_w_defaults` — SOL_IP/TCP/UDP
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

    // `_rsocket_rffi.py` keeps a separate `_MSVC` constant list because the
    // two headers name overlapping but different sets; this is that list.
    // The `Networking::WinSock` values are `<winsock2.h>`/`<ws2tcpip.h>`
    // themselves, so what is absent here is absent from the platform.
    #[cfg(windows)]
    {
        use windows_sys::Win32::Networking::WinSock as ws;
        macro_rules! cst {
            ($name:literal, $val:expr) => {
                crate::module_ns_store(ns, $name, pyre_object::w_int_new($val as i64));
            };
        }
        use windows_sys::Win32::Devices::Bluetooth as bt;
        // ── Address families ──
        cst!("AF_UNSPEC", ws::AF_UNSPEC);
        cst!("AF_INET", ws::AF_INET);
        cst!("AF_INET6", ws::AF_INET6);
        cst!("AF_APPLETALK", ws::AF_APPLETALK);
        cst!("AF_DECnet", ws::AF_DECnet);
        cst!("AF_IPX", ws::AF_IPX);
        cst!("AF_LINK", ws::AF_LINK);
        // `socketmodule.c:PyInit__socket` publishes the address families the
        // 3.14 Windows SDK exposes in addition to PyPy's older MSVC census.
        cst!("AF_SNA", 11);
        cst!("AF_IRDA", 26);
        cst!("AF_BLUETOOTH", bt::AF_BTH);
        cst!("AF_HYPERV", ws::AF_HYPERV);
        // ── Socket types ──
        cst!("SOCK_STREAM", ws::SOCK_STREAM);
        cst!("SOCK_DGRAM", ws::SOCK_DGRAM);
        cst!("SOCK_RAW", ws::SOCK_RAW);
        cst!("SOCK_RDM", ws::SOCK_RDM);
        cst!("SOCK_SEQPACKET", ws::SOCK_SEQPACKET);
        // ── Protocols ──
        cst!("IPPROTO_IP", ws::IPPROTO_IP);
        cst!("IPPROTO_HOPOPTS", ws::IPPROTO_HOPOPTS);
        cst!("IPPROTO_ICMP", ws::IPPROTO_ICMP);
        cst!("IPPROTO_IGMP", ws::IPPROTO_IGMP);
        cst!("IPPROTO_GGP", ws::IPPROTO_GGP);
        cst!("IPPROTO_ST", 5);
        cst!("IPPROTO_CBT", 7);
        cst!("IPPROTO_IGP", 9);
        cst!("IPPROTO_IPV4", ws::IPPROTO_IPV4);
        cst!("IPPROTO_TCP", ws::IPPROTO_TCP);
        cst!("IPPROTO_EGP", ws::IPPROTO_EGP);
        cst!("IPPROTO_PUP", ws::IPPROTO_PUP);
        cst!("IPPROTO_UDP", ws::IPPROTO_UDP);
        cst!("IPPROTO_IDP", ws::IPPROTO_IDP);
        cst!("IPPROTO_ICLFXBM", 78);
        cst!("IPPROTO_IPV6", ws::IPPROTO_IPV6);
        cst!("IPPROTO_ROUTING", ws::IPPROTO_ROUTING);
        cst!("IPPROTO_FRAGMENT", ws::IPPROTO_FRAGMENT);
        cst!("IPPROTO_ESP", ws::IPPROTO_ESP);
        cst!("IPPROTO_AH", ws::IPPROTO_AH);
        cst!("IPPROTO_ICMPV6", ws::IPPROTO_ICMPV6);
        cst!("IPPROTO_NONE", ws::IPPROTO_NONE);
        cst!("IPPROTO_DSTOPTS", ws::IPPROTO_DSTOPTS);
        cst!("IPPROTO_ND", ws::IPPROTO_ND);
        cst!("IPPROTO_PIM", ws::IPPROTO_PIM);
        cst!("IPPROTO_PGM", ws::IPPROTO_PGM);
        cst!("IPPROTO_RDP", ws::IPPROTO_RDP);
        cst!("IPPROTO_L2TP", 115);
        cst!("IPPROTO_SCTP", ws::IPPROTO_SCTP);
        cst!("IPPROTO_RAW", ws::IPPROTO_RAW);
        cst!("IPPROTO_MAX", ws::IPPROTO_MAX);
        // `_rsocket_rffi.py constants_w_defaults` — SOL_TCP/UDP kept for
        // PyPy compatibility.  `SOL_IP` is the platform's own here:
        // `ws2def.h` defines it, so the zero placeholder the other arm uses
        // would name `IPPROTO_IP` instead of the level.
        cst!("SOL_IP", ws::SOL_IP);
        cst!("SOL_TCP", 6);
        cst!("SOL_UDP", 17);
        // ── INADDR_* (host byte order) ──
        //
        // `PyModule_AddIntConstant` takes a C `long`, which is 32 bits here, so
        // every one of these with the top bit set is published as the negative
        // number that word spells rather than as its unsigned reading.
        cst!("INADDR_ANY", ws::INADDR_ANY as i32);
        cst!("INADDR_LOOPBACK", ws::INADDR_LOOPBACK as i32);
        cst!("INADDR_BROADCAST", ws::INADDR_BROADCAST as i32);
        cst!("INADDR_NONE", ws::INADDR_NONE as i32);
        cst!("INADDR_ALLHOSTS_GROUP", 0xe0000001u32 as i32);
        cst!("INADDR_UNSPEC_GROUP", 0xe0000000u32 as i32);
        cst!("INADDR_MAX_LOCAL_GROUP", 0xe00000ffu32 as i32);
        cst!("IPPORT_RESERVED", ws::IPPORT_RESERVED);
        cst!("IPPORT_USERRESERVED", 5000);
        // ── SOL_* / SO_* (socket level) ──
        cst!("SOL_SOCKET", ws::SOL_SOCKET);
        cst!("SO_REUSEADDR", ws::SO_REUSEADDR);
        cst!("SO_EXCLUSIVEADDRUSE", ws::SO_EXCLUSIVEADDRUSE);
        cst!("SO_KEEPALIVE", ws::SO_KEEPALIVE);
        cst!("SO_BROADCAST", ws::SO_BROADCAST);
        cst!("SO_DEBUG", ws::SO_DEBUG);
        cst!("SO_DONTROUTE", ws::SO_DONTROUTE);
        cst!("SO_LINGER", ws::SO_LINGER);
        cst!("SO_OOBINLINE", ws::SO_OOBINLINE);
        cst!("SO_RCVBUF", ws::SO_RCVBUF);
        cst!("SO_SNDBUF", ws::SO_SNDBUF);
        cst!("SO_RCVTIMEO", ws::SO_RCVTIMEO);
        cst!("SO_SNDTIMEO", ws::SO_SNDTIMEO);
        cst!("SO_SNDLOWAT", 0x1003);
        cst!("SO_RCVLOWAT", 0x1004);
        cst!("SO_ERROR", ws::SO_ERROR);
        cst!("SO_TYPE", ws::SO_TYPE);
        cst!("SO_ACCEPTCONN", ws::SO_ACCEPTCONN);
        cst!("SO_USELOOPBACK", ws::SO_USELOOPBACK);
        cst!("SO_ORIGINAL_DST", 12303);
        // Bluetooth/RFCOMM constants.  The option names are unsigned SDK
        // words but `PyModule_AddIntConstant` exposes their signed C-long
        // readings on Windows.
        cst!("BTPROTO_RFCOMM", bt::BTHPROTO_RFCOMM);
        cst!("SOL_RFCOMM", 3);
        cst!("SO_BTH_ENCRYPT", 2);
        cst!("SO_BTH_MTU", 0x80000007u32 as i32);
        cst!("SO_BTH_MTU_MAX", 0x80000008u32 as i32);
        cst!("SO_BTH_MTU_MIN", 0x8000000au32 as i32);
        // ── TCP-level ──
        cst!("TCP_NODELAY", ws::TCP_NODELAY);
        cst!("TCP_MAXSEG", ws::TCP_MAXSEG);
        cst!("TCP_KEEPIDLE", 3);
        cst!("TCP_FASTOPEN", 15);
        cst!("TCP_KEEPCNT", 16);
        cst!("TCP_KEEPINTVL", 17);
        // `SIO_TCP_SET_ACK_FREQUENCY` - the name `socketmodule.c` gives this
        // option on Windows.  It is an ioctl code, not an option number, which
        // is why `setsockopt` and `getsockopt` both special-case it.
        cst!("TCP_QUICKACK", ws::SIO_TCP_SET_ACK_FREQUENCY as i32);
        // ── IP-level ──
        cst!("IP_TTL", ws::IP_TTL);
        cst!("IP_TOS", ws::IP_TOS);
        cst!("IP_OPTIONS", ws::IP_OPTIONS);
        cst!("IP_MULTICAST_TTL", ws::IP_MULTICAST_TTL);
        cst!("IP_MULTICAST_LOOP", ws::IP_MULTICAST_LOOP);
        cst!("IP_MULTICAST_IF", ws::IP_MULTICAST_IF);
        cst!("IP_ADD_MEMBERSHIP", ws::IP_ADD_MEMBERSHIP);
        cst!("IP_DROP_MEMBERSHIP", ws::IP_DROP_MEMBERSHIP);
        cst!("IP_HDRINCL", ws::IP_HDRINCL);
        cst!("IP_RECVDSTADDR", ws::IP_RECVDSTADDR);
        cst!("IP_ADD_SOURCE_MEMBERSHIP", 15);
        cst!("IP_DROP_SOURCE_MEMBERSHIP", 16);
        cst!("IP_BLOCK_SOURCE", 17);
        cst!("IP_UNBLOCK_SOURCE", 18);
        cst!("IP_PKTINFO", 19);
        cst!("IP_RECVTTL", 21);
        cst!("IP_RECVTOS", 40);
        cst!("IP_RECVERR", 75);
        cst!("IP_DEFAULT_MULTICAST_LOOP", 1);
        cst!("IP_DEFAULT_MULTICAST_TTL", 1);
        cst!("IP_MAX_MEMBERSHIPS", 20);
        // ── IPv6 ──
        cst!("IPV6_V6ONLY", ws::IPV6_V6ONLY);
        cst!("IPV6_CHECKSUM", ws::IPV6_CHECKSUM);
        cst!("IPV6_DONTFRAG", ws::IPV6_DONTFRAG);
        cst!("IPV6_HOPLIMIT", ws::IPV6_HOPLIMIT);
        cst!("IPV6_HOPOPTS", ws::IPV6_HOPOPTS);
        cst!("IPV6_JOIN_GROUP", ws::IPV6_JOIN_GROUP);
        cst!("IPV6_LEAVE_GROUP", ws::IPV6_LEAVE_GROUP);
        cst!("IPV6_MULTICAST_HOPS", ws::IPV6_MULTICAST_HOPS);
        cst!("IPV6_MULTICAST_IF", ws::IPV6_MULTICAST_IF);
        cst!("IPV6_MULTICAST_LOOP", ws::IPV6_MULTICAST_LOOP);
        cst!("IPV6_PKTINFO", ws::IPV6_PKTINFO);
        cst!("IPV6_RECVRTHDR", ws::IPV6_RECVRTHDR);
        cst!("IPV6_RECVTCLASS", ws::IPV6_RECVTCLASS);
        cst!("IPV6_RECVERR", 75);
        cst!("IPV6_RTHDR", ws::IPV6_RTHDR);
        cst!("IPV6_TCLASS", ws::IPV6_TCLASS);
        cst!("IPV6_UNICAST_HOPS", ws::IPV6_UNICAST_HOPS);
        // ── shutdown how ──
        cst!("SHUT_RD", ws::SD_RECEIVE);
        cst!("SHUT_WR", ws::SD_SEND);
        cst!("SHUT_RDWR", ws::SD_BOTH);
        // ── Message flags ──
        cst!("MSG_OOB", ws::MSG_OOB);
        cst!("MSG_PEEK", ws::MSG_PEEK);
        cst!("MSG_DONTROUTE", ws::MSG_DONTROUTE);
        cst!("MSG_WAITALL", ws::MSG_WAITALL);
        cst!("MSG_CTRUNC", ws::MSG_CTRUNC);
        cst!("MSG_TRUNC", ws::MSG_TRUNC);
        cst!("MSG_BCAST", ws::MSG_BCAST);
        cst!("MSG_MCAST", ws::MSG_MCAST);
        cst!("MSG_ERRQUEUE", 0x1000);
        // ── Address-info flags ──
        cst!("AI_PASSIVE", ws::AI_PASSIVE);
        cst!("AI_CANONNAME", ws::AI_CANONNAME);
        cst!("AI_NUMERICHOST", ws::AI_NUMERICHOST);
        cst!("AI_NUMERICSERV", ws::AI_NUMERICSERV);
        cst!("AI_ADDRCONFIG", ws::AI_ADDRCONFIG);
        cst!("AI_V4MAPPED", ws::AI_V4MAPPED);
        cst!("AI_ALL", ws::AI_ALL);
        // ── Name-info flags ──
        cst!("NI_NUMERICHOST", ws::NI_NUMERICHOST);
        cst!("NI_NUMERICSERV", ws::NI_NUMERICSERV);
        cst!("NI_NOFQDN", ws::NI_NOFQDN);
        cst!("NI_NAMEREQD", ws::NI_NAMEREQD);
        cst!("NI_DGRAM", ws::NI_DGRAM);
        cst!("NI_MAXHOST", ws::NI_MAXHOST);
        cst!("NI_MAXSERV", ws::NI_MAXSERV);
        // ── EAI_* — WSA error codes under their <ws2tcpip.h> aliases ──
        cst!("EAI_AGAIN", ws::WSATRY_AGAIN);
        cst!("EAI_BADFLAGS", ws::WSAEINVAL);
        cst!("EAI_FAIL", ws::WSANO_RECOVERY);
        cst!("EAI_FAMILY", ws::WSAEAFNOSUPPORT);
        cst!("EAI_MEMORY", ws::WSA_NOT_ENOUGH_MEMORY);
        // `ws2tcpip.h` spells `EAI_NODATA` as `EAI_NONAME`, the RFC 3493
        // deprecation, so both name `WSAHOST_NOT_FOUND` and not `WSANO_DATA`.
        cst!("EAI_NODATA", ws::WSAHOST_NOT_FOUND);
        cst!("EAI_NONAME", ws::WSAHOST_NOT_FOUND);
        cst!("EAI_SERVICE", ws::WSATYPE_NOT_FOUND);
        cst!("EAI_SOCKTYPE", ws::WSAESOCKTNOSUPPORT);
        // ── WSAIoctl codes `socket.ioctl` names ──
        cst!("SIO_RCVALL", ws::SIO_RCVALL);
        cst!("SIO_KEEPALIVE_VALS", ws::SIO_KEEPALIVE_VALS);
        cst!("SIO_LOOPBACK_FAST_PATH", ws::SIO_LOOPBACK_FAST_PATH);
        cst!("RCVALL_OFF", ws::RCVALL_OFF);
        cst!("RCVALL_ON", ws::RCVALL_ON);
        cst!("RCVALL_SOCKETLEVELONLY", ws::RCVALL_SOCKETLEVELONLY);
        cst!("RCVALL_IPLEVEL", ws::RCVALL_IPLEVEL);
        cst!("RCVALL_MAX", 3);
        // Hyper-V socket ABI constants (`hvsocket.h`).  GUIDs and Bluetooth
        // addresses are public strings rather than integer enum members.
        cst!("HV_PROTOCOL_RAW", 1);
        cst!("HVSOCKET_CONNECT_TIMEOUT", 1);
        cst!("HVSOCKET_CONNECTED_SUSPEND", 4);
        cst!("HVSOCKET_CONNECT_TIMEOUT_MAX", 300_000);
        cst!("HVSOCKET_ADDRESS_FLAG_PASSTHRU", 1);
        for (name, value) in [
            ("BDADDR_ANY", "00:00:00:00:00:00"),
            ("BDADDR_LOCAL", "00:00:00:FF:FF:FF"),
            ("HV_GUID_ZERO", "00000000-0000-0000-0000-000000000000"),
            ("HV_GUID_WILDCARD", "00000000-0000-0000-0000-000000000000"),
            ("HV_GUID_BROADCAST", "FFFFFFFF-FFFF-FFFF-FFFF-FFFFFFFFFFFF"),
            ("HV_GUID_CHILDREN", "90DB8B89-0D35-4F79-8CE9-49EA0AC8B7CD"),
            ("HV_GUID_LOOPBACK", "E0E16197-DD56-4A10-9195-5EE7A155A838"),
            ("HV_GUID_PARENT", "A42E7CDA-D03F-480C-9CC2-A4DE20ABB878"),
        ] {
            crate::module_ns_store(ns, name, pyre_object::w_str_new(value));
        }
        // ── socket-level cap ──
        // `<winsock2.h>` defines SOMAXCONN as 0x7fffffff; the WinSock 1.1
        // value of 5 the metadata carries is the one `listen` outgrew.
        cst!("SOMAXCONN", 0x7fffffffi64);
    }

    // ── htons / htonl / ntohs / ntohl ──
    crate::module_ns_store(
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
    crate::module_ns_store(
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
    crate::module_ns_store(
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
    crate::module_ns_store(
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
    #[cfg(any(unix, windows))]
    {
        crate::module_ns_store(
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
                        crate::baseobjspace::str_utf8_w(args[0])?.to_string()
                    };
                    let c = std::ffi::CString::new(s.as_bytes())
                        .map_err(|_| crate::PyError::value_error("embedded null in argument"))?;
                    let Some(bytes) = rffi::inet_aton(&c) else {
                        return Err(crate::PyError::os_error(
                            "illegal IP address string passed to inet_aton",
                        ));
                    };
                    Ok(pyre_object::bytesobject::w_bytes_from_bytes(&bytes))
                },
                1,
            ),
        );
        crate::module_ns_store(
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
                    let Some(text) = rffi::inet_ntoa([data[0], data[1], data[2], data[3]]) else {
                        return Err(crate::PyError::os_error("inet_ntoa failed"));
                    };
                    Ok(pyre_object::w_str_new(&text))
                },
                1,
            ),
        );

        // inet_pton(af, ip) → bytes
        crate::module_ns_store(
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
                        crate::baseobjspace::str_utf8_w(args[1])?.to_string()
                    };
                    let c_ip = std::ffi::CString::new(ip.as_bytes())
                        .map_err(|_| crate::PyError::value_error("embedded null"))?;
                    let mut buf = [0u8; 16];
                    let r = unsafe {
                        rffi::inet_pton(af, c_ip.as_ptr(), buf.as_mut_ptr() as *mut libc::c_void)
                    };
                    if r != 1 {
                        return Err(crate::PyError::os_error(
                            "illegal IP address string passed to inet_pton",
                        ));
                    }
                    let n = match af {
                        x if x == rffi::AF_INET => 4,
                        x if x == rffi::AF_INET6 => 16,
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
        crate::module_ns_store(
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
                        x if x == rffi::AF_INET => 4,
                        x if x == rffi::AF_INET6 => 16,
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
                        rffi::inet_ntop(
                            af,
                            data.as_ptr() as *const libc::c_void,
                            buf.as_mut_ptr() as *mut libc::c_char,
                            buf.len() as rffi::SockLen,
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
        crate::module_ns_store(
            ns,
            "gethostname",
            crate::make_builtin_function_with_arity(
                "gethostname",
                |_| {
                    let name = rffi::hostname().map_err(|e| {
                        crate::PyError::os_error_with_errno(
                            e.raw_os_error().unwrap_or(0),
                            "gethostname",
                        )
                    })?;
                    // `interp_func.py:24` is
                    // `space.fsdecode(space.newbytes(res))` -- the hostname is
                    // opaque kernel bytes (`sethostname(2)` takes a plain
                    // `const char*`), so a byte with no UTF-8 spelling has to
                    // survive as its surrogate escape.
                    Ok(crate::gateway::fsdecode_os_str(&name))
                },
                0,
            ),
        );
    }

    // `sethostname` alone stays POSIX-only: WinSock has no counterpart and
    // `moduledef.py` does not export it where rsocket cannot provide it.
    #[cfg(all(unix, feature = "host_env"))]
    {
        // sethostname(name) → None  (host_env::socket-backed)
        crate::module_ns_store(
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
                    // `interp_func.py:405-411` tests the two accepted types in
                    // turn — a bytes name is taken as it stands, a str one is
                    // fsencoded — so a byte with no UTF-8 spelling reaches the
                    // syscall as itself, and anything else is a TypeError
                    // naming those two types.  `fsencode_w` would also accept a
                    // `__fspath__` object, which this entry point does not.
                    let name = unsafe {
                        if pyre_object::bytesobject::is_bytes(args[0]) {
                            pyre_object::bytesobject::w_bytes_data(args[0]).to_vec()
                        } else if pyre_object::is_str(args[0]) {
                            crate::gateway::fsencode(args[0])?
                        } else {
                            return Err(crate::PyError::type_error(
                                "sethostname() argument 1 must be str or bytes",
                            ));
                        }
                    };
                    // `interp_func.py:412` audits the argument as it was
                    // passed, after the conversion and before the syscall.
                    crate::module::sys::vm::audit("socket.sethostname", &[args[0]])?;
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
    }

    // The legacy resolvers. Both platforms reach them through
    // `rsocket_rffi`'s accessors, because the records they answer with are
    // not the same type on each — see the netdb section there.
    #[cfg(any(unix, windows))]
    {
        // gethostbyname(name) → ip_string.  `interp_func.py` —
        // host argument runs through encode_idna (→ idna_converter)
        // before the rsocket call.
        crate::module_ns_store(
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
                    let _netdb = rffi::netdb_lock();
                    let he = unsafe { rffi::host_by_name(c.as_ptr()) };
                    if he.is_null() {
                        let host_repr = String::from_utf8_lossy(&host_bytes).into_owned();
                        return Err(socket_converted_error(
                            "gaierror",
                            None,
                            &format!("gethostbyname failed for {host_repr}"),
                        ));
                    }
                    unsafe {
                        let first_addr = rffi::pointer_at(rffi::hostent_addr_list(he), 0);
                        if rffi::hostent_length(he) != 4 || first_addr.is_null() {
                            return Err(socket_converted_error(
                                "gaierror",
                                None,
                                "gethostbyname: no IPv4 address",
                            ));
                        }
                        let packed = std::ptr::read_unaligned(first_addr as *const u32);
                        let Some(text) = rffi::inet_ntoa(packed.to_ne_bytes()) else {
                            return Err(socket_converted_error(
                                "gaierror",
                                None,
                                "gethostbyname: address is not representable",
                            ));
                        };
                        Ok(pyre_object::w_str_new(&text))
                    }
                },
                1,
            ),
        );

        // gethostbyname_ex(name) → (name, aliases, addresses)
        // `interp_func.py` — same lookup as gethostbyname but
        // returns the full hostent triple.
        crate::module_ns_store(
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
                    let _netdb = rffi::netdb_lock();
                    let he = unsafe { rffi::host_by_name(c.as_ptr()) };
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
        crate::module_ns_store(
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
                    let _netdb = rffi::netdb_lock();
                    // Try IPv4 first, then IPv6, then fall back to
                    // gethostbyname → hostent.h_addr to obtain a raw
                    // bytestring for gethostbyaddr.
                    let mut buf4 = [0u8; 4];
                    let r4 = unsafe {
                        rffi::inet_pton(
                            rffi::AF_INET,
                            c.as_ptr(),
                            buf4.as_mut_ptr() as *mut libc::c_void,
                        )
                    };
                    let (family, addr_ptr, addr_len) = if r4 == 1 {
                        (
                            rffi::AF_INET,
                            buf4.as_ptr() as *const libc::c_void,
                            4 as rffi::SockLen,
                        )
                    } else {
                        let mut buf6 = [0u8; 16];
                        let r6 = unsafe {
                            rffi::inet_pton(
                                rffi::AF_INET6,
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
                                rffi::host_by_addr(
                                    owned.as_mut_ptr() as *const libc::c_void,
                                    16 as rffi::SockLen,
                                    rffi::AF_INET6,
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
                        let he = unsafe { rffi::host_by_name(c.as_ptr()) };
                        if he.is_null() {
                            let host_repr = String::from_utf8_lossy(&host_bytes).into_owned();
                            return Err(socket_converted_error(
                                "herror",
                                None,
                                &format!("gethostbyaddr failed for {host_repr}"),
                            ));
                        }
                        unsafe {
                            let first_addr =
                                rffi::pointer_at(rffi::hostent_addr_list(he), 0);
                            if first_addr.is_null() {
                                return Err(socket_converted_error(
                                    "herror",
                                    None,
                                    "gethostbyaddr: empty address list",
                                ));
                            }
                            (
                                rffi::hostent_addr_type(he),
                                first_addr as *const libc::c_void,
                                rffi::hostent_length(he) as rffi::SockLen,
                            )
                        }
                    };
                    let he = unsafe { rffi::host_by_addr(addr_ptr, addr_len, family) };
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
        crate::module_ns_store(
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
                    crate::baseobjspace::str_utf8_w(args[0])?.to_string()
                };
                let c_name = std::ffi::CString::new(name.as_bytes())
                    .map_err(|_| crate::PyError::value_error("embedded null"))?;
                let proto_c: Option<std::ffi::CString> =
                    if args.len() >= 2 && unsafe { pyre_object::is_str(args[1]) } {
                        let p = crate::baseobjspace::str_utf8_w(args[1])?.to_string();
                        Some(
                            std::ffi::CString::new(p.as_bytes())
                                .map_err(|_| crate::PyError::value_error("embedded null"))?,
                        )
                    } else {
                        None
                    };
                let p = unsafe {
                    rffi::serv_by_name(
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
                let port = unsafe { u16::from_be(rffi::servent_port(p)) };
                Ok(pyre_object::w_int_new(port as i64))
            }),
        );

        // getservbyport(port[, proto]) → name
        crate::module_ns_store(
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
                        let p = crate::baseobjspace::str_utf8_w(args[1])?.to_string();
                        Some(
                            std::ffi::CString::new(p.as_bytes())
                                .map_err(|_| crate::PyError::value_error("embedded null"))?,
                        )
                    } else {
                        None
                    };
                let p = unsafe {
                    rffi::serv_by_port(
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
                    std::ffi::CStr::from_ptr(rffi::servent_name(p))
                        .to_string_lossy()
                        .into_owned()
                };
                Ok(pyre_object::w_str_new(&name))
            }),
        );
    }

    // `moduledef.py:12-16`:
    //   error    = get_error(space, "error")
    //   herror   = get_error(space, "herror")
    //   gaierror = get_error(space, "gaierror")
    //   timeout  = space.w_TimeoutError
    // `socketmodule.c` names them `socket.herror` / `socket.gaierror`
    // instead, and `type.__module__` reads the qualified prefix back, so
    // `socket.gaierror.__module__` is `"socket"` rather than `"_socket"`.
    let w_os_error = crate::builtins::lookup_exc_class("OSError")
        .expect("OSError must be installed before _socket init");
    crate::module_ns_store(ns, "error", w_os_error);
    crate::module_ns_store(
        ns,
        "herror",
        crate::builtins::make_exc_type(
            "socket.herror",
            crate::builtins::exc_exception_new,
            w_os_error,
        ),
    );
    crate::module_ns_store(
        ns,
        "gaierror",
        crate::builtins::make_exc_type(
            "socket.gaierror",
            crate::builtins::exc_exception_new,
            w_os_error,
        ),
    );
    let w_timeout_error = crate::builtins::lookup_exc_class("TimeoutError")
        .expect("TimeoutError must be installed before _socket init");
    crate::module_ns_store(ns, "timeout", w_timeout_error);

    // Default timeout (None) — modulus has a getter/setter; we just stash
    // a None so attribute lookups succeed.
    crate::module_ns_store(ns, "_default_timeout", pyre_object::w_none());

    // `_rsocket_rffi.py constants['has_ipv6'] = True` — exposed by
    // PyPy's moduledef.py constants loop as a module-level boolean.
    crate::module_ns_store(ns, "has_ipv6", pyre_object::boolobject::w_bool_from(true));

    // ── module-level getdefaulttimeout / setdefaulttimeout ──
    // `interp_func.py:378-397` — None means "blocking", float means
    // "timeout in seconds".  Stored as a process-wide cell.
    crate::module_ns_store(
        ns,
        "getdefaulttimeout",
        crate::make_builtin_function_with_arity(
            "getdefaulttimeout",
            |_| Ok(get_default_socket_timeout()),
            0,
        ),
    );
    crate::module_ns_store(
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
    // `interp_socket.py:close(fd)` — the bare host close, used for
    // cleanup when callers obtain a descriptor via .detach().
    #[cfg(any(unix, windows))]
    crate::module_ns_store(
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
                let fd = rffi::socket_from_i64(unsafe { pyre_object::w_int_get_value(args[0]) });
                if unsafe { rffi::close(fd) } != 0 {
                    return Err(socket_last_error());
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
    #[cfg(any(unix, windows))]
    crate::module_ns_store(
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
                let name = crate::baseobjspace::str_utf8_w(args[0])?.to_string();
                let c_name = std::ffi::CString::new(name.as_bytes())
                    .map_err(|_| crate::PyError::value_error("embedded null in name"))?;
                let Some(proto) = rffi::protocol_by_name(&c_name) else {
                    return Err(socket_converted_error("error", None, "protocol not found"));
                };
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
        crate::module_ns_store(
            ns,
            "if_nameindex",
            crate::make_builtin_function_with_arity(
                "if_nameindex",
                |_| {
                    let head = unsafe { libc::if_nameindex() };
                    if head.is_null() {
                        return Err(socket_last_error());
                    }
                    let mut items = Vec::new();
                    let mut p = head;
                    unsafe {
                        while (*p).if_index != 0 && !(*p).if_name.is_null() {
                            // An interface name is an OS string: `dev_valid_name`
                            // rejects only NUL, '/', ':', whitespace, '.' and
                            // '..', so any other octet is legal.  The sibling
                            // `if_nametoindex` below fsencodes, so decoding
                            // here any other way breaks the round trip.
                            let name = crate::gateway::fsdecode_filename_bytes(
                                std::ffi::CStr::from_ptr((*p).if_name).to_bytes(),
                            );
                            items.push(pyre_object::w_tuple_new(vec![
                                pyre_object::w_int_new((*p).if_index as i64),
                                name,
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
        crate::module_ns_store(
            ns,
            "if_nametoindex",
            crate::make_builtin_function_with_arity(
                "if_nametoindex",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error(
                            "if_nametoindex() requires 1 argument",
                        ));
                    }
                    // An interface name is an OS string.
                    // `interp_socket.py` declares `name='text'` and
                    // compares against `rsocket.if_nameindex()`'s own names;
                    // `socketmodule.c socket_if_nametoindex` instead reads it
                    // with `PyUnicode_FSConverter`, the filesystem encoding
                    // `fsencode_w` applies, and accepts the same
                    // str / bytes / `__fspath__` argument.  Take the 3.14
                    // spelling, which is also what makes this round-trip with
                    // the `if_nameindex` / `if_indextoname` decode above.
                    let name = crate::gateway::fsencode_bytes_w(args[0])?;
                    let c_name = std::ffi::CString::new(name)
                        .map_err(|_| crate::PyError::value_error("embedded null in name"))?;
                    let idx = unsafe { libc::if_nametoindex(c_name.as_ptr()) };
                    if idx == 0 {
                        return Err(socket_last_error());
                    }
                    Ok(pyre_object::w_int_new(idx as i64))
                },
                1,
            ),
        );
        crate::module_ns_store(
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
                        return Err(socket_last_error());
                    }
                    let s = unsafe { std::ffi::CStr::from_ptr(p) };
                    Ok(crate::gateway::fsdecode_filename_bytes(s.to_bytes()))
                },
                1,
            ),
        );
    }

    // Windows has the same public trio but no POSIX `if_nameindex` array.
    // `rustpython_host_env::socket` follows the same IP Helper API route:
    // `GetIfTable2Ex` for enumeration and the WinSock conversion calls for
    // the two scalar directions.
    #[cfg(all(windows, feature = "host_env"))]
    {
        crate::module_ns_store(
            ns,
            "if_nameindex",
            crate::make_builtin_function_with_arity(
                "if_nameindex",
                |_| {
                    let entries = rustpython_host_env::socket::if_nameindex()
                        .map_err(interface_io_error)?;
                    Ok(pyre_object::w_list_new(
                        entries
                            .into_iter()
                            .map(|(index, name)| {
                                pyre_object::w_tuple_new(vec![
                                    pyre_object::w_int_new(index as i64),
                                    pyre_object::w_str_new(&name),
                                ])
                            })
                            .collect(),
                    ))
                },
                0,
            ),
        );
        crate::module_ns_store(
            ns,
            "if_nametoindex",
            crate::make_builtin_function_with_arity(
                "if_nametoindex",
                |args| {
                    let name = crate::gateway::fsencode_bytes_w(args[0])?;
                    let name = std::ffi::CString::new(name)
                        .map_err(|_| crate::PyError::value_error("embedded null in name"))?;
                    let index = rustpython_host_env::socket::if_nametoindex_checked(&name)
                        .map_err(interface_io_error)?;
                    Ok(pyre_object::w_int_new(index as i64))
                },
                1,
            ),
        );
        crate::module_ns_store(
            ns,
            "if_indextoname",
            crate::make_builtin_function_with_arity(
                "if_indextoname",
                |args| {
                    // `socket_if_indextoname`'s NET_IFINDEX converter reads
                    // `__index__` first, rejects a negative value, and only
                    // then the ones no interface index can hold.  An argument
                    // wider than a machine word has to reach those same two
                    // answers rather than a conversion error of its own, so
                    // take its sign from the object when the word conversion
                    // is the thing that fails.
                    let w_index = crate::baseobjspace::space_index(args[0])?;
                    let word = crate::builtins::space_index_w(w_index).ok();
                    let negative = match word {
                        Some(index) => index < 0,
                        // Only a long fails to fit a machine word.
                        None => unsafe {
                            pyre_object::longobject::jit_bigint_sign_i64(
                                pyre_object::longobject::w_long_get_value(w_index),
                            ) < 0
                        },
                    };
                    if negative {
                        return Err(crate::PyError::value_error("Cannot convert negative int"));
                    }
                    let index = word
                        .and_then(|index| u32::try_from(index).ok())
                        .ok_or_else(|| {
                            crate::PyError::overflow_error(
                                "Python int too large for C NET_IFINDEX",
                            )
                        })?;
                    let name = rustpython_host_env::socket::if_indextoname_checked(index)
                        .map_err(interface_io_error)?;
                    Ok(pyre_object::w_str_new(&name))
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
        crate::module_ns_store(
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
        crate::module_ns_store(
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
    // `interp_func.py` (getaddrinfo) and `:137-156`
    // (getnameinfo) — directly wrap the host's getaddrinfo / getnameinfo
    // and walk the addrinfo linked list.
    #[cfg(any(unix, windows))]
    init_socket_getaddrinfo(ns);

    // ── socket class (slice S2) ──
    #[cfg(any(unix, windows))]
    {
        let socket_tp = socket_type();
        // Expose the type itself as `socket` AND `SocketType` so the
        // stdlib's `class socket(_socket.socket):` pattern works.
        crate::module_ns_store(ns, "socket", socket_tp);
        crate::module_ns_store(ns, "SocketType", socket_tp);
    }

    // `socket.py` defines `socketpair` only when `_socket` carries one and
    // falls back to `_fallback_socketpair`'s own AF_INET pair otherwise, and
    // `fromfd` is built out of `dup` at app level everywhere else.  Both stay
    // with the POSIX calls they are made of; `dup` itself is registered for
    // Windows too, further down, out of the WinSock calls that stand in for
    // it.
    #[cfg(unix)]
    {
        // socketpair(family=AF_UNIX, type=SOCK_STREAM, proto=0)
        crate::module_ns_store(
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
                    rffi::SOCK_STREAM
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
                    return Err(socket_last_error());
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
        crate::module_ns_store(
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
                        return Err(socket_last_error());
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
        crate::module_ns_store(
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
                    return Err(socket_last_error());
                }
                unsafe {
                    libc::fcntl(new_fd, libc::F_SETFD, libc::FD_CLOEXEC);
                }
                Ok(socket_from_fd(new_fd, family, ty, proto))
            }),
        );
    }

    // `socket.py`'s `socket.dup` and `fromfd` both go through
    // `_socket.dup`, which on Windows cannot duplicate a descriptor in place:
    // a socket is not a C runtime file descriptor there.  `socket_dup` hands
    // the socket to the process it is already in with `WSADuplicateSocketW`
    // and re-opens it from the protocol info that writes, which is the pair of
    // calls `share_socket` / `socket_from_share_data` make.  The new socket is
    // non-inheritable, as PEP 446 asks and as `socket_from_share_data` leaves
    // it.
    #[cfg(all(windows, feature = "host_env"))]
    crate::module_ns_store(
        ns,
        "dup",
        crate::make_builtin_function_with_arity(
            "dup",
            |args| {
                let fd = crate::builtins::space_index_w(args[0])?;
                let share =
                    rustpython_host_env::socket::share_socket(fd as _, std::process::id())
                        .map_err(socket_io_err)?;
                let shared = rustpython_host_env::socket::socket_from_share_data(&share)
                    .map_err(socket_io_err)?;
                Ok(pyre_object::w_int_new(shared.raw as i64))
            },
            1,
        ),
    );
}

// ── hostent → (name, aliases, addrs) ──
// `interp_func.py common_wrapgethost` — packs a resolver hostent
// into the 3-tuple shape used by gethostbyname_ex / gethostbyaddr.
#[cfg(any(unix, windows))]
fn unpack_hostent(he: *mut rffi::Hostent) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    unsafe {
        let host_name = rffi::hostent_name(he);
        let name = if host_name.is_null() {
            String::new()
        } else {
            std::ffi::CStr::from_ptr(host_name)
                .to_string_lossy()
                .into_owned()
        };
        // Copy all resolver-owned storage while the process-global netdb lock
        // is held, before allocating any Python objects.
        let aliases = rffi::hostent_aliases(he);
        let mut alias_strings = Vec::new();
        if !aliases.is_null() {
            let mut index = 0;
            loop {
                let alias = rffi::pointer_at(aliases, index);
                if alias.is_null() {
                    break;
                }
                alias_strings.push(std::ffi::CStr::from_ptr(alias).to_string_lossy().into_owned());
                index += 1;
            }
        }
        let addr_list = rffi::hostent_addr_list(he);
        let addr_type = rffi::hostent_addr_type(he);
        let addr_length = rffi::hostent_length(he);
        let mut addr_strings = Vec::new();
        if !addr_list.is_null() {
            let mut index = 0;
            loop {
                let addr_ptr = rffi::pointer_at(addr_list, index);
                if addr_ptr.is_null() {
                    break;
                }
                let addr_str = if addr_type == rffi::AF_INET && addr_length == 4 {
                    let packed = std::ptr::read_unaligned(addr_ptr as *const u32).to_ne_bytes();
                    rffi::inet_ntoa(packed).unwrap_or_default()
                } else if addr_type == rffi::AF_INET6 && addr_length == 16 {
                    let mut packed_addr = [0u8; 16];
                    std::ptr::copy_nonoverlapping(
                        addr_ptr as *const u8,
                        packed_addr.as_mut_ptr(),
                        packed_addr.len(),
                    );
                    let mut buf = [0u8; 64];
                    let q = rffi::inet_ntop(
                        rffi::AF_INET6,
                        packed_addr.as_ptr() as *const libc::c_void,
                        buf.as_mut_ptr() as *mut libc::c_char,
                        buf.len() as rffi::SockLen,
                    );
                    if q.is_null() {
                        String::new()
                    } else {
                        std::ffi::CStr::from_ptr(q).to_string_lossy().into_owned()
                    }
                } else {
                    String::new()
                };
                addr_strings.push(addr_str);
                index += 1;
            }
        }
        let aliases = alias_strings
            .iter()
            .map(|alias| pyre_object::w_str_new(alias))
            .collect();
        let addrs = addr_strings
            .iter()
            .map(|addr| pyre_object::w_str_new(addr))
            .collect();
        // `w_list_new` roots the items it is handed, not the header it returns,
        // and that header is a movable nursery object (`rlist.py:116 LIST =
        // GcStruct`). Each list therefore stays pinned across the allocation
        // that follows it and is re-read from its slot afterwards.
        let _roots = pyre_object::gc_roots::push_roots();
        let aliases_slot = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(pyre_object::w_list_new(aliases));
        let addrs_slot = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(pyre_object::w_list_new(addrs));
        Ok(pyre_object::w_tuple_new(vec![
            pyre_object::w_str_new(&name),
            pyre_object::gc_roots::shadow_stack_get(aliases_slot),
            pyre_object::gc_roots::shadow_stack_get(addrs_slot),
        ]))
    }
}

// ── default socket timeout cell ──
// `rsocket.py:setdefaulttimeout|getdefaulttimeout` — process-wide
// default for socket() construction.  None == blocking; Some(secs)
// == timeout in seconds.

// Process-global default socket timeout, encoded as f64 bits.  A valid
// timeout is always a finite non-negative float, so the qNaN sentinel
// below can never collide with a real value and stands in for `None`.
const SOCKET_TIMEOUT_NONE: u64 = 0x7ff8_0000_0000_0001;
static DEFAULT_SOCKET_TIMEOUT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(SOCKET_TIMEOUT_NONE);

fn get_default_socket_timeout() -> pyre_object::PyObjectRef {
    let bits = DEFAULT_SOCKET_TIMEOUT.load(std::sync::atomic::Ordering::Relaxed);
    if bits == SOCKET_TIMEOUT_NONE {
        pyre_object::w_none()
    } else {
        pyre_object::floatobject::w_float_new(f64::from_bits(bits))
    }
}

fn set_default_socket_timeout(v: Option<f64>) {
    let bits = match v {
        None => SOCKET_TIMEOUT_NONE,
        Some(s) => {
            let b = s.to_bits();
            debug_assert_ne!(b, SOCKET_TIMEOUT_NONE);
            b
        }
    };
    DEFAULT_SOCKET_TIMEOUT.store(bits, std::sync::atomic::Ordering::Relaxed);
}

// ── getaddrinfo / getnameinfo wiring ──
//
// PyPy's `interp_func.py:294-339` walks libc's `addrinfo` linked
// list and packs each entry into a 5-tuple `(family, socktype,
// proto, canonname, sockaddr)`.  `getnameinfo` is the symmetric
// path used by stdlib socket.getnameinfo.

#[cfg(any(unix, windows))]
fn init_socket_getaddrinfo(ns: pyre_object::PyObjectRef) {
    crate::module_ns_store(
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
                } else if pyre_object::bytesobject::is_bytes(host_obj) {
                    Some(
                        std::ffi::CString::new(
                            pyre_object::bytesobject::bytes_like_data(host_obj),
                        )
                        .map_err(|_| crate::PyError::value_error("embedded null in host"))?,
                    )
                } else if pyre_object::is_str(host_obj) {
                    let s = crate::baseobjspace::str_utf8_w(host_obj)?.to_string();
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
                } else if pyre_object::bytesobject::is_bytes(port_obj) {
                    Some(
                        std::ffi::CString::new(
                            pyre_object::bytesobject::bytes_like_data(port_obj),
                        )
                        .map_err(|_| crate::PyError::value_error("embedded null in port"))?,
                    )
                } else if pyre_object::is_str(port_obj) {
                    let s = crate::baseobjspace::str_utf8_w(port_obj)?.to_string();
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
            let family = int_arg(2, rffi::AF_UNSPEC)?;
            let socktype = int_arg(3, 0)?;
            let proto = int_arg(4, 0)?;
            let flags = int_arg(5, 0)?;

            let mut hints: rffi::addrinfo = unsafe { std::mem::zeroed() };
            hints.ai_family = family;
            hints.ai_socktype = socktype;
            hints.ai_protocol = proto;
            hints.ai_flags = flags;

            let mut res: *mut rffi::addrinfo = std::ptr::null_mut();
            let host_ptr = host
                .as_ref()
                .map(|c| c.as_ptr())
                .unwrap_or(std::ptr::null());
            let port_ptr = port
                .as_ref()
                .map(|c| c.as_ptr())
                .unwrap_or(std::ptr::null());
            // A name lookup goes to the resolver and can take seconds.
            let rc = {
                let _blocked = crate::module::thread::before_external_block();
                unsafe { rffi::getaddrinfo(host_ptr, port_ptr, &hints, &mut res) }
            };
            if rc != 0 {
                let msg = rffi::gai_strerror(rc);
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
                        std::ffi::CStr::from_ptr(ai.ai_canonname.cast())
                            .to_string_lossy()
                            .into_owned()
                    };
                    // Copy sockaddr into our sockaddr_storage so we can
                    // reuse unpack_inet_addr.
                    let mut storage: rffi::sockaddr_storage = std::mem::zeroed();
                    let copy_len = (ai.ai_addrlen as usize)
                        .min(core::mem::size_of::<rffi::sockaddr_storage>());
                    std::ptr::copy_nonoverlapping(
                        ai.ai_addr as *const u8,
                        &mut storage as *mut _ as *mut u8,
                        copy_len,
                    );
                    let addr = unpack_inet_addr(&storage, copy_len as rffi::SockLen);
                    items.push(pyre_object::w_tuple_new(vec![
                        pyre_object::w_int_new(ai.ai_family as i64),
                        pyre_object::w_int_new(ai.ai_socktype as i64),
                        pyre_object::w_int_new(ai.ai_protocol as i64),
                        pyre_object::w_str_new(&canon),
                        addr,
                    ]));
                    cur = ai.ai_next;
                }
                rffi::freeaddrinfo(res);
            }
            Ok(pyre_object::w_list_new(items))
        }),
    );

    crate::module_ns_store(
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
                let host = crate::baseobjspace::str_utf8_w(host_obj)?.to_string();
                let port_v = unsafe { pyre_object::w_int_get_value(port_obj) };

                let c_host = std::ffi::CString::new(host.as_bytes())
                    .map_err(|_| crate::PyError::value_error("embedded null in host"))?;
                let c_port = std::ffi::CString::new(format!("{port_v}")).unwrap();

                let mut hints: rffi::addrinfo = unsafe { std::mem::zeroed() };
                hints.ai_family = rffi::AF_UNSPEC;
                hints.ai_socktype = rffi::SOCK_DGRAM;
                hints.ai_flags = rffi::AI_NUMERICHOST;
                let mut res: *mut rffi::addrinfo = std::ptr::null_mut();
                let rc = {
                    let _blocked = crate::module::thread::before_external_block();
                    unsafe { rffi::getaddrinfo(c_host.as_ptr(), c_port.as_ptr(), &hints, &mut res) }
                };
                if rc != 0 {
                    let msg = rffi::gai_strerror(rc);
                    return Err(socket_converted_error("gaierror", Some(rc), &msg));
                }
                let head = res;
                let ai = unsafe { &*head };
                if !ai.ai_next.is_null() {
                    unsafe { rffi::freeaddrinfo(head) };
                    return Err(socket_converted_error(
                        "error",
                        None,
                        "sockaddr resolved to multiple addresses",
                    ));
                }
                let mut host_buf = [0 as libc::c_char; rffi::NI_MAXHOST as usize];
                let mut serv_buf = [0 as libc::c_char; 32];
                // A reverse lookup goes to the resolver and can take seconds.
                let nrc = {
                    let _blocked = crate::module::thread::before_external_block();
                    unsafe {
                        rffi::getnameinfo(
                            ai.ai_addr,
                            ai.ai_addrlen as rffi::SockLen,
                            host_buf.as_mut_ptr(),
                            host_buf.len() as rffi::SockLen,
                            serv_buf.as_mut_ptr(),
                            serv_buf.len() as rffi::SockLen,
                            flags,
                        )
                    }
                };
                unsafe { rffi::freeaddrinfo(head) };
                if nrc != 0 {
                    let msg = rffi::gai_strerror(nrc);
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

#[cfg(any(unix, windows))]
fn socket_type() -> pyre_object::PyObjectRef {
    // Process-global immortal type object (see `make_builtin_type`).
    static SOCKET_TYPE_OBJ: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *SOCKET_TYPE_OBJ.get_or_init(|| {
        let tp = crate::typedef::make_builtin_type("socket", init_socket_type);
        unsafe { pyre_object::typeobject::w_type_set_hasdict(tp, true) };
        unsafe { pyre_object::w_type_set_hasuserdel(tp, true) };
        tp as usize
    }) as pyre_object::PyObjectRef
}

#[cfg(any(unix, windows))]
fn socket_io_err(e: std::io::Error) -> crate::PyError {
    let code = e.raw_os_error().unwrap_or(0);
    // `socketmodule.c set_error` raises a WinSock failure through
    // `PyErr_SetExcFromWindowsErr`: the code is a Win32 one, so it belongs in
    // `.winerror` with `.errno` derived from it, not read as an errno itself.
    #[cfg(windows)]
    {
        crate::PyError::os_error_win32_syscall2(code, pyre_object::PY_NULL, pyre_object::PY_NULL)
    }
    // `rsocket.py` carries the C `strerror` text; build the OSError in
    // its `(errno, strerror)` form so `e.errno` and `str(e)` match.
    #[cfg(not(windows))]
    {
        let strerror = unsafe {
            let p = libc::strerror(code);
            if p.is_null() {
                format!("Unknown error {code}")
            } else {
                std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        };
        crate::PyError::os_error_errno_strerror(code, strerror)
    }
}

/// The exception for a socket call that has just failed, read from wherever
/// the host records it.
#[cfg(any(unix, windows))]
fn socket_last_error() -> crate::PyError {
    socket_io_err(rffi::last_error())
}

/// `call_external_function` with the failure code the socket API reports.
/// That helper reads the C runtime's `errno`, which WinSock never writes —
/// it keeps its own last-error slot — so the code has to come from `rffi`.
#[cfg(any(unix, windows))]
fn socket_call<R>(f: impl FnOnce() -> R) -> (R, i32) {
    #[cfg(windows)]
    {
        let _blocked = crate::module::thread::before_external_block();
        let result = f();
        (result, rffi::last_error_code())
    }
    #[cfg(not(windows))]
    {
        crate::module::thread::call_external_function(f)
    }
}

#[cfg(any(unix, windows))]
fn socket_io_err_for_operation(
    obj: pyre_object::PyObjectRef,
    e: std::io::Error,
) -> crate::PyError {
    let errno = e.raw_os_error().unwrap_or(0);
    if rffi::error_is_would_block(errno) && socket_positive_timeout(obj).is_some() {
        // RPython's RSocket._select() turns expiry of a positive timeout
        // into SocketTimeout, which interp_socket maps to TimeoutError.
        // This backend uses SO_RCVTIMEO/SO_SNDTIMEO, whose equivalent
        // expiry signal is EAGAIN/EWOULDBLOCK.
        return socket_converted_error("timeout", None, "timed out");
    }
    socket_io_err(e)
}

/// The socket's timeout when it is a positive number of seconds — the state in
/// which a call has to be bounded rather than left to block.  `None` covers
/// both of the other two: blocking forever, and non-blocking.
#[cfg(any(unix, windows))]
fn socket_positive_timeout(obj: pyre_object::PyObjectRef) -> Option<f64> {
    let dict = crate::baseobjspace::getdict_native(obj);
    (!dict.is_null())
        .then(|| unsafe { pyre_object::w_dict_getitem_str(dict, "_timeout") })
        .flatten()
        .filter(|timeout| unsafe { pyre_object::is_float(*timeout) })
        .map(|timeout| unsafe { pyre_object::floatobject::w_float_get_value(timeout) })
        .filter(|timeout| *timeout > 0.0)
}

/// RPython `RSocket._select(False)` used by `RSocket.accept`: a positive
/// Python timeout is enforced with poll before entering the libc operation.
/// Darwin does not reliably apply SO_RCVTIMEO to accept(), so relying on that
/// socket option leaves timeout-driven server loops blocked indefinitely.
#[cfg(any(unix, windows))]
fn socket_wait_readable(
    obj: pyre_object::PyObjectRef,
    fd: rffi::Socket,
) -> Result<(), crate::PyError> {
    let Some(timeout) = socket_positive_timeout(obj) else {
        return Ok(());
    };
    // A handled signal restarts the wait, so the deadline is computed once:
    // reusing the full duration on every EINTR would let a steady signal
    // stream extend a finite timeout without bound.  poll carries its timeout
    // in milliseconds as a c_int, which bounds how long a single wait can ask
    // for however long the caller requested.
    let capped = timeout.min(i32::MAX as f64 / 1000.0);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs_f64(capped);
    loop {
        let remaining = deadline.saturating_duration_since(std::time::Instant::now());
        if remaining.is_zero() {
            return Err(socket_converted_error("timeout", None, "timed out"));
        }
        // poll's resolution is one millisecond; a shorter remainder must still
        // wait rather than degenerate into a busy loop.
        let timeout_ms = remaining.as_millis().max(1).min(i32::MAX as u128) as i32;
        let (ready, errno) = rffi::poll_readable(fd, timeout_ms);
        if ready > 0 {
            return Ok(());
        }
        if ready == 0 {
            return Err(socket_converted_error("timeout", None, "timed out"));
        }
        if !rffi::error_is_interrupted(errno) {
            return Err(socket_io_err(std::io::Error::from_raw_os_error(errno)));
        }
        crate::module::signal::interp_signal::checksignals_now()?;
    }
}

/// `internal_connect` for a socket carrying a positive timeout.  WinSock does
/// not apply `SO_SNDTIMEO` to `connect`, so the call runs in non-blocking mode
/// and the wait for its outcome is explicit; POSIX's `SO_SNDTIMEO` already
/// bounds the call and keeps the straight-line path.
///
/// Returns the code the attempt reported, `WSAETIMEDOUT` on expiry, so both
/// `connect` and `connect_ex` can shape it the way each one answers.
#[cfg(windows)]
fn socket_connect_timed(
    fd: rffi::Socket,
    storage: &rffi::sockaddr_storage,
    slen: rffi::SockLen,
    timeout: f64,
) -> Result<(), i32> {
    rffi::apply_timeout(fd, 0.0).map_err(|e| e.raw_os_error().unwrap_or(0))?;
    let outcome = socket_connect_wait(fd, storage, slen, timeout);
    // The socket goes back to the mode its own timeout declares whatever the
    // connect did; failing to restore it is not the answer the caller asked
    // for.
    let _ = rffi::apply_timeout(fd, timeout);
    outcome
}

#[cfg(windows)]
fn socket_connect_wait(
    fd: rffi::Socket,
    storage: &rffi::sockaddr_storage,
    slen: rffi::SockLen,
    timeout: f64,
) -> Result<(), i32> {
    let (started, errno) = socket_call(|| unsafe {
        rffi::connect(fd, storage as *const _ as *const rffi::sockaddr, slen)
    });
    if started == 0 {
        return Ok(());
    }
    if !rffi::error_is_would_block(errno) {
        return Err(errno);
    }
    // The connection is under way and reports its outcome by making the socket
    // writable.  A single wait is enough: Windows does not interrupt it.
    let timeout_ms = (timeout * 1000.0).round().clamp(1.0, i32::MAX as f64) as i32;
    let (ready, poll_errno) = rffi::poll_writable(fd, timeout_ms);
    if ready < 0 {
        return Err(poll_errno);
    }
    if ready == 0 {
        return Err(rffi::ETIMEDOUT);
    }
    // `SO_ERROR` carries the outcome of a connect that finished after its call
    // returned; zero means it succeeded.
    match socket_getsockopt_int(fd, rffi::SOL_SOCKET, rffi::SO_ERROR) {
        Ok(0) => Ok(()),
        Ok(errno) => Err(errno),
        // Reading the option failed; the last-error slot still holds why.
        Err(_) => Err(rffi::last_error_code()),
    }
}

#[cfg(any(unix, windows))]
fn socket_get_attr_i64(obj: pyre_object::PyObjectRef, key: &str) -> i64 {
    let d = crate::baseobjspace::getdict_native(obj);
    if d.is_null() {
        return -1;
    }
    if let Some(v) = unsafe { pyre_object::w_dict_getitem_str(d, key) }
        && unsafe { pyre_object::is_int(v) } {
            return unsafe { pyre_object::w_int_get_value(v) };
        }
    -1
}

#[cfg(any(unix, windows))]
fn socket_set_attr(obj: pyre_object::PyObjectRef, key: &str, v: pyre_object::PyObjectRef) {
    let d = crate::baseobjspace::getdict_native(obj);
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
#[cfg(any(unix, windows))]
fn socket_apply_timeout(fd: rffi::Socket, timeout: f64) -> Result<(), crate::PyError> {
    rffi::apply_timeout(fd, timeout).map_err(socket_io_err)
}

#[cfg(any(unix, windows))]
fn socket_fd(obj: pyre_object::PyObjectRef) -> Result<rffi::Socket, crate::PyError> {
    let fd = rffi::socket_from_i64(socket_get_attr_i64(obj, "_fd"));
    if rffi::is_invalid(fd) {
        return Err(crate::PyError::os_error("Bad file descriptor"));
    }
    Ok(fd)
}

#[cfg(any(unix, windows))]
fn socket_from_fd(
    fd: rffi::Socket,
    family: libc::c_int,
    ty: libc::c_int,
    proto: libc::c_int,
) -> pyre_object::PyObjectRef {
    socket_from_fd_with_class(fd, family, ty, proto, socket_type())
}

#[cfg(any(unix, windows))]
fn socket_from_fd_with_class(
    fd: rffi::Socket,
    family: libc::c_int,
    ty: libc::c_int,
    proto: libc::c_int,
    cls: pyre_object::PyObjectRef,
) -> pyre_object::PyObjectRef {
    // PyPy's typeobject.py allocate_instance preserves the requested
    // W_TypeObject when W_Socket.descr_init initialises a user subclass.
    // The stdlib relies on this for `class socket(_socket.socket)`: losing
    // `cls` here also loses its Python-level makefile() and lifecycle state.
    let obj = pyre_object::w_instance_new(cls);
    socket_init_state(obj, fd, family, ty, proto);
    obj
}

#[cfg(any(unix, windows))]
fn socket_init_state(
    obj: pyre_object::PyObjectRef,
    fd: rffi::Socket,
    family: libc::c_int,
    ty: libc::c_int,
    proto: libc::c_int,
) {
    socket_set_attr(obj, "_fd", pyre_object::w_int_new(rffi::socket_to_i64(fd)));
    socket_set_attr(obj, "_family", pyre_object::w_int_new(family as i64));
    socket_set_attr(obj, "_type", pyre_object::w_int_new(ty as i64));
    socket_set_attr(obj, "_proto", pyre_object::w_int_new(proto as i64));
    socket_set_attr(obj, "_timeout", pyre_object::w_none());
    // `sock_new` starts this at 0 on every socket object, and only
    // `setsockopt` moves it: `SIO_TCP_SET_ACK_FREQUENCY` has no counterpart to
    // read the live setting back with, so what was written is the answer.
    #[cfg(windows)]
    socket_set_attr(obj, "_quickack", pyre_object::w_int_new(0));
    // `interp_socket.py usecount = 1` — start the refcount at 1 so
    // `_drop` followed by no `_reuse` closes the underlying fd exactly
    // once.
    socket_set_attr(obj, "_usecount", pyre_object::w_int_new(1));
}

/// `rsocket.py get_socket_family` — the family of an existing fd,
/// read from `getsockname`'s returned `sa_family`.
#[cfg(any(unix, windows))]
fn socket_detect_family(fd: rffi::Socket) -> Result<libc::c_int, crate::PyError> {
    let mut addr: rffi::sockaddr_storage = unsafe { std::mem::zeroed() };
    let mut len = std::mem::size_of::<rffi::sockaddr_storage>() as rffi::SockLen;
    let res = unsafe { rffi::getsockname(fd, &mut addr as *mut _ as *mut rffi::sockaddr, &mut len) };
    if res != 0 {
        return Err(socket_last_error());
    }
    Ok(addr.ss_family as libc::c_int)
}

/// `rsocket.py getsockopt_int` — a single int socket option.
#[cfg(any(unix, windows))]
fn socket_getsockopt_int(
    fd: rffi::Socket,
    level: libc::c_int,
    option: libc::c_int,
) -> Result<libc::c_int, crate::PyError> {
    let mut val: libc::c_int = 0;
    let mut len = std::mem::size_of::<libc::c_int>() as rffi::SockLen;
    let res = unsafe {
        rffi::getsockopt(
            fd,
            level,
            option,
            &mut val as *mut _ as *mut libc::c_void,
            &mut len,
        )
    };
    if res != 0 {
        return Err(socket_last_error());
    }
    Ok(val)
}

/// `interp_socket.py get_so_protocol` — the protocol of an existing
/// fd via `SO_PROTOCOL`, or `-1` on platforms without it (`HAS_SO_PROTOCOL`).
#[cfg(any(target_os = "linux", target_os = "android"))]
fn socket_get_so_protocol(fd: rffi::Socket) -> Result<libc::c_int, crate::PyError> {
    socket_getsockopt_int(fd, rffi::SOL_SOCKET, libc::SO_PROTOCOL)
}
#[cfg(all(any(unix, windows), not(any(target_os = "linux", target_os = "android"))))]
fn socket_get_so_protocol(_fd: rffi::Socket) -> Result<libc::c_int, crate::PyError> {
    Ok(-1)
}

// ── address pack/unpack helpers ──
//
// Python passes IPv4 addresses as (host, port) tuples and IPv6 as
// (host, port, flowinfo, scopeid).  These helpers convert to/from
// `sockaddr_storage`.

/// `offsetof(_c.sockaddr_un, 'c_sun_path')` (`rsocket.py:403 minlen`) — where
/// the name starts inside the address, and therefore the part of an address
/// length that is not name.
#[cfg(unix)]
const SUN_PATH_OFFSET: usize = core::mem::offset_of!(libc::sockaddr_un, sun_path);

/// One `%X` conversion of the scan `setbdaddr` runs: blanks, an optional sign,
/// an optional `0x` prefix, then hex digits accumulated into the `unsigned
/// int` the C code declares.  Answers the value and what is left to read.
#[cfg(windows)]
fn scan_hex_field(text: &str) -> Option<(u32, &str)> {
    let text = text.trim_start_matches([' ', '\t', '\n', '\r', '\u{b}', '\u{c}']);
    let (negative, text) = match text.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, text.strip_prefix('+').unwrap_or(text)),
    };
    // A `0x` counts as a prefix only when a hex digit follows: otherwise the
    // conversion stops after the `0` and leaves the `x` to be read.
    let digits = match text.strip_prefix("0x").or_else(|| text.strip_prefix("0X")) {
        Some(rest) if rest.starts_with(|c: char| c.is_ascii_hexdigit()) => rest,
        _ => text,
    };
    let mut value: u32 = 0;
    let mut end = 0;
    for digit in digits.bytes().take_while(u8::is_ascii_hexdigit) {
        let digit = char::from(digit).to_digit(16).expect("hex digit");
        value = value.wrapping_mul(16).wrapping_add(digit);
        end += 1;
    }
    if end == 0 {
        return None;
    }
    Some((if negative { value.wrapping_neg() } else { value }, &digits[end..]))
}

/// `setbdaddr` — six `%X` fields with a colon between each pair, every one of
/// them below 256, and a trailing `%c` that makes anything after the sixth a
/// seventh conversion and so a failure.  The leading field is the most
/// significant octet, and `BTH_ADDR` carries the six as one number.
#[cfg(windows)]
fn parse_bdaddr(name: &str) -> Option<u64> {
    let mut rest = name;
    let mut octets = [0u32; 6];
    for (i, octet) in octets.iter_mut().enumerate() {
        if i > 0 {
            rest = rest.strip_prefix(':')?;
        }
        (*octet, rest) = scan_hex_field(rest)?;
    }
    if !rest.is_empty() || octets.iter().fold(0, |acc, octet| acc | octet) >= 256 {
        return None;
    }
    Some(octets.iter().fold(0u64, |acc, &octet| (acc << 8) | u64::from(octet)))
}

/// `makebdaddr` — `XX:XX:XX:XX:XX:XX`, most significant octet first.
#[cfg(windows)]
fn bdaddr_string(bdaddr: u64) -> String {
    let octet = |i: u32| (bdaddr >> (8 * i)) & 0xFF;
    format!(
        "{:02X}:{:02X}:{:02X}:{:02X}:{:02X}:{:02X}",
        octet(5),
        octet(4),
        octet(3),
        octet(2),
        octet(1),
        octet(0)
    )
}

/// The `AF_BLUETOOTH` case of `getsockaddrarg` (`socketmodule.c:2104-2205`).
/// Only RFCOMM reaches here: it is the one Bluetooth protocol Windows carries,
/// and `SOCKADDR_BTH` is the one address form that goes with it.
#[cfg(windows)]
fn pack_bluetooth_addr(
    caller: &str,
    proto: libc::c_int,
    addr: pyre_object::PyObjectRef,
    storage: &mut rffi::sockaddr_storage,
) -> Result<rffi::SockLen, crate::PyError> {
    use windows_sys::Win32::Devices::Bluetooth as bt;

    if proto != bt::BTHPROTO_RFCOMM as libc::c_int {
        return Err(crate::PyError::os_error(format!(
            "{caller}(): unknown Bluetooth protocol"
        )));
    }
    // `PyArg_ParseTuple(args, "sk")` fails and is answered with this one
    // message, so every shape that is not a `str` beside an integer channel —
    // a wrong length, `bytes`, an embedded NUL, a lone surrogate — reads alike.
    let wrong_format = || crate::PyError::os_error(format!("{caller}(): wrong format"));
    if !unsafe { pyre_object::is_tuple(addr) } || unsafe { pyre_object::w_tuple_len(addr) } != 2 {
        return Err(wrong_format());
    }
    let w_name = unsafe { pyre_object::w_tuple_getitem(addr, 0) }.expect("length checked above");
    let w_channel = unsafe { pyre_object::w_tuple_getitem(addr, 1) }.expect("length checked above");
    if !unsafe { pyre_object::is_str(w_name) } || !unsafe { pyre_object::is_int(w_channel) } {
        return Err(wrong_format());
    }
    let name = crate::baseobjspace::str_utf8_w(w_name).map_err(|_| wrong_format())?;
    if name.contains('\0') {
        return Err(wrong_format());
    }
    let bd_addr = parse_bdaddr(name).ok_or_else(|| crate::PyError::os_error("bad bluetooth address"))?;
    let bth = bt::SOCKADDR_BTH {
        addressFamily: bt::AF_BTH,
        btAddr: bd_addr,
        serviceClassId: Default::default(),
        // `k` keeps the low `ULONG` of whatever it is handed rather than
        // reporting an overflow.
        port: (unsafe { pyre_object::w_int_get_value(w_channel) }) as u32,
    };
    // `SOCKADDR_BTH` is packed, so its `BTH_ADDR` sits two bytes into the
    // storage and no aligned write reaches it.
    unsafe { core::ptr::write_unaligned(storage as *mut _ as *mut bt::SOCKADDR_BTH, bth) };
    Ok(core::mem::size_of::<bt::SOCKADDR_BTH>() as rffi::SockLen)
}

/// The RFCOMM case of `makesockaddr` (`socketmodule.c:1546-1560`) — the address
/// as a string beside the channel, the shape `bind` and `connect` take back.
#[cfg(windows)]
fn unpack_bluetooth_addr(storage: &rffi::sockaddr_storage) -> pyre_object::PyObjectRef {
    use windows_sys::Win32::Devices::Bluetooth as bt;

    let bth: bt::SOCKADDR_BTH =
        unsafe { core::ptr::read_unaligned(storage as *const _ as *const bt::SOCKADDR_BTH) };
    pyre_object::w_tuple_new(vec![
        pyre_object::w_str_new(&bdaddr_string(bth.btAddr)),
        pyre_object::w_int_new(i64::from(bth.port)),
    ])
}

/// `HV_PROTOCOL_RAW` — the only protocol an `AF_HYPERV` socket is opened with.
/// `hvsocket.h` defines it; `windows-sys` does not carry that header.
#[cfg(windows)]
const HV_PROTOCOL_RAW: libc::c_int = 1;

/// `SOCKADDR_HV` (`hvsocket.h`): the family, a reserved word, and the two
/// GUIDs a Hyper-V endpoint is named by.  36 bytes, so it fits a
/// `sockaddr_storage` like every other form here.
#[cfg(windows)]
#[repr(C)]
#[derive(Clone, Copy)]
struct SockaddrHv {
    family: u16,
    reserved: u16,
    vm_id: windows_sys::core::GUID,
    service_id: windows_sys::core::GUID,
}

/// The single message `PyArg_ParseTuple(args, "UU;...")` produces for every
/// shape that is a tuple but not two `str`s — a wrong length included.
#[cfg(windows)]
fn hyperv_address_shape_error() -> crate::PyError {
    crate::PyError::type_error("AF_HYPERV address must be a str tuple (vm_id, service_id)")
}

/// `UuidFromStringW`, which takes the bare 8-4-4-4-12 spelling and nothing
/// else: a braced or truncated GUID is rejected here rather than reinterpreted.
#[cfg(windows)]
fn parse_hyperv_guid(
    caller: &str,
    field: &str,
    w_text: pyre_object::PyObjectRef,
) -> Result<windows_sys::core::GUID, crate::PyError> {
    let mut wide: Vec<u16> = unsafe { pyre_object::w_str_get_wtf8(w_text) }
        .encode_wide()
        .collect();
    wide.push(0);
    let mut guid = windows_sys::core::GUID {
        data1: 0,
        data2: 0,
        data3: 0,
        data4: [0; 8],
    };
    let status =
        unsafe { windows_sys::Win32::System::Rpc::UuidFromStringW(wide.as_ptr(), &mut guid) };
    if status != windows_sys::Win32::System::Rpc::RPC_S_OK {
        return Err(crate::PyError::value_error(format!(
            "{caller}(): AF_HYPERV address {field} is not a valid UUID string"
        )));
    }
    Ok(guid)
}

/// The spelling `UuidToStringW` gives a GUID: lower case, unbraced.
#[cfg(windows)]
fn hyperv_guid_string(guid: &windows_sys::core::GUID) -> String {
    use windows_sys::Win32::System::Rpc::{RPC_S_OK, RpcStringFreeW, UuidToStringW};

    let mut text: *mut u16 = std::ptr::null_mut();
    if unsafe { UuidToStringW(guid, &mut text) } != RPC_S_OK {
        return String::new();
    }
    let mut end = 0usize;
    while unsafe { *text.add(end) } != 0 {
        end += 1;
    }
    let out = String::from_utf16_lossy(unsafe { std::slice::from_raw_parts(text, end) });
    unsafe { RpcStringFreeW(&mut text) };
    out
}

/// The `AF_HYPERV` case of `getsockaddrarg` (`socketmodule.c:2643-2712`).
#[cfg(windows)]
fn pack_hyperv_addr(
    caller: &str,
    proto: libc::c_int,
    addr: pyre_object::PyObjectRef,
    storage: &mut rffi::sockaddr_storage,
) -> Result<rffi::SockLen, crate::PyError> {
    if proto != HV_PROTOCOL_RAW {
        return Err(crate::PyError::os_error(format!(
            "{caller}(): unsupported AF_HYPERV protocol: {proto}"
        )));
    }
    if !unsafe { pyre_object::is_tuple(addr) } {
        return Err(crate::PyError::type_error(format!(
            "{caller}(): AF_HYPERV address must be tuple, not {}",
            crate::type_methods::arg_type_name(addr)
        )));
    }
    if unsafe { pyre_object::w_tuple_len(addr) } != 2 {
        return Err(hyperv_address_shape_error());
    }
    let w_vm_id = unsafe { pyre_object::w_tuple_getitem(addr, 0) }.expect("length checked above");
    let w_service_id =
        unsafe { pyre_object::w_tuple_getitem(addr, 1) }.expect("length checked above");
    if !unsafe { pyre_object::is_str(w_vm_id) } || !unsafe { pyre_object::is_str(w_service_id) } {
        return Err(hyperv_address_shape_error());
    }
    // Both GUIDs are parsed before either is stored: no step here runs Python,
    // so the two lookups above stay valid across the second parse.
    let vm_id = parse_hyperv_guid(caller, "vm_id", w_vm_id)?;
    let service_id = parse_hyperv_guid(caller, "service_id", w_service_id)?;
    let hv = unsafe { &mut *(storage as *mut _ as *mut SockaddrHv) };
    *hv = SockaddrHv {
        family: windows_sys::Win32::Networking::WinSock::AF_HYPERV,
        reserved: 0,
        vm_id,
        service_id,
    };
    Ok(core::mem::size_of::<SockaddrHv>() as rffi::SockLen)
}

/// The `AF_HYPERV` case of `makesockaddr` (`socketmodule.c:1740-1767`) — the
/// two GUIDs as strings, the same shape `bind` and `connect` accept.
#[cfg(windows)]
fn unpack_hyperv_addr(storage: &rffi::sockaddr_storage) -> pyre_object::PyObjectRef {
    let hv = unsafe { &*(storage as *const _ as *const SockaddrHv) };
    let vm_id = hyperv_guid_string(&hv.vm_id);
    let service_id = hyperv_guid_string(&hv.service_id);
    pyre_object::w_tuple_new(vec![
        pyre_object::w_str_new(&vm_id),
        pyre_object::w_str_new(&service_id),
    ])
}

#[cfg(any(unix, windows))]
fn pack_inet_addr(
    caller: &str,
    family: libc::c_int,
    proto: libc::c_int,
    addr: pyre_object::PyObjectRef,
) -> Result<(rffi::sockaddr_storage, rffi::SockLen), crate::PyError> {
    // Only the AF_HYPERV form reads these two: `getsockaddrarg` names the
    // calling method in its messages and rejects any protocol but
    // `HV_PROTOCOL_RAW`.
    #[cfg(not(windows))]
    let _ = (caller, proto);
    let mut storage: rffi::sockaddr_storage = unsafe { std::mem::zeroed() };
    // AF_UNIX is special: rsocket.py:RSocket.bind/connect accept a bare
    // bytes/str path (or a 1-tuple wrapping the path).  Pull the path
    // out before touching tuple[1], which only the AF_INET/AF_INET6
    // forms guarantee.
    #[cfg(unix)]
    if family == libc::AF_UNIX {
        let path_obj = if unsafe { pyre_object::is_tuple(addr) } {
            unsafe { pyre_object::w_tuple_getitem(addr, 0) }
                .ok_or_else(|| crate::PyError::value_error("address: missing path"))?
        } else {
            addr
        };
        // `interp_socket.py w_address = space.fsencode(w_address)`,
        // with the upstream note that it deliberately avoids `fsencode_w`
        // because Linux allows embedded NULs in an abstract-namespace path.
        // The raw `fsencode` preserves that carve-out.
        let path_bytes_vec: Vec<u8> = unsafe {
            if pyre_object::is_str(path_obj) {
                crate::gateway::fsencode(path_obj)?
            } else if pyre_object::bytesobject::is_bytes_like(path_obj) {
                pyre_object::bytesobject::bytes_like_data(path_obj).to_vec()
            } else {
                return Err(crate::PyError::type_error(
                    "AF_UNIX address must be a string or bytes path",
                ));
            }
        };
        let sun = unsafe { &mut *(&mut storage as *mut _ as *mut libc::sockaddr_un) };
        sun.sun_family = libc::AF_UNIX as rffi::SaFamily;
        let abstract_name = cfg!(target_os = "linux") && path_bytes_vec.first() == Some(&0);
        // `rsocket.py UNIXAddress.__init__`: an abstract name may fill
        // `sun_path` exactly, a regular one has to leave room for its
        // terminator.
        let capacity = sun.sun_path.len() - usize::from(!abstract_name);
        if path_bytes_vec.len() > capacity {
            return Err(crate::PyError::os_error("AF_UNIX path too long"));
        }
        for (i, &b) in path_bytes_vec.iter().enumerate() {
            sun.sun_path[i] = b as libc::c_char;
        }
        // `rsocket.py self.setdata(sun, baseofs + len(path))`: the
        // terminator is counted only for a regular name that has one.
        let addrlen = SUN_PATH_OFFSET + path_bytes_vec.len() + usize::from(!abstract_name);
        return Ok((storage, addrlen as rffi::SockLen));
    }

    #[cfg(windows)]
    if family == windows_sys::Win32::Networking::WinSock::AF_HYPERV as libc::c_int {
        let addrlen = pack_hyperv_addr(caller, proto, addr, &mut storage)?;
        return Ok((storage, addrlen));
    }

    #[cfg(windows)]
    if family == windows_sys::Win32::Devices::Bluetooth::AF_BTH as libc::c_int {
        let addrlen = pack_bluetooth_addr(caller, proto, addr, &mut storage)?;
        return Ok((storage, addrlen));
    }

    if !unsafe { pyre_object::is_tuple(addr) } {
        return Err(crate::PyError::type_error(
            "AF_INET address must be a (host, port) tuple",
        ));
    }
    // `getsockaddrarg` parses the AF_INET form with `"O&i"` — exactly two
    // items — and the AF_INET6 form with `"O&i|II"` — two to four.
    let len = unsafe { pyre_object::w_tuple_len(addr) };
    if family == rffi::AF_INET && len != 2 {
        return Err(crate::PyError::type_error(
            "AF_INET address must be a pair (host, port)",
        ));
    }
    if family == rffi::AF_INET6 && !(2..=4).contains(&len) {
        return Err(crate::PyError::type_error(
            "AF_INET6 address must be a tuple (host, port[, flowinfo[, scopeid]])",
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
        crate::baseobjspace::str_utf8_w(host_obj)?.to_string()
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
    if family == rffi::AF_INET {
        let sin = unsafe { &mut *(&mut storage as *mut _ as *mut rffi::sockaddr_in) };
        sin.sin_family = rffi::AF_INET as rffi::SaFamily;
        sin.sin_port = port;
        // inet_pton handles both "0.0.0.0" and dotted-quad.
        let r = if host.is_empty() {
            // RSocket.makeipaddr('', result) uses the wildcard address for
            // bind(), as required by socketserver and socket_helper.bind_port.
            rffi::sockaddr_in_set_addr(sin, rffi::INADDR_ANY);
            1
        } else {
            unsafe {
                rffi::inet_pton(
                    rffi::AF_INET,
                    c_host.as_ptr(),
                    &mut sin.sin_addr as *mut _ as *mut libc::c_void,
                )
            }
        };
        if r != 1 {
            // `rsocket.makeipaddr` resolves names through getaddrinfo and
            // copies the resulting address into its INETAddress.  Do the
            // same here: gethostbyname's process-global result buffer is not
            // re-entrant and was corrupted by socketserver worker threads.
            let mut hints: rffi::addrinfo = unsafe { std::mem::zeroed() };
            hints.ai_family = rffi::AF_INET;
            let mut result: *mut rffi::addrinfo = std::ptr::null_mut();
            let rc = {
                let _blocked = crate::module::thread::before_external_block();
                unsafe { rffi::getaddrinfo(c_host.as_ptr(), std::ptr::null(), &hints, &mut result) }
            };
            if rc != 0 || result.is_null() {
                return Err(crate::PyError::os_error(format!(
                    "name or service not known: {host}"
                )));
            }
            let mut current = result;
            let mut resolved = None;
            while !current.is_null() {
                let info = unsafe { &*current };
                if info.ai_family == rffi::AF_INET
                    && !info.ai_addr.is_null()
                    && info.ai_addrlen as usize >= core::mem::size_of::<rffi::sockaddr_in>()
                {
                    let resolved_addr = unsafe { &*(info.ai_addr as *const rffi::sockaddr_in) };
                    resolved = Some(rffi::sockaddr_in_get_addr(resolved_addr));
                    break;
                }
                current = info.ai_next;
            }
            unsafe { rffi::freeaddrinfo(result) };
            let resolved = resolved.ok_or_else(|| {
                crate::PyError::os_error(format!("name or service not known: {host}"))
            })?;
            rffi::sockaddr_in_set_addr(sin, resolved);
        }
        Ok((
            storage,
            core::mem::size_of::<rffi::sockaddr_in>() as rffi::SockLen,
        ))
    } else if family == rffi::AF_INET6 {
        let sin6 = unsafe { &mut *(&mut storage as *mut _ as *mut rffi::sockaddr_in6) };
        sin6.sin6_family = rffi::AF_INET6 as rffi::SaFamily;
        sin6.sin6_port = port;
        let mut buf = [0u8; 16];
        let r = if host.is_empty() {
            1
        } else {
            unsafe {
                rffi::inet_pton(
                    rffi::AF_INET6,
                    c_host.as_ptr(),
                    buf.as_mut_ptr() as *mut libc::c_void,
                )
            }
        };
        if r != 1 {
            return Err(crate::PyError::os_error(format!(
                "invalid IPv6 address: {host}"
            )));
        }
        rffi::sockaddr_in6_set_addr(sin6, buf);
        if len >= 3
            && let Some(v) = unsafe { pyre_object::w_tuple_getitem(addr, 2) } {
                sin6.sin6_flowinfo = unsafe { pyre_object::w_int_get_value(v) } as u32;
            }
        if len >= 4
            && let Some(v) = unsafe { pyre_object::w_tuple_getitem(addr, 3) } {
                let scope_id = unsafe { pyre_object::w_int_get_value(v) } as u32;
                rffi::sockaddr_in6_set_scope_id(sin6, scope_id);
            }
        Ok((
            storage,
            core::mem::size_of::<rffi::sockaddr_in6>() as rffi::SockLen,
        ))
    } else {
        Err(crate::PyError::os_error(format!(
            "unsupported address family: {family}"
        )))
    }
}

#[cfg(any(unix, windows))]
fn unpack_inet_addr(
    storage: &rffi::sockaddr_storage,
    addrlen: rffi::SockLen,
) -> pyre_object::PyObjectRef {
    let family = storage.ss_family as libc::c_int;
    if family == rffi::AF_INET {
        let sin = unsafe { &*(storage as *const _ as *const rffi::sockaddr_in) };
        let mut buf = [0u8; 64];
        let p = unsafe {
            rffi::inet_ntop(
                rffi::AF_INET,
                &sin.sin_addr as *const _ as *const libc::c_void,
                buf.as_mut_ptr() as *mut libc::c_char,
                buf.len() as rffi::SockLen,
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
    } else if family == rffi::AF_INET6 {
        let sin6 = unsafe { &*(storage as *const _ as *const rffi::sockaddr_in6) };
        let mut buf = [0u8; 64];
        let p = unsafe {
            rffi::inet_ntop(
                rffi::AF_INET6,
                &sin6.sin6_addr as *const _ as *const libc::c_void,
                buf.as_mut_ptr() as *mut libc::c_char,
                buf.len() as rffi::SockLen,
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
            pyre_object::w_int_new(rffi::sockaddr_in6_get_scope_id(sin6) as i64),
        ])
    } else {
        #[cfg(unix)]
        if family == libc::AF_UNIX {
            return unpack_unix_addr(storage, addrlen);
        }
        #[cfg(windows)]
        if family == windows_sys::Win32::Networking::WinSock::AF_HYPERV as libc::c_int {
            return unpack_hyperv_addr(storage);
        }
        #[cfg(windows)]
        if family == windows_sys::Win32::Devices::Bluetooth::AF_BTH as libc::c_int {
            return unpack_bluetooth_addr(storage);
        }
        pyre_object::w_tuple_new(vec![])
    }
}

/// `interp_socket.py:40-47` — the `sockaddr_un` half of `unpack_inet_addr`.
#[cfg(unix)]
fn unpack_unix_addr(
    storage: &rffi::sockaddr_storage,
    addrlen: rffi::SockLen,
) -> pyre_object::PyObjectRef {
    let sun = unsafe { &*(storage as *const _ as *const libc::sockaddr_un) };
    let maxlength = (addrlen as usize)
        .saturating_sub(SUN_PATH_OFFSET)
        .min(sun.sun_path.len());
    let abstract_name = cfg!(target_os = "linux") && maxlength > 0 && sun.sun_path[0] == 0;
    let end = if abstract_name {
        maxlength
    } else {
        sun.sun_path[..maxlength]
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(maxlength)
    };
    let bytes: Vec<u8> = sun.sun_path[..end].iter().map(|&b| b as u8).collect();
    if abstract_name {
        // `interp_socket.py space.newbytes(path)`: abstract names are bytes.
        pyre_object::bytesobject::w_bytes_from_bytes(&bytes)
    } else {
        // `interp_socket.py space.newfilename(path)`: read-back uses the filesystem
        // decoding so a byte with no UTF-8 spelling survives the round trip.
        crate::gateway::fsdecode_filename_bytes(&bytes)
    }
}

#[cfg(any(unix, windows))]
fn init_socket_type(ns: pyre_object::PyObjectRef) {
    // `interp_socket.py:W_Socket.__init__` first creates an empty wrapper;
    // `descr_init` below installs the RSocket state.  Keeping allocation and
    // initialisation separate is required when socket.py's Python subclass
    // explicitly calls `_socket.socket.__init__(self, ...)`.
    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
        ns,
        "__new__",
        crate::typedef::make_new_descr(|args| {
            let cls = args
                .first()
                .copied()
                .filter(|w_cls| unsafe { pyre_object::is_type(*w_cls) })
                .unwrap_or_else(socket_type);
            Ok(pyre_object::w_instance_new(cls))
        }),
    ) };
    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
        ns,
        "__del__",
        crate::make_builtin_function_with_arity(
            "__del__",
            |args| {
                let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                let fd = rffi::socket_from_i64(socket_get_attr_i64(obj, "_fd"));
                if !rffi::is_invalid(fd) {
                    if let Ok(repr) = unsafe { crate::display::py_repr_wtf8(obj) } {
                        let _ = crate::warn::warn_category(
                            &format!("unclosed {}", repr.to_string_lossy()),
                            "ResourceWarning",
                            1,
                        );
                    }
                    let _ = unsafe { rffi::close(fd) };
                    socket_set_attr(obj, "_fd", pyre_object::w_int_new(-1));
                }
                Ok(pyre_object::w_none())
            },
            1,
        ),
    ) };
    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
        ns,
        "__init__",
        crate::make_builtin_function("__init__", |args| {
            // `interp_socket.py descr_init(family=-1, type=-1, proto=-1,
            // w_fileno=None)`.
            let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
            let after_self = if args.is_empty() { args } else { &args[1..] };
            let (pos, kwargs) = crate::builtins::split_builtin_kwargs(after_self);
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
            // `interp_socket.py descr_init(family=-1, type=-1, proto=-1,
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
                    family = rffi::AF_INET;
                }
                if ty == -1 {
                    ty = rffi::SOCK_STREAM;
                }
                if proto == -1 {
                    proto = 0;
                }
                let fd = unsafe { rffi::socket(family, ty, proto) };
                if rffi::is_invalid(fd) {
                    return Err(socket_last_error());
                }
                // `rsocket.py:RSocket.__init__` keeps every newly created
                // socket out of an exec'd child (PEP 446).
                rffi::set_cloexec(fd);
                socket_init_state(obj, fd, family, ty, proto);
                return Ok(pyre_object::w_none());
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
            // `interp_socket.py` — `space.int_w(w_fileno)` accepts ints,
            // longs, and objects with `__int__` / `__index__`.
            let fd = crate::baseobjspace::int_w(fileno_obj)?;
            if fd < 0 {
                return Err(crate::PyError::value_error("negative file descriptor"));
            }
            let fd = rffi::socket_from_i64(fd);
            if family == -1 {
                family = socket_detect_family(fd)?;
            }
            if ty == -1 {
                ty = socket_getsockopt_int(fd, rffi::SOL_SOCKET, rffi::SO_TYPE)?;
            }
            if proto == -1 {
                proto = socket_get_so_protocol(fd)?;
            }
            socket_init_state(obj, fd, family, ty, proto);
            Ok(pyre_object::w_none())
        }),
    ) };

    // `interp_socket.py:1157-1160` — `family`/`type`/`proto`/`timeout`
    // are GetSetProperty data descriptors (plain attribute access, not
    // callables).  The getter receives `(descriptor, instance)`, so the
    // socket object is `args[1]`.
    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
    ) };
    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
    ) };
    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
    ) };
    // `interp_socket.py gettimeout_w` — `timeout` is the stored
    // `_timeout` object (float, or `None` when disabled).
    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
        ns,
        "timeout",
        crate::typedef::make_getset_descriptor_named(
            crate::make_builtin_function_with_arity(
                "timeout",
                |args| {
                    let d = crate::baseobjspace::getdict_native(args[1]);
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
    ) };

    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
    ) };

    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
        ns,
        "close",
        crate::make_builtin_function_with_arity(
            "close",
            |args| {
                let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                let fd = rffi::socket_from_i64(socket_get_attr_i64(obj, "_fd"));
                if !rffi::is_invalid(fd) {
                    let _ = unsafe { rffi::close(fd) };
                    socket_set_attr(obj, "_fd", pyre_object::w_int_new(-1));
                }
                Ok(pyre_object::w_none())
            },
            1,
        ),
    ) };

    // detach() → returns the fd and forgets it.
    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
    ) };

    // `interp_socket.py _reuse_w / _drop_w` — refcount methods
    // the app-level `socket._socketobject` wrapper uses to share one
    // underlying fd across `socket.makefile()` file-like aliases.
    // `_reuse` increments the usecount; `_drop` decrements and closes
    // when it reaches zero.
    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
    ) };
    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
                    let fd = rffi::socket_from_i64(socket_get_attr_i64(obj, "_fd"));
                    if !rffi::is_invalid(fd) {
                        let _ = unsafe { rffi::close(fd) };
                        socket_set_attr(obj, "_fd", pyre_object::w_int_new(-1));
                    }
                }
                Ok(pyre_object::w_none())
            },
            1,
        ),
    ) };

    // bind(addr) — addr is (host, port) for AF_INET / (host, port, flowinfo,
    // scopeid) for AF_INET6 / path string for AF_UNIX.
    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
                let proto = socket_get_attr_i64(obj, "_proto") as libc::c_int;
                let (storage, slen) = pack_inet_addr("bind", family, proto, args[1])?;
                let r =
                    unsafe { rffi::bind(fd, &storage as *const _ as *const rffi::sockaddr, slen) };
                if r != 0 {
                    return Err(socket_last_error());
                }
                Ok(pyre_object::w_none())
            },
            2,
        ),
    ) };

    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
            let r = unsafe { rffi::listen(fd, backlog) };
            if r != 0 {
                return Err(socket_last_error());
            }
            Ok(pyre_object::w_none())
        }),
    ) };

    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
        ns,
        "accept",
        crate::make_builtin_function_with_arity(
            "accept",
            |args| {
                let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                let fd = socket_fd(obj)?;
                socket_wait_readable(obj, fd)?;
                let family = socket_get_attr_i64(obj, "_family") as libc::c_int;
                let ty = socket_get_attr_i64(obj, "_type") as libc::c_int;
                let proto = socket_get_attr_i64(obj, "_proto") as libc::c_int;
                let mut storage: rffi::sockaddr_storage = unsafe { std::mem::zeroed() };
                let mut slen = core::mem::size_of::<rffi::sockaddr_storage>() as rffi::SockLen;
                let cfd = loop {
                    let (r, errno) = socket_call(|| unsafe {
                        rffi::accept(fd, &mut storage as *mut _ as *mut rffi::sockaddr, &mut slen)
                    });
                    if !rffi::is_invalid(r) {
                        break r;
                    }
                    if !rffi::error_is_interrupted(errno) {
                        return Err(socket_io_err_for_operation(
                            obj,
                            std::io::Error::from_raw_os_error(errno),
                        ));
                    }
                    // EINTR: deliver a pending signal, then retry
                    // (`converted_error` eintr_retry).
                    crate::module::signal::interp_signal::checksignals_now()?;
                };
                // `rsocket.py:RSocket._accept` returns the new descriptor
                // already closed over an exec (rsocket uses
                // accept4(SOCK_CLOEXEC) on Linux; this is the portable path).
                rffi::set_cloexec(cfd);
                let new_sock = socket_from_fd(cfd, family, ty, proto);
                let addr = unpack_inet_addr(&storage, slen);
                Ok(pyre_object::w_tuple_new(vec![new_sock, addr]))
            },
            1,
        ),
    ) };

    // `interp_socket.py socketmethodnames _accept` — primitive
    // returning `(fd, addr)`.  CPython's app-level `socket.py:262 def
    // accept` wraps this to construct the new socket object;
    // pyre's `accept` above bundles both steps for callers that
    // bypass the stdlib wrapper.
    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
        ns,
        "_accept",
        crate::make_builtin_function_with_arity(
            "_accept",
            |args| {
                let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                let fd = socket_fd(obj)?;
                socket_wait_readable(obj, fd)?;
                let mut storage: rffi::sockaddr_storage = unsafe { std::mem::zeroed() };
                let mut slen = core::mem::size_of::<rffi::sockaddr_storage>() as rffi::SockLen;
                let cfd = loop {
                    let (r, errno) = socket_call(|| unsafe {
                        rffi::accept(fd, &mut storage as *mut _ as *mut rffi::sockaddr, &mut slen)
                    });
                    if !rffi::is_invalid(r) {
                        break r;
                    }
                    if !rffi::error_is_interrupted(errno) {
                        return Err(socket_io_err_for_operation(
                            obj,
                            std::io::Error::from_raw_os_error(errno),
                        ));
                    }
                    // EINTR: deliver a pending signal, then retry
                    // (`converted_error` eintr_retry).
                    crate::module::signal::interp_signal::checksignals_now()?;
                };
                rffi::set_cloexec(cfd);
                let addr = unpack_inet_addr(&storage, slen);
                Ok(pyre_object::w_tuple_new(vec![
                    pyre_object::w_int_new(rffi::socket_to_i64(cfd)),
                    addr,
                ]))
            },
            1,
        ),
    ) };

    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
                let proto = socket_get_attr_i64(obj, "_proto") as libc::c_int;
                let (storage, slen) = pack_inet_addr("connect", family, proto, args[1])?;
                #[cfg(windows)]
                if let Some(timeout) = socket_positive_timeout(obj) {
                    return match socket_connect_timed(fd, &storage, slen, timeout) {
                        Ok(()) => Ok(pyre_object::w_none()),
                        Err(errno) => Err(socket_io_err_for_operation(
                            obj,
                            std::io::Error::from_raw_os_error(errno),
                        )),
                    };
                }
                loop {
                    let (r, errno) = socket_call(|| unsafe {
                        rffi::connect(fd, &storage as *const _ as *const rffi::sockaddr, slen)
                    });
                    if r == 0 {
                        break;
                    }
                    if !rffi::error_is_interrupted(errno) {
                        return Err(socket_io_err(std::io::Error::from_raw_os_error(errno)));
                    }
                    // EINTR: deliver a pending signal, then retry
                    // (`converted_error` eintr_retry).
                    crate::module::signal::interp_signal::checksignals_now()?;
                }
                Ok(pyre_object::w_none())
            },
            2,
        ),
    ) };

    // connect_ex(address) → errno (no exception on error)
    // `interp_socket.py:376-392` — `try: connect; except` equivalent
    // that returns the errno integer instead of raising OSError.
    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
                let proto = socket_get_attr_i64(obj, "_proto") as libc::c_int;
                let (storage, slen) = pack_inet_addr("connect_ex", family, proto, args[1])?;
                #[cfg(windows)]
                if let Some(timeout) = socket_positive_timeout(obj) {
                    let err = socket_connect_timed(fd, &storage, slen, timeout).err();
                    return Ok(pyre_object::w_int_new(err.unwrap_or(0) as i64));
                }
                // `interp_socket.py:387-391` — retry while the call is
                // interrupted (EINTR), otherwise return the errno.
                let err = loop {
                    let (r, e) = socket_call(|| unsafe {
                        rffi::connect(fd, &storage as *const _ as *const rffi::sockaddr, slen)
                    });
                    if r == 0 {
                        break 0;
                    }
                    if !rffi::error_is_interrupted(e) {
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
    ) };

    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
        ns,
        "send",
        crate::make_builtin_function("send", |args| {
            if args.len() < 2 {
                return Err(crate::PyError::type_error("send() missing buffer"));
            }
            let obj = args[0];
            let fd = socket_fd(obj)?;
            let buffer = crate::baseobjspace::simple_buffer_bytes(args[1])?.ok_or_else(|| {
                crate::PyError::type_error("send: buffer must be bytes-like")
            })?;
            let flags = if args.len() >= 3 {
                (unsafe { pyre_object::w_int_get_value(args[2]) }) as libc::c_int
            } else {
                0
            };
            // `PyBuffer_Release` after the call — `SimpleBufferBytes` has no
            // `Drop`, so an export that is never released leaves a `bytearray`
            // argument permanently exported and unable to be resized. The
            // borrow of `buffer` is scoped to the closure so the release also
            // covers the error paths.
            let result = (|| -> Result<isize, crate::PyError> {
                let buf = buffer.as_bytes();
                loop {
                    let (r, errno) = socket_call(|| unsafe {
                        rffi::send(fd, buf.as_ptr() as *const libc::c_void, buf.len(), flags)
                    });
                    if r >= 0 {
                        return Ok(r);
                    }
                    if !rffi::error_is_interrupted(errno) {
                        return Err(socket_io_err_for_operation(
                            obj,
                            std::io::Error::from_raw_os_error(errno),
                        ));
                    }
                    // EINTR: deliver a pending signal, then retry
                    // (`converted_error` eintr_retry).
                    crate::module::signal::interp_signal::checksignals_now()?;
                }
            })();
            buffer.release();
            Ok(pyre_object::w_int_new(result? as i64))
        }),
    ) };

    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
        ns,
        "sendall",
        crate::make_builtin_function("sendall", |args| {
            if args.len() < 2 {
                return Err(crate::PyError::type_error("sendall() missing buffer"));
            }
            let obj = args[0];
            let fd = socket_fd(obj)?;
            let buffer = crate::baseobjspace::simple_buffer_bytes(args[1])?.ok_or_else(|| {
                crate::PyError::type_error("sendall: buffer must be bytes-like")
            })?;
            let flags = if args.len() >= 3 {
                (unsafe { pyre_object::w_int_get_value(args[2]) }) as libc::c_int
            } else {
                0
            };
            // See `send` above: the export is released once the borrow of
            // `buffer` ends, on the error path as well.
            let result = (|| -> Result<(), crate::PyError> {
                let buf = buffer.as_bytes();
                let mut off = 0usize;
                while off < buf.len() {
                    let (n, errno) = socket_call(|| unsafe {
                        rffi::send(
                            fd,
                            buf[off..].as_ptr() as *const libc::c_void,
                            buf.len() - off,
                            flags,
                        )
                    });
                    if n < 0 {
                        if rffi::error_is_interrupted(errno) {
                            // `rsocket.py:1132` signal_checker — deliver a
                            // pending signal, then retry the remaining bytes.
                            crate::module::signal::interp_signal::checksignals_now()?;
                            continue;
                        }
                        return Err(socket_io_err_for_operation(
                            obj,
                            std::io::Error::from_raw_os_error(errno),
                        ));
                    }
                    off += n as usize;
                }
                Ok(())
            })();
            buffer.release();
            result?;
            Ok(pyre_object::w_none())
        }),
    ) };

    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
                let (r, errno) = socket_call(|| unsafe {
                    rffi::recv(fd, buf.as_mut_ptr() as *mut libc::c_void, n, flags)
                });
                if r >= 0 {
                    break r;
                }
                if !rffi::error_is_interrupted(errno) {
                    return Err(socket_io_err_for_operation(
                        obj,
                        std::io::Error::from_raw_os_error(errno),
                    ));
                }
                // EINTR: deliver a pending signal, then retry
                // (`converted_error` eintr_retry).
                crate::module::signal::interp_signal::checksignals_now()?;
            };
            buf.truncate(got as usize);
            Ok(pyre_object::bytesobject::w_bytes_from_bytes(&buf))
        }),
    ) };

    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
            let buffer = crate::baseobjspace::simple_buffer_bytes(args[1])?.ok_or_else(|| {
                crate::PyError::type_error("sendto: buffer must be bytes-like")
            })?;
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
            // See `send` above: the export is released once the borrow of
            // `buffer` ends, on the error path as well. `pack_inet_addr` runs
            // Python, so it is inside the released scope too.
            let result = (|| -> Result<isize, crate::PyError> {
                let buf = buffer.as_bytes();
                let family = socket_get_attr_i64(obj, "_family") as libc::c_int;
                let proto = socket_get_attr_i64(obj, "_proto") as libc::c_int;
                let (storage, slen) = pack_inet_addr("sendto", family, proto, addr_obj)?;
                loop {
                    let (r, errno) = socket_call(|| unsafe {
                        rffi::sendto(
                            fd,
                            buf.as_ptr() as *const libc::c_void,
                            buf.len(),
                            flags,
                            &storage as *const _ as *const rffi::sockaddr,
                            slen,
                        )
                    });
                    if r >= 0 {
                        return Ok(r);
                    }
                    if !rffi::error_is_interrupted(errno) {
                        return Err(socket_io_err_for_operation(
                            obj,
                            std::io::Error::from_raw_os_error(errno),
                        ));
                    }
                    // EINTR: deliver a pending signal, then retry
                    // (`converted_error` eintr_retry).
                    crate::module::signal::interp_signal::checksignals_now()?;
                }
            })();
            buffer.release();
            Ok(pyre_object::w_int_new(result? as i64))
        }),
    ) };

    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
            let mut storage: rffi::sockaddr_storage = unsafe { std::mem::zeroed() };
            let mut slen = core::mem::size_of::<rffi::sockaddr_storage>() as rffi::SockLen;
            let got = loop {
                let (r, errno) = socket_call(|| unsafe {
                    rffi::recvfrom(
                        fd,
                        buf.as_mut_ptr() as *mut libc::c_void,
                        n,
                        flags,
                        &mut storage as *mut _ as *mut rffi::sockaddr,
                        &mut slen,
                    )
                });
                if r >= 0 {
                    break r;
                }
                if !rffi::error_is_interrupted(errno) {
                    return Err(socket_io_err_for_operation(
                        obj,
                        std::io::Error::from_raw_os_error(errno),
                    ));
                }
                // EINTR: deliver a pending signal, then retry
                // (`converted_error` eintr_retry).
                crate::module::signal::interp_signal::checksignals_now()?;
            };
            buf.truncate(got as usize);
            let addr = unpack_inet_addr(&storage, slen);
            Ok(pyre_object::w_tuple_new(vec![
                pyre_object::bytesobject::w_bytes_from_bytes(&buf),
                addr,
            ]))
        }),
    ) };

    // recv_into(buffer, [nbytes, flags]) → nbytes_read
    // `interp_socket.py:831-863` — writes directly into a writable
    // bytes-like buffer.  nbytes==0 uses the full buffer length.
    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
        ns,
        "recv_into",
        crate::make_builtin_function("recv_into", |args| {
            if args.len() < 2 {
                return Err(crate::PyError::type_error("recv_into() missing buffer"));
            }
            let obj = args[0];
            let buf_obj = args[1];
            let mut buffer = unsafe { SocketWritableBuffer::acquire(buf_obj) }?;
            let slot = unsafe { buffer.as_mut_slice() };
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
                let (r, errno) = socket_call(|| unsafe {
                    rffi::recv(fd, slot.as_mut_ptr() as *mut libc::c_void, nbytes, flags)
                });
                if r >= 0 {
                    break r;
                }
                if !rffi::error_is_interrupted(errno) {
                    return Err(socket_io_err_for_operation(
                        obj,
                        std::io::Error::from_raw_os_error(errno),
                    ));
                }
                // EINTR: deliver a pending signal, then retry
                // (`converted_error` eintr_retry).
                crate::module::signal::interp_signal::checksignals_now()?;
            };
            Ok(pyre_object::w_int_new(got as i64))
        }),
    ) };

    // recvfrom_into(buffer, [nbytes, flags]) → (nbytes, address)
    // `interp_socket.py:866-899` — recvfrom variant that fills a
    // caller-provided buffer rather than allocating a new bytes.
    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
        ns,
        "recvfrom_into",
        crate::make_builtin_function("recvfrom_into", |args| {
            if args.len() < 2 {
                return Err(crate::PyError::type_error("recvfrom_into() missing buffer"));
            }
            let obj = args[0];
            let buf_obj = args[1];
            let mut buffer = unsafe { SocketWritableBuffer::acquire(buf_obj) }?;
            let slot = unsafe { buffer.as_mut_slice() };
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
            let mut storage: rffi::sockaddr_storage = unsafe { std::mem::zeroed() };
            let mut slen = core::mem::size_of::<rffi::sockaddr_storage>() as rffi::SockLen;
            let got = loop {
                let (r, errno) = socket_call(|| unsafe {
                    rffi::recvfrom(
                        fd,
                        slot.as_mut_ptr() as *mut libc::c_void,
                        nbytes,
                        flags,
                        &mut storage as *mut _ as *mut rffi::sockaddr,
                        &mut slen,
                    )
                });
                if r >= 0 {
                    break r;
                }
                if !rffi::error_is_interrupted(errno) {
                    return Err(socket_io_err_for_operation(
                        obj,
                        std::io::Error::from_raw_os_error(errno),
                    ));
                }
                // EINTR: deliver a pending signal, then retry
                // (`converted_error` eintr_retry).
                crate::module::signal::interp_signal::checksignals_now()?;
            };
            let addr = unpack_inet_addr(&storage, slen);
            Ok(pyre_object::w_tuple_new(vec![
                pyre_object::w_int_new(got as i64),
                addr,
            ]))
        }),
    ) };

    // recvmsg(bufsize, [ancbufsize, flags]) → (data, ancdata, msg_flags, address)
    // `interp_socket.py:525-569` — receives normal + ancillary data
    // via libc::recvmsg.  ancdata is a list of (cmsg_level, cmsg_type,
    // cmsg_data:bytes) triples walked through CMSG_FIRSTHDR /
    // CMSG_NXTHDR / CMSG_DATA.
    // The scatter/gather calls and their ancillary data are POSIX-only;
    // `socket.py:557,569` test for each before reaching for it.
    #[cfg(unix)]
    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
            let mut storage: rffi::sockaddr_storage = unsafe { std::mem::zeroed() };
            let (got, msg_flags, msg_namelen) = loop {
                let mut iov = libc::iovec {
                    iov_base: data.as_mut_ptr() as *mut libc::c_void,
                    iov_len: bufsize,
                };
                let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
                msg.msg_name = &mut storage as *mut _ as *mut libc::c_void;
                msg.msg_namelen = core::mem::size_of::<rffi::sockaddr_storage>() as rffi::SockLen;
                msg.msg_iov = &mut iov;
                msg.msg_iovlen = 1;
                if ancbufsize > 0 {
                    msg.msg_control = control.as_mut_ptr() as *mut libc::c_void;
                    msg.msg_controllen = ancbufsize as _;
                }
                let (r, errno) = socket_call(|| unsafe {
                    libc::recvmsg(fd, &mut msg, flags)
                });
                if r >= 0 {
                    break (r, msg.msg_flags, msg.msg_namelen);
                }
                if !rffi::error_is_interrupted(errno) {
                    return Err(socket_io_err_for_operation(
                        args[0],
                        std::io::Error::from_raw_os_error(errno),
                    ));
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
            let addr = unpack_inet_addr(&storage, msg_namelen);
            Ok(pyre_object::w_tuple_new(vec![
                pyre_object::bytesobject::w_bytes_from_bytes(&data),
                pyre_object::w_list_new(anc_items),
                pyre_object::w_int_new(msg_flags as i64),
                addr,
            ]))
        }),
    ) };

    // recvmsg_into(buffers, [ancbufsize, [flags]]) ->
    //   (nbytes, ancdata, msg_flags, address)
    // `interp_socket.py recvmsg_into_w` — scatter-receive into
    // a list/tuple of writable buffers; each `writebuf_w` slice
    // contributes one iovec entry.
    // The scatter/gather calls and their ancillary data are POSIX-only;
    // `socket.py:557,569` test for each before reaching for it.
    #[cfg(unix)]
    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
            let mut buffers: Vec<SocketWritableBuffer> = Vec::with_capacity(nbufs);
            for i in 0..nbufs {
                let item = unsafe {
                    if is_list {
                        pyre_object::w_list_getitem(seq, i as i64)
                    } else {
                        pyre_object::w_tuple_getitem(seq, i as i64)
                    }
                }
                .ok_or_else(|| crate::PyError::type_error("recvmsg_into: buffer item missing"))?;
                buffers.push(unsafe { SocketWritableBuffer::acquire(item) }?);
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
                .iter_mut()
                .map(|buffer| {
                    let slice = unsafe { buffer.as_mut_slice() };
                    libc::iovec {
                    iov_base: slice.as_mut_ptr() as *mut libc::c_void,
                    iov_len: slice.len(),
                    }
                })
                .collect();
            let mut control = vec![0u8; ancbufsize];
            let mut storage: rffi::sockaddr_storage = unsafe { std::mem::zeroed() };
            let (got, msg_flags, msg_namelen, controllen) = loop {
                let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
                msg.msg_name = &mut storage as *mut _ as *mut libc::c_void;
                msg.msg_namelen = core::mem::size_of::<rffi::sockaddr_storage>() as rffi::SockLen;
                msg.msg_iov = iovs.as_mut_ptr();
                msg.msg_iovlen = iovs.len() as _;
                if ancbufsize > 0 {
                    msg.msg_control = control.as_mut_ptr() as *mut libc::c_void;
                    msg.msg_controllen = ancbufsize as _;
                }
                let (r, errno) = socket_call(|| unsafe {
                    libc::recvmsg(fd, &mut msg, flags)
                });
                if r >= 0 {
                    break (r, msg.msg_flags, msg.msg_namelen, msg.msg_controllen);
                }
                if !rffi::error_is_interrupted(errno) {
                    return Err(socket_io_err_for_operation(
                        args[0],
                        std::io::Error::from_raw_os_error(errno),
                    ));
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
            let addr = unpack_inet_addr(&storage, msg_namelen);
            Ok(pyre_object::w_tuple_new(vec![
                pyre_object::w_int_new(got as i64),
                pyre_object::w_list_new(anc_items),
                pyre_object::w_int_new(msg_flags as i64),
                addr,
            ]))
        }),
    ) };

    // sendmsg(data_iter[, ancillary[, flags[, address]]]) → bytes_sent
    // `interp_socket.py:711-773` — gather-write of multiple bytes-like
    // buffers plus optional ancillary control messages.  Each cmsg is
    // a (cmsg_level, cmsg_type, cmsg_data) 3-tuple; we lay them out
    // into a single control buffer via CMSG_SPACE / CMSG_NXTHDR.
    // The scatter/gather calls and their ancillary data are POSIX-only;
    // `socket.py:557,569` test for each before reaching for it.
    #[cfg(unix)]
    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
        ns,
        "sendmsg",
        crate::make_builtin_function("sendmsg", |args| {
            if args.len() < 2 {
                return Err(crate::PyError::type_error("sendmsg() missing data"));
            }
            let obj = args[0];
            let fd = socket_fd(obj)?;

            // PyPy `interp_socket.py`: `space.unpackiterable(w_data)`,
            // then acquire/release one `BUF_SIMPLE` view per item and retain
            // the copied strings.  In particular, asyncio passes an
            // `itertools.islice` of memoryviews here, not a list or tuple.
            let _roots = pyre_object::gc_roots::push_roots();
            let data_slot = pyre_object::gc_roots::pin_roots(&[args[1]]);
            let data_items = crate::baseobjspace::unpackiterable(
                pyre_object::gc_roots::shadow_stack_get(data_slot),
                -1,
            )?;
            let items_base = pyre_object::gc_roots::pin_roots(&data_items);
            let mut data_buffers: Vec<Vec<u8>> = Vec::with_capacity(data_items.len());
            // `simple_buffer_bytes` looks up `__buffer__` and builds a
            // memoryview, so every iteration can collect; read each item back
            // from its slot instead of consuming the unrooted vector.
            for i in 0..data_items.len() {
                let item = pyre_object::gc_roots::shadow_stack_get(items_base + i);
                let Some(buffer) = crate::baseobjspace::simple_buffer_bytes(item)? else {
                    return Err(crate::PyError::type_error(
                        "sendmsg: data items must be bytes-like",
                    ));
                };
                data_buffers.push(buffer.as_bytes().to_vec());
                buffer.release();
            }
            let mut iovs: Vec<libc::iovec> = data_buffers
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
                    // interp_socket.py:811-816 acquires a BUF_SIMPLE view,
                    // just like the ordinary data items above.  This includes
                    // array.array payloads used by multiprocessing's
                    // SCM_RIGHTS fd transfer, not only bytes/bytearray.
                    let Some(buffer) = crate::baseobjspace::simple_buffer_bytes(data_o)? else {
                        return Err(crate::PyError::type_error(
                            "sendmsg: ancillary data must be bytes-like",
                        ));
                    };
                    let level = unsafe { pyre_object::w_int_get_value(level_o) } as libc::c_int;
                    let ty = unsafe { pyre_object::w_int_get_value(type_o) } as libc::c_int;
                    let data = buffer.as_bytes().to_vec();
                    buffer.release();
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
                    let proto = socket_get_attr_i64(obj, "_proto") as libc::c_int;
                    let (s, l) = pack_inet_addr("sendmsg", family, proto, args[4])?;
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
                let (r, errno) = socket_call(|| unsafe {
                    libc::sendmsg(fd, &msg, flags)
                });
                if r >= 0 {
                    break r;
                }
                if !rffi::error_is_interrupted(errno) {
                    return Err(socket_io_err_for_operation(
                        obj,
                        std::io::Error::from_raw_os_error(errno),
                    ));
                }
                // EINTR: deliver a pending signal, then retry
                // (`converted_error` eintr_retry).
                crate::module::signal::interp_signal::checksignals_now()?;
            };
            Ok(pyre_object::w_int_new(sent as i64))
        }),
    ) };

    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
                let r = unsafe { rffi::shutdown(fd, how) };
                if r != 0 {
                    return Err(socket_last_error());
                }
                Ok(pyre_object::w_none())
            },
            2,
        ),
    ) };

    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
        ns,
        "getsockname",
        crate::make_builtin_function_with_arity(
            "getsockname",
            |args| {
                let fd = socket_fd(args.first().copied().unwrap_or(pyre_object::PY_NULL))?;
                let mut storage: rffi::sockaddr_storage = unsafe { std::mem::zeroed() };
                let mut slen = core::mem::size_of::<rffi::sockaddr_storage>() as rffi::SockLen;
                let r = unsafe {
                    rffi::getsockname(fd, &mut storage as *mut _ as *mut rffi::sockaddr, &mut slen)
                };
                if r != 0 {
                    return Err(socket_last_error());
                }
                Ok(unpack_inet_addr(&storage, slen))
            },
            1,
        ),
    ) };

    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
        ns,
        "getpeername",
        crate::make_builtin_function_with_arity(
            "getpeername",
            |args| {
                let fd = socket_fd(args.first().copied().unwrap_or(pyre_object::PY_NULL))?;
                let mut storage: rffi::sockaddr_storage = unsafe { std::mem::zeroed() };
                let mut slen = core::mem::size_of::<rffi::sockaddr_storage>() as rffi::SockLen;
                let r = unsafe {
                    rffi::getpeername(fd, &mut storage as *mut _ as *mut rffi::sockaddr, &mut slen)
                };
                if r != 0 {
                    return Err(socket_last_error());
                }
                Ok(unpack_inet_addr(&storage, slen))
            },
            1,
        ),
    ) };

    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
            // `sock_setsockopt` sends an int for this option through
            // `WSAIoctl` and keeps the value, because the option number is an
            // ioctl code that `setsockopt` itself rejects.  A bytes value is
            // not covered there either and still goes the ordinary way.
            #[cfg(windows)]
            if name == windows_sys::Win32::Networking::WinSock::SIO_TCP_SET_ACK_FREQUENCY as libc::c_int
                && unsafe { pyre_object::is_int(val) }
            {
                let flag = (unsafe { pyre_object::w_int_get_value(val) }) as libc::c_int;
                if unsafe { rffi::set_ack_frequency(fd, flag) } != 0 {
                    return Err(socket_last_error());
                }
                socket_set_attr(args[0], "_quickack", pyre_object::w_int_new(flag as i64));
                return Ok(pyre_object::w_none());
            }
            let r = unsafe {
                if pyre_object::is_int(val) {
                    let v = pyre_object::w_int_get_value(val) as libc::c_int;
                    rffi::setsockopt(
                        fd,
                        level,
                        name,
                        &v as *const _ as *const libc::c_void,
                        core::mem::size_of::<libc::c_int>() as rffi::SockLen,
                    )
                } else if pyre_object::bytesobject::is_bytes_like(val) {
                    let data = pyre_object::bytesobject::bytes_like_data(val);
                    rffi::setsockopt(
                        fd,
                        level,
                        name,
                        data.as_ptr() as *const libc::c_void,
                        data.len() as rffi::SockLen,
                    )
                } else {
                    return Err(crate::PyError::type_error(
                        "setsockopt: value must be int or bytes-like",
                    ));
                }
            };
            if r != 0 {
                return Err(socket_last_error());
            }
            Ok(pyre_object::w_none())
        }),
    ) };

    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
            // `interp_socket.py getsockopt_w` — `buflen == 0`
            // (including when omitted) reads an int option; otherwise the
            // length must be in `1..=1024` and a bytes buffer is returned.
            let buflen = if args.len() >= 4 {
                unsafe { pyre_object::w_int_get_value(args[3]) }
            } else {
                0
            };
            if buflen == 0 {
                // `sock_getsockopt` answers the value `setsockopt` last wrote
                // rather than asking WinSock, which has no call that reads an
                // ioctl's current setting.
                #[cfg(windows)]
                if name == windows_sys::Win32::Networking::WinSock::SIO_TCP_SET_ACK_FREQUENCY as libc::c_int {
                    return Ok(pyre_object::w_int_new(socket_get_attr_i64(
                        args[0],
                        "_quickack",
                    )));
                }
                let mut v: libc::c_int = 0;
                let mut sz = core::mem::size_of::<libc::c_int>() as rffi::SockLen;
                let r = unsafe {
                    rffi::getsockopt(
                        fd,
                        level,
                        name,
                        &mut v as *mut _ as *mut libc::c_void,
                        &mut sz,
                    )
                };
                if r != 0 {
                    return Err(socket_last_error());
                }
                Ok(pyre_object::w_int_new(v as i64))
            } else {
                if !(0..=1024).contains(&buflen) {
                    return Err(crate::PyError::os_error("getsockopt buflen out of range"));
                }
                let buflen = buflen as usize;
                let mut buf = vec![0u8; buflen];
                let mut sz = buflen as rffi::SockLen;
                let r = unsafe {
                    rffi::getsockopt(
                        fd,
                        level,
                        name,
                        buf.as_mut_ptr() as *mut libc::c_void,
                        &mut sz,
                    )
                };
                if r != 0 {
                    return Err(socket_last_error());
                }
                buf.truncate(sz as usize);
                Ok(pyre_object::bytesobject::w_bytes_from_bytes(&buf))
            }
        }),
    ) };

    // `interp_socket.py setblocking_w` per PyPy docstring: True
    // is equivalent to `settimeout(None)`, False to `settimeout(0.0)`.
    // Routing through `socket_apply_timeout` keeps the SO_*TIMEO state
    // consistent with the timeout attribute and prevents a stale
    // SO_RCVTIMEO from surviving a `setblocking(True)` call.
    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
    ) };

    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
        ns,
        "getblocking",
        crate::make_builtin_function_with_arity(
            "getblocking",
            |args| {
                let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                socket_fd(obj)?;
                // `sock_getblocking` answers from the stored timeout rather
                // than from the descriptor: `settimeout` is its only writer,
                // and WinSock's `FIONBIO` cannot be read back.
                let d = crate::baseobjspace::getdict_native(obj);
                let blocking = d.is_null()
                    || unsafe { pyre_object::w_dict_getitem_str(d, "_timeout") }
                        .filter(|t| unsafe { pyre_object::is_float(*t) })
                        .is_none_or(|t| unsafe {
                            pyre_object::floatobject::w_float_get_value(t) != 0.0
                        });
                Ok(pyre_object::w_bool_from(blocking))
            },
            1,
        ),
    ) };

    // `interp_socket.py settimeout_w` then `rsocket.py:RSocket.
    // settimeout`: None → blocking (no O_NONBLOCK, no SO_*TIMEO); 0.0 →
    // non-blocking (O_NONBLOCK on); >0 → blocking + SO_RCVTIMEO +
    // SO_SNDTIMEO set to the duration; <0 → ValueError "Timeout value
    // out of range".
    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
                // `gettimeout()` reports a float, and the timeout readers below
                // recognise only that type; storing the caller's `int` verbatim
                // would make `settimeout(1)` silently untimed.
                let stored = if unsafe { pyre_object::is_none(w_t) } {
                    w_t
                } else {
                    pyre_object::floatobject::w_float_new(timeout)
                };
                socket_set_attr(obj, "_timeout", stored);
                Ok(pyre_object::w_none())
            },
            2,
        ),
    ) };

    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
        ns,
        "gettimeout",
        crate::make_builtin_function_with_arity(
            "gettimeout",
            |args| {
                let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
                let d = crate::baseobjspace::getdict_native(obj);
                if d.is_null() {
                    return Ok(pyre_object::w_none());
                }
                Ok(unsafe { pyre_object::w_dict_getitem_str(d, "_timeout") }
                    .unwrap_or(pyre_object::w_none()))
            },
            1,
        ),
    ) };

    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
        ns,
        "__enter__",
        crate::make_builtin_function_with_arity(
            "__enter__",
            |args| Ok(args.first().copied().unwrap_or(pyre_object::w_none())),
            1,
        ),
    ) };

    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
        ns,
        "__exit__",
        crate::make_builtin_function("__exit__", |args| {
            if let Some(&obj) = args.first() {
                let fd = rffi::socket_from_i64(socket_get_attr_i64(obj, "_fd"));
                if !rffi::is_invalid(fd) {
                    let _ = unsafe { rffi::close(fd) };
                    socket_set_attr(obj, "_fd", pyre_object::w_int_new(-1));
                }
            }
            Ok(pyre_object::w_bool_from(false))
        }),
    ) };

    // __repr__ — `interp_socket.py descr_repr`.  Format
    // matches CPython: `<socket object, fd=N, family=F, type=T, proto=P>`.
    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
    ) };

    // set_inheritable / get_inheritable — `interp_socket.py` wraps whether
    // an exec'd child keeps the descriptor.
    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
                rffi::set_inheritable(fd, want_inheritable).map_err(socket_io_err)?;
                Ok(pyre_object::w_none())
            },
            2,
        ),
    ) };
    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
        ns,
        "get_inheritable",
        crate::make_builtin_function_with_arity(
            "get_inheritable",
            |args| {
                let fd = socket_fd(args.first().copied().unwrap_or(pyre_object::PY_NULL))?;
                let inheritable = rffi::get_inheritable(fd).map_err(socket_io_err)?;
                Ok(pyre_object::w_bool_from(inheritable))
            },
            1,
        ),
    ) };
}
