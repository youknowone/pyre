//! Host socket layer — PyPy: `rpython/rlib/_rsocket_rffi.py`.
//!
//! `_rsocket_rffi.py` is the single place where rsocket's POSIX and `_MSVC`
//! spellings of the same call meet; everything above it names one API.  This
//! module plays that role for `interp_socket`: one set of names, a libc body
//! on unix and a WinSock body on Windows, so the module's own code stays
//! platform-neutral.  Anything with no counterpart on the other side
//! (`recvmsg`, `AF_UNIX`, the POSIX-only options) stays gated at the call
//! site instead of being faked here.

#[cfg(unix)]
pub use libc::{addrinfo, sockaddr, sockaddr_in, sockaddr_in6, sockaddr_storage};

#[cfg(windows)]
use windows_sys::Win32::Networking::WinSock as ws;

#[cfg(windows)]
pub use ws::{
    ADDRINFOA as addrinfo, SOCKADDR as sockaddr, SOCKADDR_IN as sockaddr_in,
    SOCKADDR_IN6 as sockaddr_in6, SOCKADDR_STORAGE as sockaddr_storage,
};

/// The descriptor a socket call takes.  POSIX numbers sockets alongside every
/// other file descriptor; WinSock hands out an unsigned kernel handle, so the
/// two differ in both width and signedness and only this alias may be assumed.
#[cfg(unix)]
pub type Socket = libc::c_int;
#[cfg(windows)]
pub type Socket = ws::SOCKET;

/// `getsockname`'s length argument.  `socklen_t` on POSIX, a plain `int` in
/// WinSock's prototypes.
#[cfg(unix)]
pub type SockLen = libc::socklen_t;
#[cfg(windows)]
pub type SockLen = i32;

/// The width `sockaddr.sa_family` is written through.
#[cfg(unix)]
pub type SaFamily = libc::sa_family_t;
#[cfg(windows)]
pub type SaFamily = ws::ADDRESS_FAMILY;

/// Whether a descriptor names no socket.  POSIX reserves every negative value.
/// Windows reserves only `INVALID_SOCKET` — `(SOCKET)~0` — and every handle it
/// hands out is unsigned and may exceed `i32::MAX`, so a socket there must
/// never be range-tested with `< 0`.
#[cfg(unix)]
pub fn is_invalid(s: Socket) -> bool {
    s < 0
}
#[cfg(windows)]
pub fn is_invalid(s: Socket) -> bool {
    s == ws::INVALID_SOCKET
}

/// `fileno()` reports the descriptor as a Python int, and `_fd` stores it as
/// one.  `PyLong_FromSocket_t` widens the Windows handle through a signed
/// 64-bit integer, so `INVALID_SOCKET` reads back as `-1` there too and the
/// closed-socket sentinel is the same value on both platforms.
pub fn socket_to_i64(s: Socket) -> i64 {
    // Through the pointer-width signed type, so a 32-bit `SOCKET` reaches the
    // widening as `-1` rather than as `4294967295`.
    s as isize as i64
}

pub fn socket_from_i64(v: i64) -> Socket {
    v as Socket
}

// ── constants the shared body compares against ──

#[cfg(unix)]
pub use libc::{
    AF_INET, AF_INET6, AF_UNSPEC, SO_ERROR, SO_TYPE, SOCK_DGRAM, SOCK_STREAM, SOL_SOCKET,
};
#[cfg(unix)]
pub const NI_MAXHOST: usize = libc::NI_MAXHOST as usize;
#[cfg(unix)]
pub const INADDR_ANY: u32 = libc::INADDR_ANY;
#[cfg(unix)]
pub const AI_NUMERICHOST: libc::c_int = libc::AI_NUMERICHOST;

#[cfg(windows)]
pub const AF_UNSPEC: libc::c_int = ws::AF_UNSPEC as libc::c_int;
#[cfg(windows)]
pub const AF_INET: libc::c_int = ws::AF_INET as libc::c_int;
#[cfg(windows)]
pub const AF_INET6: libc::c_int = ws::AF_INET6 as libc::c_int;
#[cfg(windows)]
pub const SOCK_STREAM: libc::c_int = ws::SOCK_STREAM;
#[cfg(windows)]
pub const SOCK_DGRAM: libc::c_int = ws::SOCK_DGRAM;
#[cfg(windows)]
pub const SOL_SOCKET: libc::c_int = ws::SOL_SOCKET;
#[cfg(windows)]
pub const SO_TYPE: libc::c_int = ws::SO_TYPE;
#[cfg(windows)]
pub const SO_ERROR: libc::c_int = ws::SO_ERROR;
/// The code an expired wait reports, so a timeout this module times itself
/// reads back the same as one the host produced.
#[cfg(windows)]
pub const ETIMEDOUT: i32 = ws::WSAETIMEDOUT;
#[cfg(windows)]
pub const NI_MAXHOST: usize = ws::NI_MAXHOST as usize;
#[cfg(windows)]
pub const INADDR_ANY: u32 = ws::INADDR_ANY;
#[cfg(windows)]
pub const AI_NUMERICHOST: libc::c_int = ws::AI_NUMERICHOST as libc::c_int;

// ── WinSock initialisation ──

/// `WSAStartup`.  Every WinSock entry point fails with `WSANOTINITIALISED`
/// until the process has made this call, and the only other things that make
/// it are the standard library's own socket code and socket2 — neither of
/// which this module goes through.  Idempotent, so the entry points below can
/// each open with it rather than depend on an initialisation order.
#[cfg(windows)]
pub fn init() {
    static STARTUP: std::sync::Once = std::sync::Once::new();
    STARTUP.call_once(|| {
        let mut data: ws::WSADATA = unsafe { core::mem::zeroed() };
        // 2.2 is the version every supported Windows implements; the process
        // keeps the library loaded for its whole life, so no WSACleanup pairs
        // with it.
        unsafe { ws::WSAStartup(0x0202, &mut data) };
    });
}

#[cfg(unix)]
pub fn init() {}

// ── error reporting ──

/// The code the last socket call failed with.  WinSock reports through
/// `WSAGetLastError` and never touches the C runtime's `errno`, which is what
/// `call_external_function` reads.
#[cfg(unix)]
pub fn last_error_code() -> i32 {
    std::io::Error::last_os_error().raw_os_error().unwrap_or(0)
}
#[cfg(windows)]
pub fn last_error_code() -> i32 {
    unsafe { ws::WSAGetLastError() }
}

pub fn last_error() -> std::io::Error {
    std::io::Error::from_raw_os_error(last_error_code())
}

/// Whether a code means "a signal arrived, nothing happened" and the call is
/// to be retried.
#[cfg(unix)]
pub fn error_is_interrupted(code: i32) -> bool {
    code == libc::EINTR
}
#[cfg(windows)]
pub fn error_is_interrupted(code: i32) -> bool {
    code == ws::WSAEINTR
}

/// Whether a code means the operation would have blocked.  A positive timeout
/// is carried by `SO_RCVTIMEO`/`SO_SNDTIMEO`, whose expiry WinSock reports as
/// `WSAETIMEDOUT` rather than as a would-block, so both spell the same
/// `TimeoutError` here.
#[cfg(unix)]
pub fn error_is_would_block(code: i32) -> bool {
    code == libc::EAGAIN || code == libc::EWOULDBLOCK
}
#[cfg(windows)]
pub fn error_is_would_block(code: i32) -> bool {
    code == ws::WSAEWOULDBLOCK || code == ws::WSAETIMEDOUT
}

/// `gai_strerror`.  The symbol is a header-level inline on Windows, where the
/// `EAI_*` codes are WSA error codes and the system message table describes
/// them.
#[cfg(unix)]
pub fn gai_strerror(code: libc::c_int) -> String {
    unsafe { std::ffi::CStr::from_ptr(libc::gai_strerror(code)) }
        .to_string_lossy()
        .into_owned()
}
#[cfg(all(windows, feature = "host_env"))]
pub fn gai_strerror(code: libc::c_int) -> String {
    rustpython_host_env::windows::format_error_message(Some(code as u32))
        .map(|message| {
            message
                .trim_end_matches(|c: char| c <= ' ' || c == '.')
                .to_string()
        })
        .unwrap_or_else(|| format!("Unknown error {code}"))
}
#[cfg(all(windows, not(feature = "host_env")))]
pub fn gai_strerror(code: libc::c_int) -> String {
    format!("Unknown error {code}")
}

// ── name lookups ──

/// `interp_func.py:24 gethostname` — the host's own name, as the opaque OS
/// string it is (`sethostname(2)` takes a plain `const char *`), so a byte or
/// code unit with no text spelling survives the caller's filesystem decode.
#[cfg(unix)]
pub fn hostname() -> std::io::Result<std::ffi::OsString> {
    use std::os::unix::ffi::OsStringExt;
    let mut buf = [0u8; 256];
    if unsafe { libc::gethostname(buf.as_mut_ptr() as *mut libc::c_char, buf.len()) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    Ok(std::ffi::OsString::from_vec(buf[..end].to_vec()))
}
#[cfg(all(windows, feature = "host_env"))]
pub fn hostname() -> std::io::Result<std::ffi::OsString> {
    init();
    Ok(rustpython_host_env::socket::hostname())
}
#[cfg(all(windows, not(feature = "host_env")))]
pub fn hostname() -> std::io::Result<std::ffi::OsString> {
    Err(std::io::Error::from(std::io::ErrorKind::Unsupported))
}

/// `interp_func.py:125-134 getprotobyname` — the `IPPROTO_*` number a protocol
/// name stands for, or `None` when the database does not name it.
#[cfg(unix)]
pub fn protocol_by_name(name: &std::ffi::CStr) -> Option<libc::c_int> {
    let entry = unsafe { libc::getprotobyname(name.as_ptr()) };
    (!entry.is_null()).then(|| unsafe { (*entry).p_proto })
}
#[cfg(windows)]
pub fn protocol_by_name(name: &std::ffi::CStr) -> Option<libc::c_int> {
    init();
    let entry = unsafe { ws::getprotobyname(name.as_ptr() as *const u8) };
    (!entry.is_null()).then(|| unsafe { (*entry).p_proto as libc::c_int })
}

/// `inet_aton` — the lenient dotted-quad parser, returning the four address
/// bytes in network order.  WinSock has no `inet_aton`; `inet_addr` accepts
/// the same spellings and reports failure as `INADDR_NONE`, which is the
/// substitution `socketmodule.c socket_inet_aton` makes without one.
#[cfg(unix)]
pub fn inet_aton(text: &std::ffi::CStr) -> Option<[u8; 4]> {
    unsafe extern "C" {
        #[link_name = "inet_aton"]
        fn c_inet_aton(cp: *const libc::c_char, inp: *mut libc::in_addr) -> libc::c_int;
    }
    let mut addr: libc::in_addr = unsafe { core::mem::zeroed() };
    (unsafe { c_inet_aton(text.as_ptr(), &mut addr) } != 0).then(|| addr.s_addr.to_ne_bytes())
}
#[cfg(windows)]
pub fn inet_aton(text: &std::ffi::CStr) -> Option<[u8; 4]> {
    // `INADDR_NONE` is both the failure report and the broadcast address, so
    // the one spelling that collides is answered before the call — the
    // substitution `rsocket.py`'s own `inet_addr` fallback makes.
    if text.to_bytes() == b"255.255.255.255" {
        return Some([0xff; 4]);
    }
    init();
    let addr = unsafe { ws::inet_addr(text.as_ptr() as *const u8) };
    (addr != ws::INADDR_NONE).then(|| addr.to_ne_bytes())
}

/// `inet_ntoa` — the dotted-quad spelling of four address bytes in network
/// order.
#[cfg(unix)]
pub fn inet_ntoa(packed: [u8; 4]) -> Option<String> {
    unsafe extern "C" {
        #[link_name = "inet_ntoa"]
        fn c_inet_ntoa(addr: libc::in_addr) -> *mut libc::c_char;
    }
    let addr = libc::in_addr {
        s_addr: u32::from_ne_bytes(packed),
    };
    let text = unsafe { c_inet_ntoa(addr) };
    (!text.is_null()).then(|| {
        unsafe { std::ffi::CStr::from_ptr(text) }
            .to_string_lossy()
            .into_owned()
    })
}
#[cfg(windows)]
pub fn inet_ntoa(packed: [u8; 4]) -> Option<String> {
    init();
    let addr = ws::IN_ADDR {
        S_un: ws::IN_ADDR_0 {
            S_addr: u32::from_ne_bytes(packed),
        },
    };
    let text = unsafe { ws::inet_ntoa(addr) };
    (!text.is_null()).then(|| {
        unsafe { std::ffi::CStr::from_ptr(text as *const libc::c_char) }
            .to_string_lossy()
            .into_owned()
    })
}

// ── descriptor inheritance ──

/// `rsocket.py:RSocket.__init__` closes every socket it creates over an exec
/// (PEP 446).  POSIX spells that as `FD_CLOEXEC`, Windows as the handle's
/// inherit flag.  Best-effort, like the `fcntl` call it replaces.
#[cfg(unix)]
pub fn set_cloexec(s: Socket) {
    unsafe { libc::fcntl(s, libc::F_SETFD, libc::FD_CLOEXEC) };
}
#[cfg(all(windows, feature = "host_env"))]
pub fn set_cloexec(s: Socket) {
    let _ = rustpython_host_env::socket::set_socket_inheritable(s, false);
}
#[cfg(all(windows, not(feature = "host_env")))]
pub fn set_cloexec(_s: Socket) {}

/// `interp_socket.py set_inheritable_w` — whether an exec'd child keeps this
/// socket.  POSIX inverts `FD_CLOEXEC`, Windows reads the handle's own flag.
#[cfg(unix)]
pub fn get_inheritable(s: Socket) -> std::io::Result<bool> {
    let flags = unsafe { libc::fcntl(s, libc::F_GETFD) };
    if flags < 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok((flags & libc::FD_CLOEXEC) == 0)
}
#[cfg(all(windows, feature = "host_env"))]
pub fn get_inheritable(s: Socket) -> std::io::Result<bool> {
    rustpython_host_env::nt::get_handle_inheritable(s as _)
}
#[cfg(all(windows, not(feature = "host_env")))]
pub fn get_inheritable(_s: Socket) -> std::io::Result<bool> {
    Err(std::io::Error::from(std::io::ErrorKind::Unsupported))
}

#[cfg(unix)]
pub fn set_inheritable(s: Socket, inheritable: bool) -> std::io::Result<()> {
    let flags = unsafe { libc::fcntl(s, libc::F_GETFD) };
    if flags < 0 {
        return Err(std::io::Error::last_os_error());
    }
    let wanted = if inheritable {
        flags & !libc::FD_CLOEXEC
    } else {
        flags | libc::FD_CLOEXEC
    };
    if wanted != flags && unsafe { libc::fcntl(s, libc::F_SETFD, wanted) } < 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}
#[cfg(all(windows, feature = "host_env"))]
pub fn set_inheritable(s: Socket, inheritable: bool) -> std::io::Result<()> {
    rustpython_host_env::socket::set_socket_inheritable(s, inheritable)
}
#[cfg(all(windows, not(feature = "host_env")))]
pub fn set_inheritable(_s: Socket, _inheritable: bool) -> std::io::Result<()> {
    Err(std::io::Error::from(std::io::ErrorKind::Unsupported))
}

// ── blocking mode and timeouts ──

/// `rsocket.py:RSocket.settimeout` — put a live socket into the blocking mode
/// a Python timeout asks for.  A negative `timeout` is the `None` sentinel
/// (block indefinitely), `0.0` is non-blocking, and a positive one blocks with
/// `SO_RCVTIMEO`/`SO_SNDTIMEO` bounding each wait.
///
/// POSIX carries the mode in the descriptor's `O_NONBLOCK` and the durations
/// as `struct timeval`; WinSock has `FIONBIO` and a `DWORD` of milliseconds.
#[cfg(unix)]
pub fn apply_timeout(s: Socket, timeout: f64) -> std::io::Result<()> {
    let flags = unsafe { libc::fcntl(s, libc::F_GETFL, 0) };
    if flags < 0 {
        return Err(std::io::Error::last_os_error());
    }
    // Bit-clear without unary `!` so the static analyzer accepts the helper
    // (the analyzer rejects bitwise-not on signed `c_int`).
    let new_flags = if timeout == 0.0 {
        flags | libc::O_NONBLOCK
    } else if (flags & libc::O_NONBLOCK) != 0 {
        flags - libc::O_NONBLOCK
    } else {
        flags
    };
    if new_flags != flags && unsafe { libc::fcntl(s, libc::F_SETFL, new_flags) } < 0 {
        return Err(std::io::Error::last_os_error());
    }
    let tv = if timeout > 0.0 {
        libc::timeval {
            tv_sec: timeout.trunc() as libc::time_t,
            tv_usec: ((timeout - timeout.trunc()) * 1_000_000.0).round() as libc::suseconds_t,
        }
    } else {
        libc::timeval {
            tv_sec: 0,
            tv_usec: 0,
        }
    };
    for option in [libc::SO_RCVTIMEO, libc::SO_SNDTIMEO] {
        let r = unsafe {
            libc::setsockopt(
                s,
                libc::SOL_SOCKET,
                option,
                &tv as *const _ as *const libc::c_void,
                core::mem::size_of::<libc::timeval>() as SockLen,
            )
        };
        if r != 0 {
            return Err(std::io::Error::last_os_error());
        }
    }
    Ok(())
}

#[cfg(windows)]
pub fn apply_timeout(s: Socket, timeout: f64) -> std::io::Result<()> {
    let mut nonblocking: u32 = u32::from(timeout == 0.0);
    if unsafe { ws::ioctlsocket(s, ws::FIONBIO, &mut nonblocking) } == ws::SOCKET_ERROR {
        return Err(last_error());
    }
    // `SO_RCVTIMEO` reads a millisecond count, and zero there means "no
    // timeout"; a sub-millisecond request still has to expire, so it waits the
    // shortest period the option can express.
    let millis: u32 = if timeout > 0.0 {
        ((timeout * 1000.0).round() as u64).clamp(1, u32::MAX as u64) as u32
    } else {
        0
    };
    for option in [ws::SO_RCVTIMEO, ws::SO_SNDTIMEO] {
        let r = unsafe {
            ws::setsockopt(
                s,
                ws::SOL_SOCKET,
                option,
                &millis as *const u32 as *const u8,
                core::mem::size_of::<u32>() as i32,
            )
        };
        if r == ws::SOCKET_ERROR {
            return Err(last_error());
        }
    }
    Ok(())
}

/// `RSocket._select(False)`: wait until a socket has something to read, for at
/// most `timeout_ms`.  Returns the call's result — positive when ready, `0` on
/// expiry, negative on failure — paired with the code a failure reported.
#[cfg(unix)]
pub fn poll_readable(s: Socket, timeout_ms: libc::c_int) -> (libc::c_int, i32) {
    let mut pollfd = libc::pollfd {
        fd: s,
        events: libc::POLLIN,
        revents: 0,
    };
    crate::module::thread::call_external_function(|| unsafe {
        libc::poll(&mut pollfd, 1, timeout_ms)
    })
}
#[cfg(windows)]
pub fn poll_readable(s: Socket, timeout_ms: libc::c_int) -> (libc::c_int, i32) {
    poll_one(s, ws::POLLIN, timeout_ms)
}

/// The other half of `RSocket._select`, which `internal_connect` waits on: a
/// connection started in non-blocking mode reports its outcome by making the
/// socket writable.  Windows-only because `SO_SNDTIMEO` already bounds a
/// POSIX connect.
///
/// `select` with the socket in `exceptfds` as well as `writefds`, which is how
/// `internal_select(connect=1)` waits: a refused connection makes the socket
/// readable and exceptional rather than writable, and `WSAPoll` did not wake
/// for one at all before Windows 10 version 2004 — the caller would read
/// `WSAETIMEDOUT` where the connection error belongs.
#[cfg(windows)]
pub fn poll_writable(s: Socket, timeout_ms: libc::c_int) -> (libc::c_int, i32) {
    let mut writefds = ws::FD_SET {
        fd_count: 1,
        ..Default::default()
    };
    writefds.fd_array[0] = s;
    let mut exceptfds = writefds;
    let timeout = ws::TIMEVAL {
        tv_sec: timeout_ms / 1000,
        tv_usec: (timeout_ms % 1000) * 1000,
    };
    let _blocked = crate::module::thread::before_external_block();
    // WinSock ignores `nfds`: its fd_set is a counted SOCKET array, not a
    // descriptor-indexed bitmap.
    let ready = unsafe {
        ws::select(
            0,
            std::ptr::null_mut(),
            &mut writefds,
            &mut exceptfds,
            &timeout,
        )
    };
    (ready, last_error_code())
}

#[cfg(windows)]
fn poll_one(s: Socket, events: i16, timeout_ms: libc::c_int) -> (libc::c_int, i32) {
    let mut pollfd = ws::WSAPOLLFD {
        fd: s,
        events,
        revents: 0,
    };
    let _blocked = crate::module::thread::before_external_block();
    let ready = unsafe { ws::WSAPoll(&mut pollfd, 1, timeout_ms) };
    (ready, last_error_code())
}

// ── address-structure accessors ──
//
// The fields these reach are the ones the two platforms spell differently:
// WinSock wraps `in_addr` / `in6_addr` / the IPv6 scope id in anonymous
// unions, so they cannot be read or written by name from shared code.

#[cfg(unix)]
pub fn sockaddr_in_get_addr(sin: &sockaddr_in) -> u32 {
    sin.sin_addr.s_addr
}
#[cfg(windows)]
pub fn sockaddr_in_get_addr(sin: &sockaddr_in) -> u32 {
    unsafe { sin.sin_addr.S_un.S_addr }
}

/// The IPv4 address in network byte order, as `inet_pton` writes it.
#[cfg(unix)]
pub fn sockaddr_in_set_addr(sin: &mut sockaddr_in, addr: u32) {
    sin.sin_addr.s_addr = addr;
}
#[cfg(windows)]
pub fn sockaddr_in_set_addr(sin: &mut sockaddr_in, addr: u32) {
    sin.sin_addr.S_un.S_addr = addr;
}

#[cfg(unix)]
pub fn sockaddr_in6_set_addr(sin6: &mut sockaddr_in6, addr: [u8; 16]) {
    sin6.sin6_addr.s6_addr = addr;
}
#[cfg(windows)]
pub fn sockaddr_in6_set_addr(sin6: &mut sockaddr_in6, addr: [u8; 16]) {
    sin6.sin6_addr.u.Byte = addr;
}

#[cfg(unix)]
pub fn sockaddr_in6_get_scope_id(sin6: &sockaddr_in6) -> u32 {
    sin6.sin6_scope_id
}
#[cfg(windows)]
pub fn sockaddr_in6_get_scope_id(sin6: &sockaddr_in6) -> u32 {
    unsafe { sin6.Anonymous.sin6_scope_id }
}

#[cfg(unix)]
pub fn sockaddr_in6_set_scope_id(sin6: &mut sockaddr_in6, scope_id: u32) {
    sin6.sin6_scope_id = scope_id;
}
#[cfg(windows)]
pub fn sockaddr_in6_set_scope_id(sin6: &mut sockaddr_in6, scope_id: u32) {
    sin6.Anonymous.sin6_scope_id = scope_id;
}

// ── the calls themselves ──
//
// One signature per call, taking the widths the shared body already works in:
// byte counts as `usize`, transfer results as `isize`.  WinSock's `int`-shaped
// prototypes are narrowed here, at the single place that knows the ceiling.

/// WinSock counts a transfer in an `int`, so a single call can move at most
/// `i32::MAX` bytes; a longer request is served over several calls, exactly as
/// a short read leaves the caller looping today.
#[cfg(windows)]
fn clamp_transfer_len(len: usize) -> i32 {
    len.min(i32::MAX as usize) as i32
}

#[cfg(unix)]
pub unsafe fn socket(family: libc::c_int, ty: libc::c_int, proto: libc::c_int) -> Socket {
    unsafe { libc::socket(family, ty, proto) }
}
#[cfg(windows)]
pub unsafe fn socket(family: libc::c_int, ty: libc::c_int, proto: libc::c_int) -> Socket {
    init();
    unsafe { ws::socket(family, ty, proto) }
}

#[cfg(unix)]
pub unsafe fn close(s: Socket) -> libc::c_int {
    unsafe { libc::close(s) }
}
#[cfg(windows)]
pub unsafe fn close(s: Socket) -> libc::c_int {
    unsafe { ws::closesocket(s) }
}

#[cfg(unix)]
pub unsafe fn bind(s: Socket, addr: *const sockaddr, len: SockLen) -> libc::c_int {
    unsafe { libc::bind(s, addr, len) }
}
#[cfg(windows)]
pub unsafe fn bind(s: Socket, addr: *const sockaddr, len: SockLen) -> libc::c_int {
    unsafe { ws::bind(s, addr, len) }
}

#[cfg(unix)]
pub unsafe fn connect(s: Socket, addr: *const sockaddr, len: SockLen) -> libc::c_int {
    unsafe { libc::connect(s, addr, len) }
}
#[cfg(windows)]
pub unsafe fn connect(s: Socket, addr: *const sockaddr, len: SockLen) -> libc::c_int {
    unsafe { ws::connect(s, addr, len) }
}

#[cfg(unix)]
pub unsafe fn listen(s: Socket, backlog: libc::c_int) -> libc::c_int {
    unsafe { libc::listen(s, backlog) }
}
#[cfg(windows)]
pub unsafe fn listen(s: Socket, backlog: libc::c_int) -> libc::c_int {
    unsafe { ws::listen(s, backlog) }
}

#[cfg(unix)]
pub unsafe fn accept(s: Socket, addr: *mut sockaddr, len: *mut SockLen) -> Socket {
    unsafe { libc::accept(s, addr, len) }
}
#[cfg(windows)]
pub unsafe fn accept(s: Socket, addr: *mut sockaddr, len: *mut SockLen) -> Socket {
    unsafe { ws::accept(s, addr, len) }
}

#[cfg(unix)]
pub unsafe fn send(
    s: Socket,
    buf: *const core::ffi::c_void,
    len: usize,
    flags: libc::c_int,
) -> isize {
    unsafe { libc::send(s, buf, len, flags) }
}
#[cfg(windows)]
pub unsafe fn send(
    s: Socket,
    buf: *const core::ffi::c_void,
    len: usize,
    flags: libc::c_int,
) -> isize {
    unsafe { ws::send(s, buf.cast(), clamp_transfer_len(len), flags) as isize }
}

#[cfg(unix)]
pub unsafe fn recv(
    s: Socket,
    buf: *mut core::ffi::c_void,
    len: usize,
    flags: libc::c_int,
) -> isize {
    unsafe { libc::recv(s, buf, len, flags) }
}
#[cfg(windows)]
pub unsafe fn recv(
    s: Socket,
    buf: *mut core::ffi::c_void,
    len: usize,
    flags: libc::c_int,
) -> isize {
    unsafe { ws::recv(s, buf.cast(), clamp_transfer_len(len), flags) as isize }
}

#[cfg(unix)]
pub unsafe fn sendto(
    s: Socket,
    buf: *const core::ffi::c_void,
    len: usize,
    flags: libc::c_int,
    addr: *const sockaddr,
    addrlen: SockLen,
) -> isize {
    unsafe { libc::sendto(s, buf, len, flags, addr, addrlen) }
}
#[cfg(windows)]
pub unsafe fn sendto(
    s: Socket,
    buf: *const core::ffi::c_void,
    len: usize,
    flags: libc::c_int,
    addr: *const sockaddr,
    addrlen: SockLen,
) -> isize {
    unsafe { ws::sendto(s, buf.cast(), clamp_transfer_len(len), flags, addr, addrlen) as isize }
}

#[cfg(unix)]
pub unsafe fn recvfrom(
    s: Socket,
    buf: *mut core::ffi::c_void,
    len: usize,
    flags: libc::c_int,
    addr: *mut sockaddr,
    addrlen: *mut SockLen,
) -> isize {
    unsafe { libc::recvfrom(s, buf, len, flags, addr, addrlen) }
}
#[cfg(windows)]
pub unsafe fn recvfrom(
    s: Socket,
    buf: *mut core::ffi::c_void,
    len: usize,
    flags: libc::c_int,
    addr: *mut sockaddr,
    addrlen: *mut SockLen,
) -> isize {
    unsafe { ws::recvfrom(s, buf.cast(), clamp_transfer_len(len), flags, addr, addrlen) as isize }
}

#[cfg(unix)]
pub unsafe fn getsockname(s: Socket, addr: *mut sockaddr, len: *mut SockLen) -> libc::c_int {
    unsafe { libc::getsockname(s, addr, len) }
}
#[cfg(windows)]
pub unsafe fn getsockname(s: Socket, addr: *mut sockaddr, len: *mut SockLen) -> libc::c_int {
    unsafe { ws::getsockname(s, addr, len) }
}

#[cfg(unix)]
pub unsafe fn getpeername(s: Socket, addr: *mut sockaddr, len: *mut SockLen) -> libc::c_int {
    unsafe { libc::getpeername(s, addr, len) }
}
#[cfg(windows)]
pub unsafe fn getpeername(s: Socket, addr: *mut sockaddr, len: *mut SockLen) -> libc::c_int {
    unsafe { ws::getpeername(s, addr, len) }
}

#[cfg(unix)]
pub unsafe fn getsockopt(
    s: Socket,
    level: libc::c_int,
    option: libc::c_int,
    value: *mut core::ffi::c_void,
    len: *mut SockLen,
) -> libc::c_int {
    unsafe { libc::getsockopt(s, level, option, value, len) }
}
#[cfg(windows)]
pub unsafe fn getsockopt(
    s: Socket,
    level: libc::c_int,
    option: libc::c_int,
    value: *mut core::ffi::c_void,
    len: *mut SockLen,
) -> libc::c_int {
    unsafe { ws::getsockopt(s, level, option, value.cast(), len) }
}

#[cfg(unix)]
pub unsafe fn setsockopt(
    s: Socket,
    level: libc::c_int,
    option: libc::c_int,
    value: *const core::ffi::c_void,
    len: SockLen,
) -> libc::c_int {
    unsafe { libc::setsockopt(s, level, option, value, len) }
}
#[cfg(windows)]
pub unsafe fn setsockopt(
    s: Socket,
    level: libc::c_int,
    option: libc::c_int,
    value: *const core::ffi::c_void,
    len: SockLen,
) -> libc::c_int {
    unsafe { ws::setsockopt(s, level, option, value.cast(), len) }
}

#[cfg(unix)]
pub unsafe fn shutdown(s: Socket, how: libc::c_int) -> libc::c_int {
    unsafe { libc::shutdown(s, how) }
}
#[cfg(windows)]
pub unsafe fn shutdown(s: Socket, how: libc::c_int) -> libc::c_int {
    unsafe { ws::shutdown(s, how) }
}

#[cfg(unix)]
pub unsafe fn getaddrinfo(
    node: *const libc::c_char,
    service: *const libc::c_char,
    hints: *const addrinfo,
    res: *mut *mut addrinfo,
) -> libc::c_int {
    unsafe { libc::getaddrinfo(node, service, hints, res) }
}
#[cfg(windows)]
pub unsafe fn getaddrinfo(
    node: *const libc::c_char,
    service: *const libc::c_char,
    hints: *const addrinfo,
    res: *mut *mut addrinfo,
) -> libc::c_int {
    init();
    unsafe { ws::getaddrinfo(node as *const u8, service as *const u8, hints, res) }
}

#[cfg(unix)]
pub unsafe fn freeaddrinfo(res: *mut addrinfo) {
    unsafe { libc::freeaddrinfo(res) }
}
#[cfg(windows)]
pub unsafe fn freeaddrinfo(res: *mut addrinfo) {
    unsafe { ws::freeaddrinfo(res) }
}

#[cfg(unix)]
pub unsafe fn getnameinfo(
    addr: *const sockaddr,
    addrlen: SockLen,
    host: *mut libc::c_char,
    hostlen: SockLen,
    service: *mut libc::c_char,
    servicelen: SockLen,
    flags: libc::c_int,
) -> libc::c_int {
    unsafe { libc::getnameinfo(addr, addrlen, host, hostlen, service, servicelen, flags) }
}
#[cfg(windows)]
pub unsafe fn getnameinfo(
    addr: *const sockaddr,
    addrlen: SockLen,
    host: *mut libc::c_char,
    hostlen: SockLen,
    service: *mut libc::c_char,
    servicelen: SockLen,
    flags: libc::c_int,
) -> libc::c_int {
    init();
    unsafe {
        ws::getnameinfo(
            addr,
            addrlen,
            host as *mut u8,
            hostlen as u32,
            service as *mut u8,
            servicelen as u32,
            flags,
        )
    }
}

// <arpa/inet.h>'s two address converters, which the libc crate does not
// declare on any unix target we ship.  Aliased so the wrappers below can keep
// the header's names.
#[cfg(unix)]
unsafe extern "C" {
    #[link_name = "inet_pton"]
    fn c_inet_pton(
        af: libc::c_int,
        src: *const libc::c_char,
        dst: *mut libc::c_void,
    ) -> libc::c_int;
    #[link_name = "inet_ntop"]
    fn c_inet_ntop(
        af: libc::c_int,
        src: *const libc::c_void,
        dst: *mut libc::c_char,
        size: libc::socklen_t,
    ) -> *const libc::c_char;
}

#[cfg(unix)]
pub unsafe fn inet_pton(
    family: libc::c_int,
    src: *const libc::c_char,
    dst: *mut core::ffi::c_void,
) -> libc::c_int {
    unsafe { c_inet_pton(family, src, dst as *mut libc::c_void) }
}
#[cfg(windows)]
pub unsafe fn inet_pton(
    family: libc::c_int,
    src: *const libc::c_char,
    dst: *mut core::ffi::c_void,
) -> libc::c_int {
    init();
    unsafe { ws::inet_pton(family, src as *const u8, dst) }
}

#[cfg(unix)]
pub unsafe fn inet_ntop(
    family: libc::c_int,
    src: *const core::ffi::c_void,
    dst: *mut libc::c_char,
    size: SockLen,
) -> *const libc::c_char {
    unsafe { c_inet_ntop(family, src as *const libc::c_void, dst, size) }
}
#[cfg(windows)]
pub unsafe fn inet_ntop(
    family: libc::c_int,
    src: *const core::ffi::c_void,
    dst: *mut libc::c_char,
    size: SockLen,
) -> *const libc::c_char {
    init();
    unsafe { ws::inet_ntop(family, src, dst as *mut u8, size as usize) as *const libc::c_char }
}

// ---------------------------------------------------------------------------
// The legacy `<netdb.h>` resolvers.
//
// `libc` 0.2.186 declares none of them, and the two records they answer with
// are not the same type on both platforms: `hostent`'s `h_addrtype` and
// `h_length` are `int` on POSIX and `short` in WinSock, and `servent` orders
// `s_proto` before `s_port` on Win64 and the other way round everywhere else.
// A single `#[repr(C)]` mirror would therefore be wrong on one side, so the
// record stays opaque and every field is read through an accessor.
// ---------------------------------------------------------------------------

/// The resolver's `hostent` record. Never constructed here — only the pointer
/// the resolver returns is ever held, and it points into process-global
/// storage that the next lookup overwrites.
#[cfg(unix)]
#[repr(C)]
#[allow(non_snake_case)]
pub struct Hostent {
    h_name: *const libc::c_char,
    h_aliases: *mut *mut libc::c_char,
    h_addrtype: libc::c_int,
    h_length: libc::c_int,
    h_addr_list: *mut *mut libc::c_char,
}
#[cfg(windows)]
pub type Hostent = ws::HOSTENT;

/// The resolver's `servent` record, held on the same terms as [`Hostent`].
#[cfg(unix)]
#[repr(C)]
#[allow(non_snake_case)]
pub struct Servent {
    s_name: *const libc::c_char,
    s_aliases: *mut *mut libc::c_char,
    s_port: libc::c_int,
    s_proto: *const libc::c_char,
}
#[cfg(windows)]
pub type Servent = ws::SERVENT;

#[cfg(unix)]
unsafe extern "C" {
    #[link_name = "gethostbyname"]
    fn c_gethostbyname(name: *const libc::c_char) -> *mut Hostent;
    #[link_name = "gethostbyaddr"]
    fn c_gethostbyaddr(
        addr: *const libc::c_void,
        len: libc::socklen_t,
        family: libc::c_int,
    ) -> *mut Hostent;
    #[link_name = "getservbyname"]
    fn c_getservbyname(name: *const libc::c_char, proto: *const libc::c_char) -> *mut Servent;
    #[link_name = "getservbyport"]
    fn c_getservbyport(port: libc::c_int, proto: *const libc::c_char) -> *mut Servent;
}

/// `rsocket._get_netdb_lock_thread`: the lookups below answer with a pointer
/// into one process-global record, so a second lookup on another thread
/// invalidates the first one's answer. Hold this across both the lookup and
/// the copying out of everything the caller needs.
pub fn netdb_lock() -> std::sync::MutexGuard<'static, ()> {
    static NETDB_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    NETDB_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Read one slot of a `char **` array the resolver owns.
///
/// Darwin packs these arrays at byte-aligned addresses — immediately after a C
/// string, for instance. C permits the resulting unaligned pointer load and
/// Rust references do not, so the slot is read as raw storage, matching
/// `rsocket.gethost_common`'s rffi access rather than manufacturing an aligned
/// reference.
pub unsafe fn pointer_at(array: *mut *mut libc::c_char, index: usize) -> *mut libc::c_char {
    unsafe { std::ptr::read_unaligned(array.add(index)) }
}

/// # Safety
/// `name` must be a valid NUL-terminated C string. The returned pointer is
/// borrowed from process-global storage — see [`netdb_lock`].
pub unsafe fn host_by_name(name: *const libc::c_char) -> *mut Hostent {
    #[cfg(unix)]
    {
        unsafe { c_gethostbyname(name) }
    }
    #[cfg(windows)]
    {
        init();
        unsafe { ws::gethostbyname(name as *const u8) }
    }
}

/// # Safety
/// `addr` must point to `len` readable bytes, and the answer is borrowed —
/// see [`host_by_name`].
pub unsafe fn host_by_addr(
    addr: *const libc::c_void,
    len: SockLen,
    family: libc::c_int,
) -> *mut Hostent {
    #[cfg(unix)]
    {
        unsafe { c_gethostbyaddr(addr, len, family) }
    }
    #[cfg(windows)]
    {
        init();
        unsafe { ws::gethostbyaddr(addr as *const u8, len, family) }
    }
}

/// # Safety
/// Both arguments must be valid NUL-terminated C strings or null, and the
/// answer is borrowed — see [`host_by_name`].
pub unsafe fn serv_by_name(name: *const libc::c_char, proto: *const libc::c_char) -> *mut Servent {
    #[cfg(unix)]
    {
        unsafe { c_getservbyname(name, proto) }
    }
    #[cfg(windows)]
    {
        init();
        unsafe { ws::getservbyname(name as *const u8, proto as *const u8) }
    }
}

/// # Safety
/// `proto` must be a valid NUL-terminated C string or null, and the answer is
/// borrowed — see [`host_by_name`].
pub unsafe fn serv_by_port(port: libc::c_int, proto: *const libc::c_char) -> *mut Servent {
    #[cfg(unix)]
    {
        unsafe { c_getservbyport(port, proto) }
    }
    #[cfg(windows)]
    {
        init();
        unsafe { ws::getservbyport(port, proto as *const u8) }
    }
}

/// # Safety
/// `h` must be a live answer from one of the host lookups above.
pub unsafe fn hostent_name(h: *mut Hostent) -> *const libc::c_char {
    unsafe { (*h).h_name as *const libc::c_char }
}

/// # Safety
/// See [`hostent_name`].
pub unsafe fn hostent_aliases(h: *mut Hostent) -> *mut *mut libc::c_char {
    unsafe { (*h).h_aliases as *mut *mut libc::c_char }
}

/// # Safety
/// See [`hostent_name`].
pub unsafe fn hostent_addr_type(h: *mut Hostent) -> libc::c_int {
    unsafe { (*h).h_addrtype as libc::c_int }
}

/// # Safety
/// See [`hostent_name`].
pub unsafe fn hostent_length(h: *mut Hostent) -> libc::c_int {
    unsafe { (*h).h_length as libc::c_int }
}

/// # Safety
/// See [`hostent_name`].
pub unsafe fn hostent_addr_list(h: *mut Hostent) -> *mut *mut libc::c_char {
    unsafe { (*h).h_addr_list as *mut *mut libc::c_char }
}

/// # Safety
/// `s` must be a live answer from one of the service lookups above.
pub unsafe fn servent_name(s: *mut Servent) -> *const libc::c_char {
    unsafe { (*s).s_name as *const libc::c_char }
}

/// The port as the record stores it: network byte order, in the low 16 bits.
///
/// # Safety
/// See [`servent_name`].
pub unsafe fn servent_port(s: *mut Servent) -> u16 {
    unsafe { (*s).s_port as u16 }
}
