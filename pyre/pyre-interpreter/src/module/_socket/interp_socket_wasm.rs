//! `_socket` on a target with no socket layer — PyPy: pypy/module/_socket/
//!
//! The module is published here the way a build whose C library carries the
//! headers but not the calls publishes it: the `socket` type, the numbers from
//! `<sys/socket.h>` and `<netdb.h>`, and none of the entry points that would
//! need a descriptor.  `socket.py` subclasses `_socket.socket` in its module
//! body and reaches everything else through `hasattr`, so that is the shape
//! that lets `import socket` — and `asyncio`, `socketserver`, `ftplib` behind
//! it — resolve, while an actual connection attempt reports that this platform
//! has none rather than pretending to open one.
//!
//! The numbers are musl's, which is wasi-libc's, matching the `sys.platform`
//! the guest reports.  `has_ipv6` stays false: carrying the `AF_INET6` number
//! is what a header does, and answering that the runtime supports the family
//! is a separate claim.

use pyre_object::PyObjectRef;

/// The `socket` type: real enough to subclass, empty of anything that would
/// need a descriptor behind it.
fn socket_type() -> PyObjectRef {
    static SOCKET_TYPE_OBJ: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *SOCKET_TYPE_OBJ.get_or_init(|| {
        let tp = crate::typedef::make_builtin_type("socket", init_socket_type);
        crate::typedef::mark_cpython_heap_type(tp, false);
        unsafe { pyre_object::typeobject::w_type_set_hasdict(tp, true) };
        tp as usize
    }) as PyObjectRef
}

fn init_socket_type(ns: PyObjectRef) {
    // Allocation is separate from initialisation here as it is everywhere
    // else: `socket.py`'s subclass calls `_socket.socket.__init__` itself, so
    // `__new__` must hand back an instance without having opened anything.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
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
        )
    };
    // The one call this type would have to make, and the one it cannot: there
    // is no descriptor for a socket to be, so construction reports that rather
    // than handing back an object whose every method would have to.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__init__",
            crate::make_builtin_function("__init__", |_args| {
                Err(crate::PyError::os_error_syscall(
                    crate::builtins::wasm_errno::ENOTSUP,
                    pyre_object::w_none(),
                ))
            }),
        )
    };
}

/// The two families the address converters carry a parser for, as the constant
/// table below spells them.
pub(super) const AF_INET: i32 = 2;
pub(super) const AF_INET6: i32 = 10;

/// The host this guest runs as, which is the `nodename` `uname` reports: wasi's
/// `gethostname` is written over `uname` and hands back that field.
const NODE_NAME: &str = "(none)";

/// `inet_aton` — the lenient parser, which reads one to four parts and gives
/// the last one every byte the earlier ones did not claim.  A part is decimal,
/// octal behind a `0`, or hexadecimal behind a `0x`.
pub(super) fn inet_aton(text: &std::ffi::CStr) -> Option<[u8; 4]> {
    let mut parts = Vec::new();
    for field in text.to_str().ok()?.split('.') {
        let (digits, radix) = match field.as_bytes() {
            [b'0', b'x' | b'X', rest @ ..] => (rest, 16),
            [b'0', rest @ ..] if !rest.is_empty() => (rest, 8),
            other => (other, 10),
        };
        let digits = std::str::from_utf8(digits).ok()?;
        parts.push(u32::from_str_radix(digits, radix).ok()?);
        if parts.len() > 4 {
            return None;
        }
    }
    // The last part carries the bytes the leading ones left: `127.1` is
    // `127.0.0.1`, and a bare number is the whole address.
    let leading = parts.len().checked_sub(1)?;
    let mut address: u32 = 0;
    for (index, part) in parts.iter().enumerate() {
        let width = if index == leading {
            32 - 8 * leading
        } else {
            8
        };
        if width < 32 && *part >= 1 << width {
            return None;
        }
        address |= part << (32 - 8 * index - width);
    }
    Some(address.to_be_bytes())
}

/// `inet_ntoa` — the dotted-quad spelling of four address bytes.
pub(super) fn inet_ntoa(packed: [u8; 4]) -> Option<String> {
    Some(format!(
        "{}.{}.{}.{}",
        packed[0], packed[1], packed[2], packed[3]
    ))
}

/// `inet_pton`'s two failures, which it reports through different channels: a
/// negative return leaves `EAFNOSUPPORT` behind, a zero says the string was
/// read and rejected.
pub(crate) enum PtonError {
    /// A family with no parser behind it.
    Family(i32),
    /// An address this family's parser refused.
    Address,
}

/// `inet_pton` — the strict parser.  Unlike `inet_aton` it reads exactly four
/// decimal octets, refuses a leading zero, and carries the IPv6 grammar.
pub(super) fn pton(family: i32, text: &std::ffi::CStr) -> Result<Vec<u8>, PtonError> {
    let bytes = text.to_bytes();
    match family {
        AF_INET => pton_v4(bytes).map(|a| a.to_vec()).ok_or(PtonError::Address),
        AF_INET6 => pton_v6(bytes).map(|a| a.to_vec()).ok_or(PtonError::Address),
        _ => Err(PtonError::Family(crate::builtins::wasm_errno::EAFNOSUPPORT)),
    }
}

/// One to three decimal digits with no redundant leading zero, which is the
/// only octet spelling the strict parser reads.
fn strict_octet(field: &[u8]) -> Option<u8> {
    if field.is_empty() || field.len() > 3 || (field.len() > 1 && field[0] == b'0') {
        return None;
    }
    let mut value: u32 = 0;
    for digit in field {
        value = value * 10 + u32::from(digit.checked_sub(b'0').filter(|d| *d < 10)?);
    }
    u8::try_from(value).ok()
}

fn pton_v4(text: &[u8]) -> Option<[u8; 4]> {
    let mut address = [0u8; 4];
    let mut fields = text.split(|b| *b == b'.');
    for slot in &mut address {
        *slot = strict_octet(fields.next()?)?;
    }
    fields.next().is_none().then_some(address)
}

fn pton_v6(text: &[u8]) -> Option<[u8; 16]> {
    // A trailing dotted quad occupies the last two groups, so it is taken off
    // first and the hexadecimal grammar reads what is left.
    let (head, tail) = match text.iter().position(|b| *b == b'.') {
        None => (text, None),
        Some(_) => {
            let colon = text.iter().rposition(|b| *b == b':')?;
            (&text[..colon], Some(pton_v4(&text[colon + 1..])?))
        }
    };
    let groups_wanted = if tail.is_some() { 6 } else { 8 };

    let (before, after) = match find_double_colon(head)? {
        None => (head, None),
        Some(at) => (&head[..at], Some(&head[at + 2..])),
    };
    let leading = hex_groups(before, groups_wanted)?;
    let trailing = match after {
        None => Vec::new(),
        Some(rest) => hex_groups(rest, groups_wanted - leading.len())?,
    };
    // Without `::` every group must be spelled; with it at least one must not.
    let elided = groups_wanted - leading.len() - trailing.len();
    if (after.is_none() && elided != 0) || (after.is_some() && elided == 0) {
        return None;
    }

    let mut address = [0u8; 16];
    let mut out = address.iter_mut();
    for group in leading
        .iter()
        .chain(std::iter::repeat_n(&0u16, elided))
        .chain(trailing.iter())
    {
        *out.next()? = (group >> 8) as u8;
        *out.next()? = *group as u8;
    }
    if let Some(quad) = tail {
        address[12..].copy_from_slice(&quad);
    }
    Some(address)
}

/// The offset of the one `::` a spelling may carry, or nothing when a second
/// one makes the address ambiguous.
fn find_double_colon(text: &[u8]) -> Option<Option<usize>> {
    let mut found = None;
    let mut index = 0;
    while index + 1 < text.len() {
        if text[index] == b':' && text[index + 1] == b':' {
            if found.is_some() {
                return None;
            }
            found = Some(index);
            index += 1;
        }
        index += 1;
    }
    Some(found)
}

/// Colon-separated groups of one to four hexadecimal digits.  An empty run is
/// no groups at all, which is what either side of a leading or trailing `::`
/// reads as.
fn hex_groups(text: &[u8], limit: usize) -> Option<Vec<u16>> {
    if text.is_empty() {
        return Some(Vec::new());
    }
    let mut groups = Vec::new();
    for field in text.split(|b| *b == b':') {
        if field.is_empty() || field.len() > 4 || groups.len() == limit {
            return None;
        }
        let mut value: u16 = 0;
        for digit in field {
            value = value * 16 + u16::from((*digit as char).to_digit(16)? as u8);
        }
        groups.push(value);
    }
    Some(groups)
}

/// `inet_ntop` — the canonical spelling of a packed address.
///
/// The IPv6 form is written out group by group and then has its longest run of
/// zero groups replaced by `::`, which is the rewrite musl performs and the
/// reason a run of one group is left spelled out.
pub(super) fn ntop(family: i32, packed: &[u8]) -> Option<String> {
    match family {
        AF_INET => inet_ntoa([packed[0], packed[1], packed[2], packed[3]]),
        AF_INET6 => Some(ntop_v6(packed)),
        _ => None,
    }
}

fn ntop_v6(packed: &[u8]) -> String {
    let group = |i: usize| (u16::from(packed[2 * i]) << 8) | u16::from(packed[2 * i + 1]);
    // An IPv4-mapped address keeps its last four bytes in dotted-quad form.
    let text = if packed[..12] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff] {
        format!(
            "{:x}:{:x}:{:x}:{:x}:{:x}:{:x}:{}.{}.{}.{}",
            group(0),
            group(1),
            group(2),
            group(3),
            group(4),
            group(5),
            packed[12],
            packed[13],
            packed[14],
            packed[15]
        )
    } else {
        (0..8)
            .map(|i| format!("{:x}", group(i)))
            .collect::<Vec<_>>()
            .join(":")
    };

    // The longest run of `:` and `0` that starts the string or starts at a
    // colon, taken only when it spans more than one zero group.
    let bytes = text.as_bytes();
    let (mut best, mut longest) = (0, 2);
    for start in 0..bytes.len() {
        if start != 0 && bytes[start] != b':' {
            continue;
        }
        let run = bytes[start..]
            .iter()
            .take_while(|b| **b == b':' || **b == b'0')
            .count();
        if run > longest {
            (best, longest) = (start, run);
        }
    }
    if longest <= 3 {
        return text;
    }
    format!("{}::{}", &text[..best], &text[best + longest..])
}

/// The `<sys/socket.h>` / `<netinet/in.h>` / `<netdb.h>` numbers, as musl
/// spells them.
///
/// `AF_UNIX` is left out because wasi-libc carries no `<sys/un.h>`, and the
/// name is read as a capability rather than as a number: `socketserver`
/// defines its four Unix server classes behind `hasattr(socket, "AF_UNIX")`
/// and `socket.socketpair` defaults to the family.
const CONSTANTS: &[(&str, i64)] = &[
    // ── Address families ──
    ("AF_UNSPEC", 0),
    ("AF_INET", 2),
    ("AF_INET6", 10),
    // ── Socket types ──
    ("SOCK_STREAM", 1),
    ("SOCK_DGRAM", 2),
    ("SOCK_RAW", 3),
    ("SOCK_RDM", 4),
    ("SOCK_SEQPACKET", 5),
    // ── Protocols ──
    ("IPPROTO_IP", 0),
    ("IPPROTO_ICMP", 1),
    ("IPPROTO_TCP", 6),
    ("IPPROTO_UDP", 17),
    ("IPPROTO_IPV6", 41),
    ("IPPROTO_RAW", 255),
    // ── Option levels and names ──
    ("SOL_SOCKET", 1),
    ("SO_DEBUG", 1),
    ("SO_REUSEADDR", 2),
    ("SO_TYPE", 3),
    ("SO_ERROR", 4),
    ("SO_DONTROUTE", 5),
    ("SO_BROADCAST", 6),
    ("SO_SNDBUF", 7),
    ("SO_RCVBUF", 8),
    ("SO_KEEPALIVE", 9),
    ("SO_OOBINLINE", 10),
    ("SO_LINGER", 13),
    ("SO_REUSEPORT", 15),
    ("SO_RCVLOWAT", 18),
    ("SO_SNDLOWAT", 19),
    ("SO_RCVTIMEO", 20),
    ("SO_SNDTIMEO", 21),
    ("SO_ACCEPTCONN", 30),
    ("TCP_NODELAY", 1),
    ("TCP_MAXSEG", 2),
    ("TCP_KEEPIDLE", 4),
    ("TCP_KEEPINTVL", 5),
    ("TCP_KEEPCNT", 6),
    ("IPV6_V6ONLY", 26),
    // ── shutdown(2) ──
    ("SHUT_RD", 0),
    ("SHUT_WR", 1),
    ("SHUT_RDWR", 2),
    // ── send / recv flags ──
    ("MSG_OOB", 0x0001),
    ("MSG_PEEK", 0x0002),
    ("MSG_DONTROUTE", 0x0004),
    ("MSG_CTRUNC", 0x0008),
    ("MSG_TRUNC", 0x0020),
    ("MSG_DONTWAIT", 0x0040),
    ("MSG_EOR", 0x0080),
    ("MSG_WAITALL", 0x0100),
    ("MSG_NOSIGNAL", 0x4000),
    // ── getaddrinfo / getnameinfo ──
    ("AI_PASSIVE", 0x0001),
    ("AI_CANONNAME", 0x0002),
    ("AI_NUMERICHOST", 0x0004),
    ("AI_V4MAPPED", 0x0008),
    ("AI_ALL", 0x0010),
    ("AI_ADDRCONFIG", 0x0020),
    ("AI_NUMERICSERV", 0x0400),
    ("NI_NUMERICHOST", 0x0001),
    ("NI_NUMERICSERV", 0x0002),
    ("NI_NOFQDN", 0x0004),
    ("NI_NAMEREQD", 0x0008),
    ("NI_DGRAM", 0x0010),
    ("NI_MAXHOST", 255),
    ("NI_MAXSERV", 32),
    ("EAI_BADFLAGS", -1),
    ("EAI_NONAME", -2),
    ("EAI_AGAIN", -3),
    ("EAI_FAIL", -4),
    ("EAI_NODATA", -5),
    ("EAI_FAMILY", -6),
    ("EAI_SOCKTYPE", -7),
    ("EAI_SERVICE", -8),
    ("EAI_ADDRFAMILY", -9),
    ("EAI_MEMORY", -10),
    ("EAI_SYSTEM", -11),
    ("EAI_OVERFLOW", -12),
    ("INADDR_ANY", 0x0000_0000),
    ("INADDR_BROADCAST", 0xffff_ffff_u32 as i64),
    ("INADDR_LOOPBACK", 0x7f00_0001),
    ("INADDR_NONE", 0xffff_ffff_u32 as i64),
    ("SOMAXCONN", 128),
];

/// The names `register_module` adds where there is no host socket layer to
/// build the rest out of.
pub(super) fn register_names(ns: PyObjectRef) {
    for (name, value) in CONSTANTS {
        crate::module_ns_store(ns, name, pyre_object::w_int_new(*value));
    }
    let socket_tp = socket_type();
    crate::module_ns_store(ns, "socket", socket_tp);
    crate::module_ns_store(ns, "SocketType", socket_tp);
    // `gethostname` needs no socket layer -- wasi answers it out of `uname` --
    // and `platform._node` calls it, so `platform.uname()` depends on it.
    crate::module_ns_store(
        ns,
        "gethostname",
        crate::make_builtin_function_with_arity(
            "gethostname",
            |_args| Ok(pyre_object::w_str_new(NODE_NAME)),
            0,
        ),
    );
}
