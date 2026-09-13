//! Wasm `_ssl` backend. rustls + aws-lc cannot build for
//! `wasm32-unknown-unknown` (`UnixTime::now` / native crypto), so this
//! module is the host-free surface: MemoryBIO works, TLS operations fail.

const UNAVAILABLE: &str = "TLS is not available on this platform";

pub const PROTOCOL_TLS: i32 = 2;
pub const PROTOCOL_TLS_CLIENT: i32 = 16;
pub const PROTOCOL_TLS_SERVER: i32 = 17;
pub const PROTOCOL_TLSV1: i32 = 3;
pub const PROTOCOL_TLSV1_1: i32 = 4;
pub const PROTOCOL_TLSV1_2: i32 = 5;
pub const PROTOCOL_TLSV1_3: i32 = 6;
pub const CERT_NONE: i32 = 0;
pub const CERT_OPTIONAL: i32 = 1;
pub const CERT_REQUIRED: i32 = 2;
pub const PROTO_TLSV1_2: i32 = 0x0303;
pub const PROTO_TLSV1_3: i32 = 0x0304;
pub const SSL3_RT_CHANGE_CIPHER_SPEC: i32 = 20;
pub const SSL3_RT_ALERT: i32 = 21;
pub const SSL3_RT_HANDSHAKE: i32 = 22;
pub const SSL3_RT_APPLICATION_DATA: i32 = 23;
pub const SSL3_RT_HEADER: i32 = 256;
pub const SSL3_MT_CHANGE_CIPHER_SPEC: i32 = 0x0101;
pub const TLS_ERROR_SSL: i32 = 1;
pub const TLS_ERROR_WANT_READ: i32 = 2;
pub const TLS_ERROR_WANT_WRITE: i32 = 3;
pub const TLS_ERROR_ZERO_RETURN: i32 = 6;
pub const TLS_ERROR_EOF: i32 = 8;
pub const TLS_ERROR_NO_MEMORY: i32 = 9;
pub const TLS_ERROR_CERT_VERIFY_BASE: i32 = 1_000;

pub type NativeResult<T> = Result<T, (i32, String)>;
pub type TlsResult<T> = Result<T, (i32, String)>;

bitflags::bitflags! {
    #[repr(transparent)]
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub struct SslOp: u64 {
        const ALL = 0x0000_0bfb;
        const NO_SSLV3 = 0x0200_0000;
        const NO_TLSV1 = 0x0400_0000;
        const NO_TLSV1_1 = 0x1000_0000;
        const NO_TLSV1_2 = 0x0800_0000;
        const NO_TLSV1_3 = 0x2000_0000;
        const NO_COMPRESSION = 0x0002_0000;
        const CIPHER_SERVER_PREFERENCE = 0x0040_0000;
        const ENABLE_MIDDLEBOX_COMPAT = 0x0010_0000;
        const NO_TICKET = 0x0000_4000;
        const LEGACY_SERVER_CONNECT = 0x4;
        const NO_RENEGOTIATION = 0x4000_0000;
        const IGNORE_UNEXPECTED_EOF = 0x80;
        const DEFAULT = Self::ALL.bits()
            | Self::NO_SSLV3.bits()
            | Self::NO_COMPRESSION.bits()
            | Self::CIPHER_SERVER_PREFERENCE.bits()
            | Self::ENABLE_MIDDLEBOX_COMPAT.bits();
    }
}

bitflags::bitflags! {
    #[repr(transparent)]
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub struct VerifyFlags: i32 {
        const DEFAULT = 0;
        const CRL_CHECK_LEAF = 4;
        const CRL_CHECK_CHAIN = 12;
        const X509_STRICT = 32;
        const ALLOW_PROXY_CERTS = 64;
        const X509_TRUSTED_FIRST = 32768;
        const X509_PARTIAL_CHAIN = 0x80000;
    }
}

fn unavailable<T>() -> NativeResult<T> {
    Err((TLS_ERROR_SSL, UNAVAILABLE.to_string()))
}

pub fn ensure_provider() {}

pub struct MemoryBio {
    buffer: Vec<u8>,
    start: usize,
    eof_written: bool,
}

impl MemoryBio {
    fn pending(&self) -> usize {
        self.buffer.len() - self.start
    }

    fn compact(&mut self) {
        if self.start == self.buffer.len() {
            self.buffer.clear();
            self.start = 0;
        } else if self.start >= 4096 && self.start * 2 >= self.buffer.len() {
            self.buffer.copy_within(self.start.., 0);
            self.buffer.truncate(self.buffer.len() - self.start);
            self.start = 0;
        }
    }
}

pub fn memory_bio_new() -> *mut MemoryBio {
    Box::into_raw(Box::new(MemoryBio {
        buffer: Vec::new(),
        start: 0,
        eof_written: false,
    }))
}

pub unsafe fn memory_bio_free(bio: *mut MemoryBio) {
    if !bio.is_null() {
        unsafe { drop(Box::from_raw(bio)) };
    }
}

pub unsafe fn memory_bio_read(bio: *mut MemoryBio, size: usize) -> Vec<u8> {
    let bio = unsafe { &mut *bio };
    let count = size.min(bio.pending());
    let end = bio.start + count;
    let out = bio.buffer[bio.start..end].to_vec();
    bio.start = end;
    bio.compact();
    out
}

pub unsafe fn memory_bio_write(bio: *mut MemoryBio, data: &[u8]) -> Result<usize, &'static str> {
    let bio = unsafe { &mut *bio };
    if bio.eof_written {
        return Err("cannot write() after write_eof()");
    }
    bio.compact();
    bio.buffer.extend_from_slice(data);
    Ok(data.len())
}

pub unsafe fn memory_bio_write_eof(bio: *mut MemoryBio) {
    unsafe { (*bio).eof_written = true };
}

pub unsafe fn memory_bio_pending(bio: *const MemoryBio) -> usize {
    unsafe { (*bio).pending() }
}

pub unsafe fn memory_bio_eof(bio: *const MemoryBio) -> bool {
    let bio = unsafe { &*bio };
    bio.eof_written && bio.pending() == 0
}

pub struct Context {
    identity: usize,
    protocol: i32,
    check_hostname: bool,
    verify_mode: i32,
    verify_flags: i32,
    options: u64,
    minimum_version: i32,
    maximum_version: i32,
}

static NEXT_CONTEXT_IDENTITY: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(1);

pub fn context_new(protocol: i32) -> Result<*mut Context, &'static str> {
    Ok(Box::into_raw(Box::new(Context {
        identity: NEXT_CONTEXT_IDENTITY.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        protocol,
        check_hostname: protocol == PROTOCOL_TLS_CLIENT,
        verify_mode: if protocol == PROTOCOL_TLS_CLIENT {
            CERT_REQUIRED
        } else {
            CERT_NONE
        },
        verify_flags: VerifyFlags::DEFAULT.bits(),
        options: SslOp::DEFAULT.bits(),
        minimum_version: -2,
        maximum_version: -1,
    })))
}

pub unsafe fn context_free(context: *mut Context) {
    if !context.is_null() {
        unsafe { drop(Box::from_raw(context)) };
    }
}

macro_rules! context_scalar {
    ($get:ident, $set:ident, $field:ident, $ty:ty) => {
        pub unsafe fn $get(context: *const Context) -> $ty {
            unsafe { (*context).$field }
        }
        pub unsafe fn $set(context: *mut Context, value: $ty) {
            unsafe { (*context).$field = value };
        }
    };
}

context_scalar!(context_protocol, context_set_protocol, protocol, i32);
context_scalar!(
    context_check_hostname,
    context_set_check_hostname,
    check_hostname,
    bool
);
context_scalar!(
    context_verify_mode,
    context_set_verify_mode,
    verify_mode,
    i32
);
context_scalar!(
    context_verify_flags,
    context_set_verify_flags,
    verify_flags,
    i32
);
context_scalar!(context_options, context_set_options, options, u64);
context_scalar!(
    context_minimum_version,
    context_set_minimum_version,
    minimum_version,
    i32
);
context_scalar!(
    context_maximum_version,
    context_set_maximum_version,
    maximum_version,
    i32
);

pub unsafe fn context_identity(context: *const Context) -> usize {
    unsafe { (*context).identity }
}
pub unsafe fn context_set_num_tickets(_context: *mut Context, _tickets: usize) {}
pub unsafe fn context_set_keylog_filename(
    _context: *mut Context,
    _path: Option<&std::path::Path>,
) -> NativeResult<()> {
    unavailable()
}
pub unsafe fn context_session_stats(_context: *const Context) -> (usize, usize) {
    (0, 0)
}
pub fn parse_length_prefixed_alpn(_data: &[u8]) -> Result<Vec<Vec<u8>>, &'static str> {
    Ok(Vec::new())
}
pub unsafe fn context_set_alpn(_context: *mut Context, _protocols: Vec<Vec<u8>>) {}
pub unsafe fn context_add_roots(_context: *mut Context, _ders: Vec<Vec<u8>>) {}
pub unsafe fn context_load_cert_chain(
    _context: *mut Context,
    _cert: &std::path::Path,
    _key: &std::path::Path,
    _password: Option<&[u8]>,
) -> NativeResult<bool> {
    unavailable()
}
pub unsafe fn context_load_verify_file(
    _context: *mut Context,
    _path: &std::path::Path,
) -> NativeResult<usize> {
    unavailable()
}
pub unsafe fn context_add_verify_dir(_context: *mut Context, _path: &std::path::Path) {}
pub unsafe fn context_add_verified_root(
    _context: *mut Context,
    _der: Vec<u8>,
) -> NativeResult<bool> {
    unavailable()
}
pub unsafe fn context_load_verify_data(
    _context: *mut Context,
    _data: &[u8],
    _pem: bool,
) -> NativeResult<usize> {
    unavailable()
}
pub fn default_verify_paths() -> (String, String) {
    (String::new(), String::new())
}
pub unsafe fn context_load_native_roots(_context: *mut Context) -> NativeResult<usize> {
    unavailable()
}
pub unsafe fn context_cert_store_stats(_context: *const Context) -> (usize, usize) {
    (0, 0)
}
pub unsafe fn context_ca_certs(_context: *const Context) -> Vec<Vec<u8>> {
    Vec::new()
}
pub unsafe fn context_cipher_enabled(_context: *const Context, _index: usize) -> bool {
    false
}
pub unsafe fn context_set_cipher_list(
    _context: *mut Context,
    _list: &str,
) -> Result<(), &'static str> {
    Err(UNAVAILABLE)
}
pub unsafe fn context_set_ecdh_curve(
    _context: *mut Context,
    _curve: &str,
) -> Result<(), &'static str> {
    Err(UNAVAILABLE)
}

pub struct DecodedCertificate;

pub fn certificate_decode_der(_der: &[u8]) -> NativeResult<*mut DecodedCertificate> {
    unavailable()
}
pub fn certificate_decode_file(_path: &std::path::Path) -> NativeResult<*mut DecodedCertificate> {
    unavailable()
}
pub fn certificate_subject_rfc2253(_der: &[u8]) -> NativeResult<String> {
    unavailable()
}
pub fn certificate_subject_hash(_der: &[u8]) -> NativeResult<i64> {
    unavailable()
}
pub unsafe fn certificate_free(cert: *mut DecodedCertificate) {
    if !cert.is_null() {
        unsafe { drop(Box::from_raw(cert)) };
    }
}
pub unsafe fn certificate_version(_cert: *const DecodedCertificate) -> i32 {
    0
}
pub unsafe fn certificate_name_rdn_count(
    _cert: *const DecodedCertificate,
    _subject: bool,
) -> usize {
    0
}
pub unsafe fn certificate_name_attribute_count(
    _cert: *const DecodedCertificate,
    _subject: bool,
    _rdn: usize,
) -> usize {
    0
}
pub unsafe fn certificate_name_attribute_key(
    _cert: *const DecodedCertificate,
    _subject: bool,
    _rdn: usize,
    _attribute: usize,
) -> String {
    String::new()
}
pub unsafe fn certificate_name_attribute_value(
    _cert: *const DecodedCertificate,
    _subject: bool,
    _rdn: usize,
    _attribute: usize,
) -> String {
    String::new()
}
pub unsafe fn certificate_url_count(_cert: *const DecodedCertificate, _kind: i32) -> usize {
    0
}
pub unsafe fn certificate_url(
    _cert: *const DecodedCertificate,
    _kind: i32,
    _index: usize,
) -> String {
    String::new()
}
pub unsafe fn certificate_san_count(_cert: *const DecodedCertificate) -> usize {
    0
}
pub unsafe fn certificate_san_kind(
    _cert: *const DecodedCertificate,
    _index: usize,
) -> &'static str {
    ""
}
pub unsafe fn certificate_san_value(_cert: *const DecodedCertificate, _index: usize) -> String {
    String::new()
}
pub unsafe fn certificate_san_directory_rdn_count(
    _cert: *const DecodedCertificate,
    _san: usize,
) -> usize {
    0
}
pub unsafe fn certificate_san_directory_attribute_count(
    _cert: *const DecodedCertificate,
    _san: usize,
    _rdn: usize,
) -> usize {
    0
}
pub unsafe fn certificate_san_directory_attribute_key(
    _cert: *const DecodedCertificate,
    _san: usize,
    _rdn: usize,
    _attribute: usize,
) -> String {
    String::new()
}
pub unsafe fn certificate_san_directory_attribute_value(
    _cert: *const DecodedCertificate,
    _san: usize,
    _rdn: usize,
    _attribute: usize,
) -> String {
    String::new()
}
pub unsafe fn certificate_not_after(_cert: *const DecodedCertificate) -> String {
    String::new()
}
pub unsafe fn certificate_not_before(_cert: *const DecodedCertificate) -> String {
    String::new()
}
pub unsafe fn certificate_serial_number(_cert: *const DecodedCertificate) -> String {
    String::new()
}

#[derive(Clone, Copy, Debug)]
pub struct OidInfo {
    pub nid: i32,
    pub short_name: &'static str,
    pub long_name: &'static str,
    pub oid: Option<&'static str>,
}

pub fn oid_by_nid(_nid: i32) -> Option<OidInfo> {
    None
}
pub fn oid_by_oid_string(_oid: &str) -> Option<OidInfo> {
    None
}
pub fn oid_by_name(_name: &str) -> Option<OidInfo> {
    None
}

pub fn cipher_count() -> usize {
    0
}
pub fn cipher_name(_index: usize) -> &'static str {
    ""
}
pub fn cipher_id(_index: usize) -> i64 {
    0
}
pub fn cipher_description(_index: usize) -> String {
    String::new()
}
pub fn cipher_protocol(_index: usize) -> &'static str {
    ""
}
pub fn cipher_bits(_index: usize) -> i32 {
    0
}
pub fn cipher_aead(_index: usize) -> bool {
    false
}
pub fn cipher_symmetric(_index: usize) -> &'static str {
    ""
}
pub fn cipher_digest(_index: usize) -> &'static str {
    ""
}
pub fn cipher_kea(_index: usize) -> &'static str {
    ""
}
pub fn cipher_auth(_index: usize) -> &'static str {
    ""
}
pub fn validate_cipher_string(_pattern: &str) -> Result<(), &'static str> {
    Err(UNAVAILABLE)
}
pub fn default_cipher_string() -> String {
    String::new()
}

pub struct NativeSession {
    context_identity: usize,
}

pub unsafe fn session_clone(session: *const NativeSession) -> *mut NativeSession {
    if session.is_null() {
        return std::ptr::null_mut();
    }
    Box::into_raw(Box::new(NativeSession {
        context_identity: unsafe { (*session).context_identity },
    }))
}
pub unsafe fn session_free(session: *mut NativeSession) {
    if !session.is_null() {
        unsafe { drop(Box::from_raw(session)) };
    }
}
pub unsafe fn session_context_identity(session: *const NativeSession) -> usize {
    if session.is_null() {
        0
    } else {
        unsafe { (*session).context_identity }
    }
}
pub unsafe fn session_id(_session: *const NativeSession) -> Vec<u8> {
    Vec::new()
}
pub unsafe fn session_creation_time(_session: *const NativeSession) -> u64 {
    0
}
pub unsafe fn session_timeout(_session: *const NativeSession) -> u64 {
    0
}

pub struct TlsMessageEvent {
    pub write: bool,
    pub version: u16,
    pub content_type: u16,
    pub message_type: u16,
    pub data: Vec<u8>,
}

pub fn certificate_verify_message(_code: i32) -> &'static str {
    UNAVAILABLE
}

pub struct TlsConnection;

pub unsafe fn connection_new(
    _context: *const Context,
    _server_side: bool,
    _server_hostname: Option<&str>,
    _session: *const NativeSession,
) -> NativeResult<*mut TlsConnection> {
    unavailable()
}
pub unsafe fn connection_free(connection: *mut TlsConnection) {
    if !connection.is_null() {
        unsafe { drop(Box::from_raw(connection)) };
    }
}
pub unsafe fn connection_receive_tls(
    _connection: *mut TlsConnection,
    _data: &[u8],
) -> TlsResult<usize> {
    Err((TLS_ERROR_SSL, UNAVAILABLE.to_string()))
}
pub unsafe fn connection_waiting_for_server_config(_connection: *const TlsConnection) -> bool {
    false
}
pub unsafe fn connection_server_name(_connection: *const TlsConnection) -> Option<String> {
    None
}
pub unsafe fn connection_accept_server(
    _connection: *mut TlsConnection,
    _context: *const Context,
) -> TlsResult<()> {
    Err((TLS_ERROR_SSL, UNAVAILABLE.to_string()))
}
pub unsafe fn connection_reject_server(
    _connection: *mut TlsConnection,
    _alert: u8,
) -> TlsResult<()> {
    Err((TLS_ERROR_SSL, UNAVAILABLE.to_string()))
}
pub unsafe fn connection_take_message_events(
    _connection: *mut TlsConnection,
) -> Vec<TlsMessageEvent> {
    Vec::new()
}
pub unsafe fn connection_tls_unique(_connection: *mut TlsConnection) -> Option<Vec<u8>> {
    None
}
pub unsafe fn connection_take_verified_root(_connection: *mut TlsConnection) -> Option<Vec<u8>> {
    None
}
pub unsafe fn connection_take_tls(_connection: *mut TlsConnection) -> TlsResult<Vec<u8>> {
    Err((TLS_ERROR_SSL, UNAVAILABLE.to_string()))
}
pub unsafe fn connection_peek_tls(_connection: *mut TlsConnection) -> TlsResult<Vec<u8>> {
    Err((TLS_ERROR_SSL, UNAVAILABLE.to_string()))
}
pub unsafe fn connection_consume_tls(_connection: *mut TlsConnection, _count: usize) {}
pub unsafe fn connection_is_handshaking(_connection: *const TlsConnection) -> bool {
    false
}
pub unsafe fn connection_wants_read(_connection: *const TlsConnection) -> bool {
    false
}
pub unsafe fn connection_write_plain(
    _connection: *mut TlsConnection,
    _data: &[u8],
) -> TlsResult<usize> {
    Err((TLS_ERROR_SSL, UNAVAILABLE.to_string()))
}
pub unsafe fn connection_read_plain(
    _connection: *mut TlsConnection,
    _size: usize,
) -> TlsResult<Vec<u8>> {
    Err((TLS_ERROR_SSL, UNAVAILABLE.to_string()))
}
pub unsafe fn connection_send_close_notify(_connection: *mut TlsConnection) {}
pub unsafe fn connection_pending_plaintext(_connection: *mut TlsConnection) -> usize {
    0
}
pub unsafe fn connection_peer_closed(_connection: *mut TlsConnection) -> bool {
    false
}
pub unsafe fn connection_alpn(_connection: *const TlsConnection) -> Option<Vec<u8>> {
    None
}
pub unsafe fn connection_version(_connection: *const TlsConnection) -> Option<&'static str> {
    None
}
pub unsafe fn connection_peer_certificate(_connection: *const TlsConnection) -> Option<Vec<u8>> {
    None
}
pub unsafe fn connection_peer_certificates(_connection: *const TlsConnection) -> Vec<Vec<u8>> {
    Vec::new()
}
pub unsafe fn connection_verified_certificates(_connection: *mut TlsConnection) -> Vec<Vec<u8>> {
    Vec::new()
}
pub unsafe fn connection_session_reused(_connection: *const TlsConnection) -> Option<bool> {
    None
}
pub unsafe fn connection_session(_connection: *const TlsConnection) -> *mut NativeSession {
    std::ptr::null_mut()
}
pub unsafe fn connection_cipher(_connection: *const TlsConnection) -> Option<(String, i32)> {
    None
}
