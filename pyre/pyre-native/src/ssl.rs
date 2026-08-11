//! rustls backend for Python's `_ssl` module.
//!
//! This crate is deliberately outside the Charon/LLBC extraction.  The
//! interpreter owns each backend value through an opaque pointer and reaches
//! it only through these non-generic, non-inlined functions, exactly as it
//! reaches the native hash and zlib engines.  No Python object lives here.

use std::io::Cursor;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Once};
use std::time::{SystemTime, UNIX_EPOCH};

use rustls::client::ClientSessionStore;
use rustls::pki_types::{
    CertificateDer, CertificateRevocationListDer, PrivateKeyDer, PrivatePkcs8KeyDer,
};
use rustls::sign::CertifiedKey;
use x509_parser::prelude::FromDer;

static INSTALL_PROVIDER: Once = Once::new();

/// Install the process-wide rustls provider before constructing TLS state.
///
/// Rustls requires one provider for a process.  AWS-LC is selected by the
/// workspace feature because it is RustPython's tested rustls provider and its
/// name is already admitted by CPython's `test_ssl` backend-version check.
#[inline(never)]
pub fn ensure_provider() {
    INSTALL_PROVIDER.call_once(|| {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
    });
}

/// In-memory encrypted transport used by `SSLObject`.
///
/// Keeping the unread suffix as `(Vec, start)` avoids the repeated whole-buffer
/// shifts in RustPython's `Vec::drain(..n)` implementation.  Compaction happens
/// only after a substantial prefix has been consumed.
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

#[inline(never)]
pub fn memory_bio_new() -> *mut MemoryBio {
    Box::into_raw(Box::new(MemoryBio {
        buffer: Vec::new(),
        start: 0,
        eof_written: false,
    }))
}

/// # Safety
/// `bio` must be null or a pointer returned by [`memory_bio_new`] that has not
/// already been freed.
#[inline(never)]
pub unsafe fn memory_bio_free(bio: *mut MemoryBio) {
    if !bio.is_null() {
        unsafe { drop(Box::from_raw(bio)) };
    }
}

/// # Safety
/// `bio` must point to a live [`MemoryBio`].
#[inline(never)]
pub unsafe fn memory_bio_read(bio: *mut MemoryBio, size: usize) -> Vec<u8> {
    let bio = unsafe { &mut *bio };
    let count = size.min(bio.pending());
    let end = bio.start + count;
    let out = bio.buffer[bio.start..end].to_vec();
    bio.start = end;
    bio.compact();
    out
}

/// # Safety
/// `bio` must point to a live [`MemoryBio`].
#[inline(never)]
pub unsafe fn memory_bio_write(bio: *mut MemoryBio, data: &[u8]) -> Result<usize, &'static str> {
    let bio = unsafe { &mut *bio };
    if bio.eof_written {
        return Err("cannot write() after write_eof()");
    }
    bio.compact();
    bio.buffer.extend_from_slice(data);
    Ok(data.len())
}

/// # Safety
/// `bio` must point to a live [`MemoryBio`].
#[inline(never)]
pub unsafe fn memory_bio_write_eof(bio: *mut MemoryBio) {
    unsafe { (*bio).eof_written = true };
}

/// # Safety
/// `bio` must point to a live [`MemoryBio`].
#[inline(never)]
pub unsafe fn memory_bio_pending(bio: *const MemoryBio) -> usize {
    unsafe { (&*bio).pending() }
}

/// # Safety
/// `bio` must point to a live [`MemoryBio`].
#[inline(never)]
pub unsafe fn memory_bio_eof(bio: *const MemoryBio) -> bool {
    let bio = unsafe { &*bio };
    bio.eof_written && bio.pending() == 0
}

/// Mutable Python-visible SSL context settings plus rustls trust material.
/// Connection configs are built from this state when `_wrap_socket` or
/// `_wrap_bio` is called; rustls configs themselves are intentionally not
/// mutated after publication.
pub struct Context {
    protocol: i32,
    check_hostname: bool,
    verify_mode: i32,
    verify_flags: i32,
    options: u64,
    minimum_version: i32,
    maximum_version: i32,
    alpn_protocols: Vec<Vec<u8>>,
    roots: rustls::RootCertStore,
    root_der: Vec<Vec<u8>>,
    // OpenSSL's hashed CA directories are lookup sources, not eagerly loaded
    // stores.  Keep the directories themselves and materialize candidates in
    // each immutable rustls ClientConfig; the certificate actually selected
    // for a successful chain is published to `root_der` after the handshake.
    capaths: Vec<std::path::PathBuf>,
    crls: Vec<CertificateRevocationListDer<'static>>,
    cipher_suites: Option<Vec<rustls::SupportedCipherSuite>>,
    ecdh_curve: Option<EcdhCurve>,
    certified_keys: Vec<Arc<CertifiedKey>>,
    server_session_store: Arc<dyn rustls::server::StoresServerSessions>,
    server_ticketer: Arc<dyn rustls::server::ProducesTickets>,
    accept_count: AtomicUsize,
    session_hits: AtomicUsize,
}

#[derive(Clone, Copy)]
enum EcdhCurve {
    Secp256r1,
    Secp384r1,
    X25519,
}

pub const PROTOCOL_TLS: i32 = 2;
pub const PROTOCOL_TLS_CLIENT: i32 = 16;
pub const PROTOCOL_TLS_SERVER: i32 = 17;
pub const CERT_NONE: i32 = 0;
pub const CERT_OPTIONAL: i32 = 1;
pub const CERT_REQUIRED: i32 = 2;
pub const DEFAULT_OPTIONS: u64 =
    0x0000_0bfb | 0x0200_0000 | 0x0002_0000 | 0x0040_0000 | 0x0010_0000;

impl Context {
    fn new(protocol: i32) -> Result<Self, &'static str> {
        ensure_provider();
        if !matches!(
            protocol,
            PROTOCOL_TLS | PROTOCOL_TLS_CLIENT | PROTOCOL_TLS_SERVER | 5 | 6
        ) {
            return Err("invalid or unsupported protocol version");
        }
        let client = protocol == PROTOCOL_TLS_CLIENT;
        let server_ticketer = rustls::crypto::aws_lc_rs::Ticketer::new()
            .map_err(|_| "failed to initialize TLS session ticket encryption")?;
        Ok(Self {
            protocol,
            check_hostname: client,
            verify_mode: if client { CERT_REQUIRED } else { CERT_NONE },
            verify_flags: 32768,
            options: DEFAULT_OPTIONS,
            minimum_version: match protocol {
                5 => 0x303,
                6 => 0x304,
                _ => -2,
            },
            maximum_version: match protocol {
                5 => 0x303,
                6 => 0x304,
                _ => -1,
            },
            alpn_protocols: Vec::new(),
            roots: rustls::RootCertStore::empty(),
            root_der: Vec::new(),
            capaths: Vec::new(),
            crls: Vec::new(),
            cipher_suites: None,
            ecdh_curve: None,
            certified_keys: Vec::new(),
            server_session_store: rustls::server::ServerSessionMemoryCache::new(256),
            server_ticketer,
            accept_count: AtomicUsize::new(0),
            session_hits: AtomicUsize::new(0),
        })
    }
}

#[inline(never)]
pub unsafe fn context_session_stats(context: *const Context) -> (usize, usize) {
    let context = unsafe { &*context };
    (
        context.accept_count.load(Ordering::Relaxed),
        context.session_hits.load(Ordering::Relaxed),
    )
}

#[inline(never)]
pub fn context_new(protocol: i32) -> Result<*mut Context, &'static str> {
    Context::new(protocol).map(|context| Box::into_raw(Box::new(context)))
}

/// # Safety
/// `context` must be null or a pointer returned by [`context_new`] that has not
/// already been freed.
#[inline(never)]
pub unsafe fn context_free(context: *mut Context) {
    if !context.is_null() {
        unsafe { drop(Box::from_raw(context)) };
    }
}

macro_rules! context_scalar {
    ($get:ident, $set:ident, $field:ident, $ty:ty) => {
        #[inline(never)]
        pub unsafe fn $get(context: *const Context) -> $ty {
            unsafe { (*context).$field }
        }

        #[inline(never)]
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

/// Store the wire-format ALPN list after the interpreter has validated it.
///
/// # Safety
/// `context` must point to a live [`Context`].
#[inline(never)]
pub unsafe fn context_set_alpn(context: *mut Context, protocols: Vec<Vec<u8>>) {
    unsafe { (*context).alpn_protocols = protocols };
}

/// Add DER trust anchors, returning `(accepted, rejected)`.
///
/// # Safety
/// `context` must point to a live [`Context`].
#[inline(never)]
pub unsafe fn context_add_roots(
    context: *mut Context,
    certs: Vec<CertificateDer<'static>>,
) -> (usize, usize) {
    let context = unsafe { &mut *context };
    let mut accepted = 0;
    let mut rejected = 0;
    for cert in certs {
        match context.add_root(cert.as_ref().to_vec()) {
            Ok(true) => accepted += 1,
            Ok(false) => {}
            Err(_) => rejected += 1,
        }
    }
    (accepted, rejected)
}

pub type NativeResult<T> = Result<T, (i32, String)>;

fn io_error(error: std::io::Error) -> (i32, String) {
    (error.raw_os_error().unwrap_or(-1), error.to_string())
}

fn pem_error(message: impl std::fmt::Display) -> (i32, String) {
    (0, format!("[SSL] PEM routines: {message}"))
}

impl Context {
    /// Add one trust anchor while retaining its full DER exactly once.  The
    /// full certificate is needed for `get_ca_certs`; rustls intentionally
    /// stores only the parsed trust-anchor projection.
    fn add_root(&mut self, der: Vec<u8>) -> NativeResult<bool> {
        if self.root_der.iter().any(|known| known == &der) {
            return Ok(false);
        }
        self.roots
            .add(CertificateDer::from(der.clone()))
            .map_err(|error| pem_error(error))?;
        self.root_der.push(der);
        Ok(true)
    }
}

fn read_pem_certificates(data: &[u8]) -> NativeResult<Vec<CertificateDer<'static>>> {
    let mut cursor = Cursor::new(data);
    let certs = rustls_pemfile::certs(&mut cursor)
        .collect::<Result<Vec<_>, _>>()
        .map_err(pem_error)?;
    if certs.is_empty() {
        return Err(pem_error("no start line"));
    }
    Ok(certs)
}

fn read_private_key(data: &[u8], password: Option<&[u8]>) -> NativeResult<PrivateKeyDer<'static>> {
    if let Some(password) = password {
        use der::SecretDocument;
        use pkcs8::EncryptedPrivateKeyInfoRef;

        let pem = String::from_utf8_lossy(data);
        if let Some(start) = pem.find("-----BEGIN ENCRYPTED PRIVATE KEY-----") {
            let tail = &pem[start..];
            let end_marker = "-----END ENCRYPTED PRIVATE KEY-----";
            let end = tail
                .find(end_marker)
                .ok_or_else(|| pem_error("unterminated encrypted private key"))?
                + end_marker.len();
            let (_, document) = SecretDocument::from_pem(&tail[..end])
                .map_err(|error| pem_error(format!("bad encrypted private key: {error}")))?;
            let encrypted = EncryptedPrivateKeyInfoRef::try_from(document.as_bytes())
                .map_err(|error| pem_error(format!("bad encrypted private key: {error}")))?;
            let decrypted = encrypted
                .decrypt(password)
                .map_err(|error| pem_error(format!("bad decrypt: {error}")))?;
            return Ok(PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(
                decrypted.as_bytes().to_vec(),
            )));
        }
    }

    rustls_pemfile::private_key(&mut Cursor::new(data))
        .map_err(pem_error)?
        .ok_or_else(|| pem_error("no private key found"))
}

/// Load and validate the context's certificate/private-key pair.
///
/// The replacement is committed only after parsing, provider key loading, and
/// public/private key matching all succeed, so concurrent connection creation
/// never observes a half-updated pair.
///
/// # Safety
/// `context` must point to a live [`Context`].
#[inline(never)]
pub unsafe fn context_load_cert_chain(
    context: *mut Context,
    cert_path: &str,
    key_path: &str,
    password: Option<&[u8]>,
) -> NativeResult<()> {
    ensure_provider();
    let cert_data = std::fs::read(cert_path).map_err(io_error)?;
    let certs = read_pem_certificates(&cert_data)?;
    let key_data = std::fs::read(key_path).map_err(io_error)?;
    let key = read_private_key(&key_data, password)?;
    let signing_key = rustls::crypto::aws_lc_rs::sign::any_supported_type(&key)
        .map_err(|error| pem_error(format!("unsupported private key: {error}")))?;
    let certified = CertifiedKey::new(certs, signing_key);
    certified.keys_match().map_err(|_| {
        (
            0,
            "[SSL: KEY_VALUES_MISMATCH] key values mismatch".to_string(),
        )
    })?;
    let context = unsafe { &mut *context };
    let algorithm = certified.key.algorithm();
    let certified = Arc::new(certified);
    if let Some(existing) = context
        .certified_keys
        .iter_mut()
        .find(|known| known.key.algorithm() == algorithm)
    {
        *existing = certified;
    } else {
        context.certified_keys.push(certified);
    }
    Ok(())
}

fn parse_concatenated_der(mut data: &[u8]) -> NativeResult<Vec<Vec<u8>>> {
    let mut certs = Vec::new();
    while !data.is_empty() {
        let before = data.len();
        let (remaining, _) = x509_parser::parse_x509_certificate(data)
            .map_err(|error| pem_error(format!("not enough data: {error}")))?;
        let consumed = before - remaining.len();
        if consumed == 0 {
            return Err(pem_error("not enough data"));
        }
        certs.push(data[..consumed].to_vec());
        data = remaining;
    }
    Ok(certs)
}

/// # Safety
/// `context` must point to a live [`Context`].
#[inline(never)]
pub unsafe fn context_load_verify_file(context: *mut Context, path: &str) -> NativeResult<usize> {
    let data = std::fs::read(path).map_err(io_error)?;
    let items = rustls_pemfile::read_all(&mut Cursor::new(data))
        .collect::<Result<Vec<_>, _>>()
        .map_err(pem_error)?;
    let mut certificates = Vec::new();
    let mut crls = Vec::new();
    for item in items {
        match item {
            rustls_pemfile::Item::X509Certificate(cert) => {
                certificates.push(cert.as_ref().to_vec())
            }
            rustls_pemfile::Item::Crl(crl) => crls.push(crl),
            _ => {}
        }
    }
    if certificates.is_empty() && crls.is_empty() {
        return Err(pem_error("no start line"));
    }
    let added = unsafe { context_add_verify_der(context, certificates.into_iter())? };
    let context = unsafe { &mut *context };
    for crl in crls {
        if !context
            .crls
            .iter()
            .any(|known| known.as_ref() == crl.as_ref())
        {
            context.crls.push(crl);
        }
    }
    Ok(added)
}

/// Register an OpenSSL-style hashed certificate directory for lazy lookup.
/// The directory has already been validated by the interpreter boundary.
///
/// # Safety
/// `context` must point to a live [`Context`].
#[inline(never)]
pub unsafe fn context_add_verify_dir(context: *mut Context, path: &str) {
    let context = unsafe { &mut *context };
    let path = std::path::PathBuf::from(path);
    if !context.capaths.iter().any(|known| known == &path) {
        context.capaths.push(path);
    }
}

/// Publish the trust anchor selected from a lazy `capath` lookup.
///
/// # Safety
/// `context` must point to a live [`Context`].
#[inline(never)]
pub unsafe fn context_add_verified_root(context: *mut Context, der: Vec<u8>) -> NativeResult<bool> {
    unsafe { (&mut *context).add_root(der) }
}

/// Load PEM text or one/more concatenated DER certificates.
///
/// # Safety
/// `context` must point to a live [`Context`].
#[inline(never)]
pub unsafe fn context_load_verify_data(
    context: *mut Context,
    data: &[u8],
    pem: bool,
) -> NativeResult<usize> {
    let certs: Vec<Vec<u8>> = if pem {
        read_pem_certificates(data)?
            .into_iter()
            .map(|cert| cert.as_ref().to_vec())
            .collect()
    } else {
        parse_concatenated_der(data)?
    };
    unsafe { context_add_verify_der(context, certs.into_iter()) }
}

unsafe fn context_add_verify_der(
    context: *mut Context,
    certs: impl Iterator<Item = Vec<u8>>,
) -> NativeResult<usize> {
    let context = unsafe { &mut *context };
    let mut added = 0;
    for der in certs {
        added += usize::from(context.add_root(der)?);
    }
    Ok(added)
}

/// Load trust anchors from the platform provider (Keychain on macOS, native
/// certificate stores on Windows, discovered bundle/directories on Unix).
///
/// # Safety
/// `context` must point to a live [`Context`].
#[inline(never)]
pub unsafe fn context_load_native_roots(context: *mut Context) -> NativeResult<usize> {
    let result = rustls_native_certs::load_native_certs();
    let added = unsafe {
        context_add_verify_der(
            context,
            result.certs.into_iter().map(|cert| cert.as_ref().to_vec()),
        )?
    };
    if added == 0 && !result.errors.is_empty() {
        return Err(pem_error(&result.errors[0]));
    }
    Ok(added)
}

fn certificate_is_ca(der: &[u8]) -> bool {
    x509_parser::certificate::X509Certificate::from_der(der)
        .ok()
        .map(|(_, cert)| {
            cert.basic_constraints()
                .ok()
                .flatten()
                .is_some_and(|constraints| constraints.value.ca)
                // OpenSSL's X509_check_ca retains its legacy rule for a
                // self-issued X.509v1 trust anchor without BasicConstraints.
                || (cert.version().0 == 0 && cert.subject() == cert.issuer())
        })
        .unwrap_or(false)
}

/// Return `(all_x509, ca_x509)`.
///
/// # Safety
/// `context` must point to a live [`Context`].
#[inline(never)]
pub unsafe fn context_cert_store_stats(context: *const Context) -> (usize, usize) {
    let context = unsafe { &*context };
    (
        context.root_der.len(),
        context
            .root_der
            .iter()
            .filter(|der| certificate_is_ca(der))
            .count(),
    )
}

/// Full DER for CA certificates only.
///
/// # Safety
/// `context` must point to a live [`Context`].
#[inline(never)]
pub unsafe fn context_ca_certs(context: *const Context) -> Vec<Vec<u8>> {
    unsafe { &*context }
        .root_der
        .iter()
        .filter(|der| certificate_is_ca(der))
        .cloned()
        .collect()
}

type DistinguishedName = Vec<Vec<(String, String)>>;

struct SubjectAlternativeName {
    kind: &'static str,
    value: String,
    directory_name: DistinguishedName,
}

/// Owned projection of the X.509 fields exposed by CPython's private
/// `_test_decode_cert()` helper and by `SSLContext.get_ca_certs()`. Keeping
/// this projection native prevents x509-parser's borrowed type graph from
/// entering the translated interpreter.
pub struct DecodedCertificate {
    issuer: DistinguishedName,
    subject: DistinguishedName,
    not_after: String,
    not_before: String,
    serial_number: String,
    version: i32,
    ocsp: Vec<String>,
    ca_issuers: Vec<String>,
    crl_distribution_points: Vec<String>,
    subject_alt_names: Vec<SubjectAlternativeName>,
}

fn oid_attribute_name(oid: &str) -> String {
    match oid {
        "2.5.4.3" => "commonName".to_string(),
        "2.5.4.6" => "countryName".to_string(),
        "2.5.4.7" => "localityName".to_string(),
        "2.5.4.8" => "stateOrProvinceName".to_string(),
        "2.5.4.10" => "organizationName".to_string(),
        "2.5.4.11" => "organizationalUnitName".to_string(),
        "1.2.840.113549.1.9.1" => "emailAddress".to_string(),
        _ => oid.to_string(),
    }
}

fn decode_name(name: &x509_parser::x509::X509Name<'_>) -> DistinguishedName {
    name.iter()
        .map(|rdn| {
            rdn.iter()
                .map(|attribute| {
                    let oid = attribute.attr_type().to_id_string();
                    let value = attribute
                        .attr_value()
                        .as_str()
                        .map(str::to_string)
                        .unwrap_or_else(|_| {
                            String::from_utf8_lossy(attribute.attr_value().data).into_owned()
                        });
                    (oid_attribute_name(&oid), value)
                })
                .collect()
        })
        .collect()
}

fn format_certificate_time(value: &x509_parser::time::ASN1Time) -> String {
    let date = value.to_datetime();
    const MONTHS: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    format!(
        "{} {:>2} {:02}:{:02}:{:02} {:04} GMT",
        MONTHS[date.month() as usize - 1],
        date.day(),
        date.hour(),
        date.minute(),
        date.second(),
        date.year(),
    )
}

fn format_ip_address(ip: &[u8]) -> String {
    match ip.len() {
        4 => format!("{}.{}.{}.{}", ip[0], ip[1], ip[2], ip[3]),
        16 => ip
            .chunks_exact(2)
            .map(|part| format!("{:X}", u16::from_be_bytes([part[0], part[1]])))
            .collect::<Vec<_>>()
            .join(":"),
        _ => "<invalid>".to_string(),
    }
}

fn decode_certificate(der: &[u8]) -> NativeResult<DecodedCertificate> {
    use x509_parser::extensions::{DistributionPointName, GeneralName, ParsedExtension};
    use x509_parser::oid_registry::{
        OID_PKIX_AUTHORITY_INFO_ACCESS, OID_X509_EXT_CRL_DISTRIBUTION_POINTS,
    };

    let (_, cert) = x509_parser::parse_x509_certificate(der)
        .map_err(|error| pem_error(format!("failed to parse certificate: {error}")))?;
    let mut decoded = DecodedCertificate {
        issuer: decode_name(cert.issuer()),
        subject: decode_name(cert.subject()),
        not_after: format_certificate_time(&cert.validity().not_after),
        not_before: format_certificate_time(&cert.validity().not_before),
        serial_number: {
            let mut serial = cert.serial.to_str_radix(16).to_uppercase();
            if serial.len() % 2 != 0 {
                serial.insert(0, '0');
            }
            serial
        },
        version: cert.version().0 as i32 + 1,
        ocsp: Vec::new(),
        ca_issuers: Vec::new(),
        crl_distribution_points: Vec::new(),
        subject_alt_names: Vec::new(),
    };

    if let Ok(extensions) = cert.tbs_certificate.extensions_map() {
        if let Some(extension) = extensions.get(&OID_PKIX_AUTHORITY_INFO_ACCESS)
            && let ParsedExtension::AuthorityInfoAccess(access) = extension.parsed_extension()
        {
            for description in &access.accessdescs {
                if let GeneralName::URI(uri) = &description.access_location {
                    match description.access_method.to_id_string().as_str() {
                        "1.3.6.1.5.5.7.48.1" => decoded.ocsp.push((*uri).to_string()),
                        "1.3.6.1.5.5.7.48.2" => decoded.ca_issuers.push((*uri).to_string()),
                        _ => {}
                    }
                }
            }
        }
        if let Some(extension) = extensions.get(&OID_X509_EXT_CRL_DISTRIBUTION_POINTS)
            && let ParsedExtension::CRLDistributionPoints(points) = extension.parsed_extension()
        {
            for point in &points.points {
                if let Some(DistributionPointName::FullName(names)) = &point.distribution_point {
                    for name in names {
                        if let GeneralName::URI(uri) = name {
                            decoded.crl_distribution_points.push((*uri).to_string());
                        }
                    }
                }
            }
        }
    }

    if let Ok(Some(extension)) = cert.subject_alternative_name() {
        for name in &extension.value.general_names {
            let entry = match name {
                GeneralName::DNSName(value) => SubjectAlternativeName {
                    kind: "DNS",
                    value: (*value).to_string(),
                    directory_name: Vec::new(),
                },
                GeneralName::IPAddress(value) => SubjectAlternativeName {
                    kind: "IP Address",
                    value: format_ip_address(value),
                    directory_name: Vec::new(),
                },
                GeneralName::RFC822Name(value) => SubjectAlternativeName {
                    kind: "email",
                    value: (*value).to_string(),
                    directory_name: Vec::new(),
                },
                GeneralName::URI(value) => SubjectAlternativeName {
                    kind: "URI",
                    value: (*value).to_string(),
                    directory_name: Vec::new(),
                },
                GeneralName::OtherName(_, _) => SubjectAlternativeName {
                    kind: "othername",
                    value: "<unsupported>".to_string(),
                    directory_name: Vec::new(),
                },
                GeneralName::DirectoryName(value) => SubjectAlternativeName {
                    kind: "DirName",
                    value: String::new(),
                    directory_name: decode_name(value),
                },
                GeneralName::RegisteredID(value) => SubjectAlternativeName {
                    kind: "Registered ID",
                    value: value.to_id_string(),
                    directory_name: Vec::new(),
                },
                _ => continue,
            };
            decoded.subject_alt_names.push(entry);
        }
    }
    Ok(decoded)
}

#[inline(never)]
pub fn certificate_decode_der(der: &[u8]) -> NativeResult<*mut DecodedCertificate> {
    decode_certificate(der).map(|cert| Box::into_raw(Box::new(cert)))
}

#[inline(never)]
pub fn certificate_decode_file(path: &str) -> NativeResult<*mut DecodedCertificate> {
    let data = std::fs::read(path).map_err(io_error)?;
    let certs = read_pem_certificates(&data)?;
    certificate_decode_der(certs[0].as_ref())
}

/// # Safety
/// `cert` must be null or a live pointer returned by a certificate decoder.
#[inline(never)]
pub unsafe fn certificate_free(cert: *mut DecodedCertificate) {
    if !cert.is_null() {
        unsafe { drop(Box::from_raw(cert)) };
    }
}

macro_rules! certificate_string {
    ($name:ident, $field:ident) => {
        #[inline(never)]
        pub unsafe fn $name(cert: *const DecodedCertificate) -> String {
            unsafe { (*cert).$field.clone() }
        }
    };
}

certificate_string!(certificate_not_after, not_after);
certificate_string!(certificate_not_before, not_before);
certificate_string!(certificate_serial_number, serial_number);

#[inline(never)]
pub unsafe fn certificate_version(cert: *const DecodedCertificate) -> i32 {
    unsafe { (*cert).version }
}

fn decoded_name(cert: &DecodedCertificate, subject: bool) -> &DistinguishedName {
    if subject { &cert.subject } else { &cert.issuer }
}

#[inline(never)]
pub unsafe fn certificate_name_rdn_count(cert: *const DecodedCertificate, subject: bool) -> usize {
    decoded_name(unsafe { &*cert }, subject).len()
}

#[inline(never)]
pub unsafe fn certificate_name_attribute_count(
    cert: *const DecodedCertificate,
    subject: bool,
    rdn: usize,
) -> usize {
    decoded_name(unsafe { &*cert }, subject)[rdn].len()
}

#[inline(never)]
pub unsafe fn certificate_name_attribute_key(
    cert: *const DecodedCertificate,
    subject: bool,
    rdn: usize,
    attribute: usize,
) -> String {
    decoded_name(unsafe { &*cert }, subject)[rdn][attribute]
        .0
        .clone()
}

#[inline(never)]
pub unsafe fn certificate_name_attribute_value(
    cert: *const DecodedCertificate,
    subject: bool,
    rdn: usize,
    attribute: usize,
) -> String {
    decoded_name(unsafe { &*cert }, subject)[rdn][attribute]
        .1
        .clone()
}

fn certificate_urls(cert: &DecodedCertificate, kind: i32) -> &Vec<String> {
    match kind {
        0 => &cert.ocsp,
        1 => &cert.ca_issuers,
        _ => &cert.crl_distribution_points,
    }
}

#[inline(never)]
pub unsafe fn certificate_url_count(cert: *const DecodedCertificate, kind: i32) -> usize {
    certificate_urls(unsafe { &*cert }, kind).len()
}

#[inline(never)]
pub unsafe fn certificate_url(cert: *const DecodedCertificate, kind: i32, index: usize) -> String {
    certificate_urls(unsafe { &*cert }, kind)[index].clone()
}

#[inline(never)]
pub unsafe fn certificate_san_count(cert: *const DecodedCertificate) -> usize {
    unsafe { (*cert).subject_alt_names.len() }
}

fn decoded_san(cert: &DecodedCertificate, index: usize) -> &SubjectAlternativeName {
    &cert.subject_alt_names[index]
}

#[inline(never)]
pub unsafe fn certificate_san_kind(cert: *const DecodedCertificate, index: usize) -> &'static str {
    decoded_san(unsafe { &*cert }, index).kind
}

#[inline(never)]
pub unsafe fn certificate_san_value(cert: *const DecodedCertificate, index: usize) -> String {
    decoded_san(unsafe { &*cert }, index).value.clone()
}

#[inline(never)]
pub unsafe fn certificate_san_directory_rdn_count(
    cert: *const DecodedCertificate,
    index: usize,
) -> usize {
    decoded_san(unsafe { &*cert }, index).directory_name.len()
}

#[inline(never)]
pub unsafe fn certificate_san_directory_attribute_count(
    cert: *const DecodedCertificate,
    index: usize,
    rdn: usize,
) -> usize {
    decoded_san(unsafe { &*cert }, index).directory_name[rdn].len()
}

#[inline(never)]
pub unsafe fn certificate_san_directory_attribute_key(
    cert: *const DecodedCertificate,
    index: usize,
    rdn: usize,
    attribute: usize,
) -> String {
    decoded_san(unsafe { &*cert }, index).directory_name[rdn][attribute]
        .0
        .clone()
}

#[inline(never)]
pub unsafe fn certificate_san_directory_attribute_value(
    cert: *const DecodedCertificate,
    index: usize,
    rdn: usize,
    attribute: usize,
) -> String {
    decoded_san(unsafe { &*cert }, index).directory_name[rdn][attribute]
        .1
        .clone()
}

#[derive(Clone, Copy)]
struct CipherInfo {
    name: &'static str,
    protocol: &'static str,
    bits: i32,
    aead: bool,
    symmetric: &'static str,
    digest: &'static str,
    kea: &'static str,
    auth: &'static str,
}

const CIPHERS: &[CipherInfo] = &[
    CipherInfo {
        name: "TLS_AES_128_GCM_SHA256",
        protocol: "TLSv1.3",
        bits: 128,
        aead: true,
        symmetric: "aes-128-gcm",
        digest: "sha256",
        kea: "kx-any",
        auth: "auth-any",
    },
    CipherInfo {
        name: "TLS_AES_256_GCM_SHA384",
        protocol: "TLSv1.3",
        bits: 256,
        aead: true,
        symmetric: "aes-256-gcm",
        digest: "sha384",
        kea: "kx-any",
        auth: "auth-any",
    },
    CipherInfo {
        name: "TLS_CHACHA20_POLY1305_SHA256",
        protocol: "TLSv1.3",
        bits: 256,
        aead: true,
        symmetric: "chacha20-poly1305",
        digest: "sha256",
        kea: "kx-any",
        auth: "auth-any",
    },
    CipherInfo {
        name: "ECDHE-ECDSA-AES128-GCM-SHA256",
        protocol: "TLSv1.2",
        bits: 128,
        aead: true,
        symmetric: "aes-128-gcm",
        digest: "sha256",
        kea: "kx-ecdhe",
        auth: "auth-ecdsa",
    },
    CipherInfo {
        name: "ECDHE-ECDSA-AES256-GCM-SHA384",
        protocol: "TLSv1.2",
        bits: 256,
        aead: true,
        symmetric: "aes-256-gcm",
        digest: "sha384",
        kea: "kx-ecdhe",
        auth: "auth-ecdsa",
    },
    CipherInfo {
        name: "ECDHE-RSA-AES128-GCM-SHA256",
        protocol: "TLSv1.2",
        bits: 128,
        aead: true,
        symmetric: "aes-128-gcm",
        digest: "sha256",
        kea: "kx-ecdhe",
        auth: "auth-rsa",
    },
    CipherInfo {
        name: "ECDHE-RSA-AES256-GCM-SHA384",
        protocol: "TLSv1.2",
        bits: 256,
        aead: true,
        symmetric: "aes-256-gcm",
        digest: "sha384",
        kea: "kx-ecdhe",
        auth: "auth-rsa",
    },
];

#[inline(never)]
pub fn cipher_count() -> usize {
    CIPHERS.len()
}
#[inline(never)]
pub fn cipher_name(index: usize) -> &'static str {
    CIPHERS[index].name
}
#[inline(never)]
pub fn cipher_protocol(index: usize) -> &'static str {
    CIPHERS[index].protocol
}
#[inline(never)]
pub fn cipher_bits(index: usize) -> i32 {
    CIPHERS[index].bits
}
#[inline(never)]
pub fn cipher_aead(index: usize) -> bool {
    CIPHERS[index].aead
}
#[inline(never)]
pub fn cipher_symmetric(index: usize) -> &'static str {
    CIPHERS[index].symmetric
}
#[inline(never)]
pub fn cipher_digest(index: usize) -> &'static str {
    CIPHERS[index].digest
}
#[inline(never)]
pub fn cipher_kea(index: usize) -> &'static str {
    CIPHERS[index].kea
}
#[inline(never)]
pub fn cipher_auth(index: usize) -> &'static str {
    CIPHERS[index].auth
}

#[inline(never)]
pub fn validate_cipher_string(pattern: &str) -> Result<(), &'static str> {
    parse_cipher_string(pattern).map(|_| ())
}

fn cipher_pattern_matches(suite: rustls::SupportedCipherSuite, pattern: &str) -> bool {
    if suite.tls13().is_some() {
        return false;
    }
    let name = format!("{:?}", suite.suite());
    match pattern {
        "ALL" | "DEFAULT" | "HIGH" => true,
        "AES128" => name.contains("AES_128"),
        "AES256" => name.contains("AES_256"),
        "AESGCM" => name.contains("AES") && name.contains("GCM"),
        "CHACHA20" => name.contains("CHACHA20"),
        "ECDHE" | "KECDHE" => name.contains("ECDHE"),
        "ECDSA" | "AECDSA" => name.contains("ECDSA"),
        "RSA" | "ARSA" | "KRSA" => name.contains("RSA"),
        "NULL" | "ANULL" | "ENULL" => false,
        _ => {
            let compact_name = name.replace('_', "").replace('-', "");
            let compact_pattern = pattern.replace('_', "").replace('-', "");
            compact_name.contains(&compact_pattern)
        }
    }
}

fn parse_cipher_string(pattern: &str) -> Result<Vec<rustls::SupportedCipherSuite>, &'static str> {
    ensure_provider();
    let provider =
        rustls::crypto::CryptoProvider::get_default().expect("the _ssl provider is installed");
    let mut selected = Vec::new();
    let mut exclusions = Vec::new();
    for raw in pattern.split(':') {
        let token = raw.trim().to_ascii_uppercase();
        if token.is_empty() || token.starts_with('@') || token.starts_with('+') {
            continue;
        }
        if let Some(excluded) = token.strip_prefix('!') {
            exclusions.push(excluded.to_string());
            continue;
        }
        let parts: Vec<&str> = token.split('+').collect();
        for suite in &provider.cipher_suites {
            if parts
                .iter()
                .all(|part| cipher_pattern_matches(*suite, part))
                && !selected
                    .iter()
                    .any(|known: &rustls::SupportedCipherSuite| known.suite() == suite.suite())
            {
                selected.push(*suite);
            }
        }
    }
    selected.retain(|suite| {
        !exclusions
            .iter()
            .any(|pattern| cipher_pattern_matches(*suite, pattern))
    });
    if selected.is_empty() {
        Err("No cipher can be selected")
    } else {
        Ok(selected)
    }
}

/// Apply OpenSSL's TLS <= 1.2 cipher-list selection. TLS 1.3 suites remain
/// provider defaults, matching `SSL_CTX_set_cipher_list` semantics.
///
/// # Safety
/// `context` must point to a live [`Context`].
#[inline(never)]
pub unsafe fn context_set_cipher_list(
    context: *mut Context,
    pattern: &str,
) -> Result<(), &'static str> {
    unsafe { (*context).cipher_suites = Some(parse_cipher_string(pattern)?) };
    Ok(())
}

/// Restrict key exchange to the curve selected by SSLContext.set_ecdh_curve.
/// OpenSSL stores this choice directly on SSL_CTX; keep the equivalent state
/// on our native Context and apply it to each immutable rustls provider built
/// from that context.
///
/// # Safety
/// `context` must point to a live [`Context`].
#[inline(never)]
pub unsafe fn context_set_ecdh_curve(
    context: *mut Context,
    curve: &str,
) -> Result<(), &'static str> {
    let curve = match curve {
        "prime256v1" => EcdhCurve::Secp256r1,
        "secp384r1" => EcdhCurve::Secp384r1,
        "X25519" => EcdhCurve::X25519,
        _ => return Err("unknown elliptic curve name"),
    };
    unsafe { (*context).ecdh_curve = Some(curve) };
    Ok(())
}

#[derive(Debug)]
struct NoCertificateVerification;

impl rustls::client::danger::ServerCertVerifier for NoCertificateVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &rustls::pki_types::ServerName<'_>,
        _ocsp_response: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        signature: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        let provider =
            rustls::crypto::CryptoProvider::get_default().expect("the _ssl provider is installed");
        rustls::crypto::verify_tls12_signature(
            message,
            cert,
            signature,
            &provider.signature_verification_algorithms,
        )
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        signature: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        let provider =
            rustls::crypto::CryptoProvider::get_default().expect("the _ssl provider is installed");
        rustls::crypto::verify_tls13_signature(
            message,
            cert,
            signature,
            &provider.signature_verification_algorithms,
        )
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        rustls::crypto::CryptoProvider::get_default()
            .expect("the _ssl provider is installed")
            .signature_verification_algorithms
            .supported_schemes()
    }
}

/// Keep WebPKI's chain, time, purpose, and signature validation while making
/// Python's `SSLContext.check_hostname` an independent policy switch.
///
/// rustls 0.23's WebPkiServerVerifier validates the complete chain before its
/// final server-name check. We deliberately supply an unrelated valid DNS
/// name and suppress only the two typed name-mismatch outcomes. Unlike
/// RustPython's certificate-name extraction workaround, this neither parses
/// SAN/CN a second time nor invents wildcard normalization rules.
#[derive(Debug)]
struct ChainOnlyServerVerifier {
    inner: Arc<dyn rustls::client::danger::ServerCertVerifier>,
}

#[derive(Debug)]
struct PolicyServerVerifier {
    inner: Arc<dyn rustls::client::danger::ServerCertVerifier>,
    require_authority_key_identifier: bool,
    require_crl: bool,
    has_crl: bool,
}

impl rustls::client::danger::ServerCertVerifier for PolicyServerVerifier {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        intermediates: &[CertificateDer<'_>],
        server_name: &rustls::pki_types::ServerName<'_>,
        ocsp_response: &[u8],
        now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        if self.require_crl && !self.has_crl {
            return Err(rustls::Error::InvalidCertificate(
                rustls::CertificateError::UnknownRevocationStatus,
            ));
        }
        if self.require_authority_key_identifier {
            let (_, certificate) = x509_parser::parse_x509_certificate(end_entity.as_ref())
                .map_err(|_| {
                    rustls::Error::InvalidCertificate(rustls::CertificateError::BadEncoding)
                })?;
            let has_aki = certificate
                .extensions()
                .iter()
                .any(|extension| extension.oid.to_id_string() == "2.5.29.35");
            if !has_aki {
                return Err(rustls::Error::InvalidCertificate(
                    rustls::CertificateError::ApplicationVerificationFailure,
                ));
            }
        }
        self.inner
            .verify_server_cert(end_entity, intermediates, server_name, ocsp_response, now)
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        signature: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        self.inner.verify_tls12_signature(message, cert, signature)
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        signature: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        self.inner.verify_tls13_signature(message, cert, signature)
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        self.inner.supported_verify_schemes()
    }
}

impl rustls::client::danger::ServerCertVerifier for ChainOnlyServerVerifier {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        intermediates: &[CertificateDer<'_>],
        _server_name: &rustls::pki_types::ServerName<'_>,
        ocsp_response: &[u8],
        now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        let unrelated_name = rustls::pki_types::ServerName::try_from("pyre.invalid")
            .expect("the fixed chain-only verifier name is valid");
        match self.inner.verify_server_cert(
            end_entity,
            intermediates,
            &unrelated_name,
            ocsp_response,
            now,
        ) {
            Err(rustls::Error::InvalidCertificate(
                rustls::CertificateError::NotValidForName
                | rustls::CertificateError::NotValidForNameContext { .. },
            )) => Ok(rustls::client::danger::ServerCertVerified::assertion()),
            result => result,
        }
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        signature: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        self.inner.verify_tls12_signature(message, cert, signature)
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        signature: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        self.inner.verify_tls13_signature(message, cert, signature)
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        self.inner.supported_verify_schemes()
    }
}

fn is_capath_hash_name(name: &std::ffi::OsStr) -> bool {
    let Some(name) = name.to_str() else {
        return false;
    };
    let Some((hash, suffix)) = name.split_once('.') else {
        return false;
    };
    hash.len() == 8
        && hash.bytes().all(|byte| byte.is_ascii_hexdigit())
        && !suffix.is_empty()
        && suffix.bytes().all(|byte| byte.is_ascii_digit())
}

fn capath_certificates(context: &Context) -> Vec<Vec<u8>> {
    let mut paths = Vec::new();
    for directory in &context.capaths {
        let Ok(entries) = std::fs::read_dir(directory) else {
            continue;
        };
        for entry in entries.flatten() {
            if is_capath_hash_name(&entry.file_name()) {
                paths.push(entry.path());
            }
        }
    }
    paths.sort();

    let mut result = Vec::new();
    for path in paths {
        let Ok(data) = std::fs::read(path) else {
            continue;
        };
        let Ok(certificates) = read_pem_certificates(&data) else {
            continue;
        };
        for certificate in certificates {
            let der = certificate.as_ref().to_vec();
            if !result.iter().any(|known| known == &der) {
                result.push(der);
            }
        }
    }
    result
}

fn provider_for_context(context: &Context) -> Arc<rustls::crypto::CryptoProvider> {
    let mut provider = rustls::crypto::CryptoProvider::get_default()
        .expect("the _ssl provider is installed")
        .as_ref()
        .clone();
    if let Some(selected) = &context.cipher_suites {
        provider.cipher_suites.retain(|suite| {
            suite.tls13().is_some()
                || selected
                    .iter()
                    .any(|chosen| chosen.suite() == suite.suite())
        });
    }
    if let Some(curve) = context.ecdh_curve {
        provider.kx_groups = vec![match curve {
            EcdhCurve::Secp256r1 => rustls::crypto::aws_lc_rs::kx_group::SECP256R1,
            EcdhCurve::Secp384r1 => rustls::crypto::aws_lc_rs::kx_group::SECP384R1,
            EcdhCurve::X25519 => rustls::crypto::aws_lc_rs::kx_group::X25519,
        }];
    }
    Arc::new(provider)
}

#[derive(Debug)]
struct MultiCertResolver {
    keys: Vec<Arc<CertifiedKey>>,
}

impl MultiCertResolver {
    fn choose(&self, schemes: &[rustls::SignatureScheme]) -> Option<Arc<CertifiedKey>> {
        self.keys
            .iter()
            .find(|key| key.key.choose_scheme(schemes).is_some())
            .cloned()
    }
}

impl rustls::server::ResolvesServerCert for MultiCertResolver {
    fn resolve(&self, client_hello: rustls::server::ClientHello<'_>) -> Option<Arc<CertifiedKey>> {
        self.choose(client_hello.signature_schemes())
    }
}

impl rustls::client::ResolvesClientCert for MultiCertResolver {
    fn resolve(
        &self,
        _root_hint_subjects: &[&[u8]],
        signature_schemes: &[rustls::SignatureScheme],
    ) -> Option<Arc<CertifiedKey>> {
        self.choose(signature_schemes)
    }

    fn has_certs(&self) -> bool {
        !self.keys.is_empty()
    }
}

/// Per-connection rustls store. CPython only resumes a client connection when
/// an SSLSession is supplied explicitly, so unlike rustls' default config this
/// cache is not shared implicitly by every connection from one context.
#[derive(Debug)]
struct CapturingClientSessionStore {
    inner: rustls::client::ClientSessionMemoryCache,
    latest_tls12: Mutex<
        Option<(
            rustls::pki_types::ServerName<'static>,
            rustls::client::Tls12ClientSessionValue,
        )>,
    >,
    public_id: Mutex<Option<Vec<u8>>>,
    creation_time: Mutex<Option<u64>>,
}

impl CapturingClientSessionStore {
    fn new() -> Self {
        Self {
            inner: rustls::client::ClientSessionMemoryCache::new(8),
            latest_tls12: Mutex::new(None),
            public_id: Mutex::new(None),
            creation_time: Mutex::new(None),
        }
    }

    fn seed(&self, session: &NativeSession) {
        self.inner
            .set_tls12_session(session.server_name.clone(), session.value.clone());
        *self
            .latest_tls12
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) =
            Some((session.server_name.clone(), session.value.clone()));
        *self
            .public_id
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(session.id.clone());
        *self
            .creation_time
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(session.creation_time);
    }

    fn ensure_metadata(&self) {
        let mut id = self
            .public_id
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if id.is_none() {
            let mut bytes = vec![0u8; 32];
            let provider = rustls::crypto::CryptoProvider::get_default()
                .expect("the _ssl provider is installed");
            if provider.secure_random.fill(&mut bytes).is_ok() {
                *id = Some(bytes);
            }
        }
        drop(id);
        let mut created = self
            .creation_time
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if created.is_none() {
            *created = Some(
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            );
        }
    }

    fn snapshot(
        &self,
        context_identity: usize,
        config: Arc<rustls::ClientConfig>,
    ) -> Option<NativeSession> {
        let (server_name, value) = self
            .latest_tls12
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()?;
        self.ensure_metadata();
        Some(NativeSession {
            context_identity,
            config,
            server_name,
            value,
            id: self
                .public_id
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone()?,
            creation_time: self
                .creation_time
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .unwrap_or(0),
            timeout: 43_200,
        })
    }
}

impl rustls::client::ClientSessionStore for CapturingClientSessionStore {
    fn set_kx_hint(
        &self,
        server_name: rustls::pki_types::ServerName<'static>,
        group: rustls::NamedGroup,
    ) {
        self.inner.set_kx_hint(server_name, group);
    }

    fn kx_hint(
        &self,
        server_name: &rustls::pki_types::ServerName<'_>,
    ) -> Option<rustls::NamedGroup> {
        self.inner.kx_hint(server_name)
    }

    fn set_tls12_session(
        &self,
        server_name: rustls::pki_types::ServerName<'static>,
        value: rustls::client::Tls12ClientSessionValue,
    ) {
        self.inner
            .set_tls12_session(server_name.clone(), value.clone());
        *self
            .latest_tls12
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some((server_name, value));
        self.ensure_metadata();
    }

    fn tls12_session(
        &self,
        server_name: &rustls::pki_types::ServerName<'_>,
    ) -> Option<rustls::client::Tls12ClientSessionValue> {
        // Keep the one TLS 1.2 value directly on this per-connection store,
        // matching ClientSessionStore's documented cardinality.  The inner
        // cache remains responsible for KX hints and TLS 1.3 ticket queues.
        let session = self
            .latest_tls12
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
            .filter(|(known_name, _)| known_name == server_name)
            .map(|(_, value)| value.clone());
        session
    }

    fn remove_tls12_session(&self, server_name: &rustls::pki_types::ServerName<'static>) {
        self.inner.remove_tls12_session(server_name);
        let mut latest = self
            .latest_tls12
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if latest
            .as_ref()
            .is_some_and(|(known_name, _)| known_name == server_name)
        {
            *latest = None;
        }
    }

    fn insert_tls13_ticket(
        &self,
        server_name: rustls::pki_types::ServerName<'static>,
        value: rustls::client::Tls13ClientSessionValue,
    ) {
        self.inner.insert_tls13_ticket(server_name, value);
    }

    fn take_tls13_ticket(
        &self,
        server_name: &rustls::pki_types::ServerName<'static>,
    ) -> Option<rustls::client::Tls13ClientSessionValue> {
        self.inner.take_tls13_ticket(server_name)
    }
}

pub struct NativeSession {
    context_identity: usize,
    config: Arc<rustls::ClientConfig>,
    server_name: rustls::pki_types::ServerName<'static>,
    value: rustls::client::Tls12ClientSessionValue,
    id: Vec<u8>,
    creation_time: u64,
    timeout: u64,
}

#[inline(never)]
pub unsafe fn session_clone(session: *const NativeSession) -> *mut NativeSession {
    if session.is_null() {
        return std::ptr::null_mut();
    }
    let session = unsafe { &*session };
    Box::into_raw(Box::new(NativeSession {
        context_identity: session.context_identity,
        config: session.config.clone(),
        server_name: session.server_name.clone(),
        value: session.value.clone(),
        id: session.id.clone(),
        creation_time: session.creation_time,
        timeout: session.timeout,
    }))
}

#[inline(never)]
pub unsafe fn session_free(session: *mut NativeSession) {
    if !session.is_null() {
        unsafe { drop(Box::from_raw(session)) };
    }
}

#[inline(never)]
pub unsafe fn session_context_identity(session: *const NativeSession) -> usize {
    unsafe { (&*session).context_identity }
}

#[inline(never)]
pub unsafe fn session_id(session: *const NativeSession) -> Vec<u8> {
    unsafe { (&*session).id.clone() }
}

#[inline(never)]
pub unsafe fn session_creation_time(session: *const NativeSession) -> u64 {
    unsafe { (&*session).creation_time }
}

#[inline(never)]
pub unsafe fn session_timeout(session: *const NativeSession) -> u64 {
    unsafe { (&*session).timeout }
}

fn client_config(context: &Context) -> NativeResult<(rustls::ClientConfig, Vec<Vec<u8>>)> {
    let builder = rustls::ClientConfig::builder_with_provider(provider_for_context(context))
        .with_protocol_versions(enabled_versions(context)?)
        .map_err(|error| (0, format!("[SSL] invalid TLS configuration: {error}")))?;
    let deferred_roots = capath_certificates(context);
    let mut roots = context.roots.clone();
    for der in &deferred_roots {
        // A hashed directory is intentionally tolerant of stale or malformed
        // entries. Only usable X.509 anchors participate in rustls lookup.
        let _ = roots.add(CertificateDer::from(der.clone()));
    }
    let wants_client_cert = if context.verify_mode == CERT_NONE {
        builder
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(NoCertificateVerification))
    } else if !roots.is_empty() {
        let mut verifier_builder = rustls::client::WebPkiServerVerifier::builder(Arc::new(roots));
        if !context.crls.is_empty() {
            verifier_builder = verifier_builder.with_crls(context.crls.clone());
        }
        let verifier = verifier_builder.build().map_err(|error| {
            (
                0,
                format!("[SSL] cannot build certificate verifier: {error}"),
            )
        })?;
        let mut verifier: Arc<dyn rustls::client::danger::ServerCertVerifier> =
            Arc::new(PolicyServerVerifier {
                inner: verifier,
                require_authority_key_identifier: context.verify_flags & 32 != 0,
                require_crl: context.verify_flags & 12 != 0,
                has_crl: !context.crls.is_empty(),
            });
        if !context.check_hostname {
            verifier = Arc::new(ChainOnlyServerVerifier { inner: verifier });
        }
        builder
            .dangerous()
            .with_custom_certificate_verifier(verifier)
    } else {
        builder.with_root_certificates(roots)
    };
    let mut config = if !context.certified_keys.is_empty() {
        wants_client_cert.with_client_cert_resolver(Arc::new(MultiCertResolver {
            keys: context.certified_keys.clone(),
        }))
    } else {
        wants_client_cert.with_no_client_auth()
    };
    config.alpn_protocols = context.alpn_protocols.clone();
    Ok((config, deferred_roots))
}

fn server_config(context: &Context) -> NativeResult<rustls::ServerConfig> {
    if context.certified_keys.is_empty() {
        return Err((
            0,
            "[SSL] server-side connection requires a certificate and private key".to_string(),
        ));
    }
    let builder = rustls::ServerConfig::builder_with_provider(provider_for_context(context))
        .with_protocol_versions(enabled_versions(context)?)
        .map_err(|error| (0, format!("[SSL] invalid TLS configuration: {error}")))?;
    let wants_server_cert = if context.verify_mode == CERT_NONE || context.roots.is_empty() {
        builder.with_no_client_auth()
    } else {
        let verifier =
            rustls::server::WebPkiClientVerifier::builder(Arc::new(context.roots.clone()));
        let verifier = if context.verify_mode == CERT_OPTIONAL {
            verifier.allow_unauthenticated()
        } else {
            verifier
        }
        .build()
        .map_err(|error| (0, format!("[SSL] cannot build client verifier: {error}")))?;
        builder.with_client_cert_verifier(verifier)
    };
    let mut config = wants_server_cert.with_cert_resolver(Arc::new(MultiCertResolver {
        keys: context.certified_keys.clone(),
    }));
    config.alpn_protocols = context.alpn_protocols.clone();
    config.session_storage = context.server_session_store.clone();
    config.ticketer = context.server_ticketer.clone();
    Ok(config)
}

fn enabled_versions(
    context: &Context,
) -> NativeResult<&'static [&'static rustls::SupportedProtocolVersion]> {
    static TLS12_ONLY: &[&rustls::SupportedProtocolVersion] = &[&rustls::version::TLS12];
    static TLS13_ONLY: &[&rustls::SupportedProtocolVersion] = &[&rustls::version::TLS13];
    let minimum = context.minimum_version;
    let maximum = context.maximum_version;
    let tls12 = (minimum < 0 || minimum <= 0x303)
        && (maximum < 0 || maximum >= 0x303)
        && context.options & 0x0800_0000 == 0;
    let tls13 = (minimum < 0 || minimum <= 0x304)
        && (maximum < 0 || maximum >= 0x304)
        && context.options & 0x2000_0000 == 0;
    match (tls12, tls13) {
        (true, true) => Ok(rustls::DEFAULT_VERSIONS),
        (true, false) => Ok(TLS12_ONLY),
        (false, true) => Ok(TLS13_ONLY),
        (false, false) => Err((0, "[SSL] no protocols available".to_string())),
    }
}

pub const TLS_ERROR_SSL: i32 = 1;
pub const TLS_ERROR_WANT_READ: i32 = 2;
pub const TLS_ERROR_WANT_WRITE: i32 = 3;
pub const TLS_ERROR_ZERO_RETURN: i32 = 6;
pub const TLS_ERROR_EOF: i32 = 8;
/// Internal discriminator carrying an OpenSSL-compatible X509 verification
/// code to the interpreter without changing Python's public `errno` (which
/// remains SSL_ERROR_SSL == 1).
pub const TLS_ERROR_CERT_VERIFY_BASE: i32 = 1_000;

pub type TlsResult<T> = Result<T, (i32, String)>;

/// One protocol-level event for CPython's private `_msg_callback` hook.
/// Record framing stays in the TLS engine; the interpreter only turns these
/// inert values into Python callback arguments.
pub struct TlsMessageEvent {
    pub write: bool,
    pub version: u16,
    pub content_type: u16,
    pub message_type: u16,
    pub data: Vec<u8>,
}

#[derive(Default)]
struct TlsRecordObserver {
    records: Vec<u8>,
    handshakes: Vec<u8>,
    encrypted: bool,
}

impl TlsRecordObserver {
    fn observe(&mut self, bytes: &[u8], write: bool, events: &mut Vec<TlsMessageEvent>) {
        self.records.extend_from_slice(bytes);
        loop {
            if self.records.len() < 5 {
                return;
            }
            let content_type = self.records[0];
            let version = u16::from_be_bytes([self.records[1], self.records[2]]);
            let payload_len = u16::from_be_bytes([self.records[3], self.records[4]]) as usize;
            if self.records.len() < 5 + payload_len {
                return;
            }
            let record: Vec<u8> = self.records.drain(..5 + payload_len).collect();
            events.push(TlsMessageEvent {
                write,
                version,
                content_type: 0x100,
                message_type: content_type as u16,
                data: record[..5].to_vec(),
            });
            let payload = &record[5..];
            match content_type {
                20 => {
                    events.push(TlsMessageEvent {
                        write,
                        version,
                        content_type: 20,
                        message_type: 0x101,
                        data: payload.to_vec(),
                    });
                    // In TLS 1.2, protocol messages following CCS are
                    // encrypted. TLS 1.3 uses application-data records for
                    // encrypted handshake traffic and never enters here.
                    self.encrypted = true;
                    self.handshakes.clear();
                }
                21 if payload.len() >= 2 => events.push(TlsMessageEvent {
                    write,
                    version,
                    content_type: 21,
                    message_type: payload[1] as u16,
                    data: payload.to_vec(),
                }),
                22 if !self.encrypted => {
                    self.handshakes.extend_from_slice(payload);
                    loop {
                        if self.handshakes.len() < 4 {
                            break;
                        }
                        let message_len = ((self.handshakes[1] as usize) << 16)
                            | ((self.handshakes[2] as usize) << 8)
                            | self.handshakes[3] as usize;
                        if self.handshakes.len() < 4 + message_len {
                            break;
                        }
                        let message: Vec<u8> = self.handshakes.drain(..4 + message_len).collect();
                        events.push(TlsMessageEvent {
                            write,
                            version,
                            content_type: 22,
                            message_type: message[0] as u16,
                            data: message,
                        });
                    }
                }
                _ => {}
            }
        }
    }
}

fn rustls_error(error: impl std::fmt::Display) -> (i32, String) {
    (TLS_ERROR_SSL, format!("[SSL] {error}"))
}

#[allow(deprecated)] // rustls can still return the compatibility variant.
fn certificate_error_details(error: &rustls::CertificateError) -> (i32, &'static str) {
    use rustls::CertificateError;
    match error {
        CertificateError::Expired | CertificateError::ExpiredContext { .. } => {
            (10, "certificate has expired")
        }
        CertificateError::NotValidYet | CertificateError::NotValidYetContext { .. } => {
            (9, "certificate is not yet valid")
        }
        CertificateError::Revoked => (23, "certificate revoked"),
        CertificateError::UnknownIssuer => (20, "unable to get local issuer certificate"),
        CertificateError::BadSignature => (7, "certificate signature failure"),
        CertificateError::NotValidForName | CertificateError::NotValidForNameContext { .. } => {
            (62, "hostname mismatch")
        }
        CertificateError::InvalidPurpose | CertificateError::InvalidPurposeContext { .. } => {
            (26, "unsuitable certificate purpose")
        }
        CertificateError::BadEncoding => (5, "unable to decode certificate"),
        CertificateError::UnhandledCriticalExtension => (34, "unhandled critical extension"),
        CertificateError::UnknownRevocationStatus => (3, "unable to get certificate CRL"),
        CertificateError::ExpiredRevocationList
        | CertificateError::ExpiredRevocationListContext { .. } => (12, "CRL has expired"),
        CertificateError::UnsupportedSignatureAlgorithm
        | CertificateError::UnsupportedSignatureAlgorithmContext { .. }
        | CertificateError::UnsupportedSignatureAlgorithmForPublicKeyContext { .. } => {
            (7, "certificate signature failure")
        }
        CertificateError::InvalidOcspResponse => (50, "application verification failure"),
        CertificateError::ApplicationVerificationFailure | CertificateError::Other(_) => {
            (1, "certificate verify failed")
        }
        _ => (1, "certificate verify failed"),
    }
}

fn rustls_protocol_error(error: rustls::Error) -> (i32, String) {
    if let rustls::Error::InvalidCertificate(certificate_error) = &error {
        let (verify_code, verify_message) = certificate_error_details(certificate_error);
        return (
            TLS_ERROR_CERT_VERIFY_BASE + verify_code,
            format!("[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: {verify_message}"),
        );
    }
    if let rustls::Error::AlertReceived(alert) = &error {
        let reason = match alert {
            &rustls::AlertDescription::AccessDenied => "TLSV1_ALERT_ACCESS_DENIED",
            &rustls::AlertDescription::HandshakeFailure => "SSLV3_ALERT_HANDSHAKE_FAILURE",
            &rustls::AlertDescription::InternalError => "TLSV1_ALERT_INTERNAL_ERROR",
            &rustls::AlertDescription::ProtocolVersion => "TLSV1_ALERT_PROTOCOL_VERSION",
            &rustls::AlertDescription::UnknownCA => "TLSV1_ALERT_UNKNOWN_CA",
            _ => "TLSV1_ALERT_UNKNOWN",
        };
        return (
            TLS_ERROR_SSL,
            format!("[SSL: {reason}] received fatal TLS alert"),
        );
    }
    if matches!(
        error,
        rustls::Error::PeerIncompatible(rustls::PeerIncompatible::NoCipherSuitesInCommon)
    ) {
        return (
            TLS_ERROR_SSL,
            "[SSL: NO_SHARED_CIPHER] no shared cipher".to_string(),
        );
    }
    rustls_error(error)
}

#[inline(never)]
pub fn certificate_verify_message(code: i32) -> &'static str {
    match code {
        10 => "certificate has expired",
        9 => "certificate is not yet valid",
        23 => "certificate revoked",
        20 => "unable to get local issuer certificate",
        7 => "certificate signature failure",
        62 => "hostname mismatch",
        26 => "unsuitable certificate purpose",
        5 => "unable to decode certificate",
        34 => "unhandled critical extension",
        3 => "unable to get certificate CRL",
        12 => "CRL has expired",
        50 => "application verification failure",
        _ => "certificate verify failed",
    }
}

pub struct TlsConnection {
    inner: Option<rustls::Connection>,
    acceptor: Option<rustls::server::Acceptor>,
    accepted: Option<rustls::server::Accepted>,
    pending_tls: Vec<u8>,
    pending_tls_start: usize,
    deferred_roots: Vec<Vec<u8>>,
    verified_deferred_root: Option<Vec<u8>>,
    incoming_observer: TlsRecordObserver,
    outgoing_observer: TlsRecordObserver,
    message_events: Vec<TlsMessageEvent>,
    pending_received_tls: Vec<u8>,
    pending_received_tls_start: usize,
    client_config: Option<Arc<rustls::ClientConfig>>,
    client_session_store: Option<Arc<CapturingClientSessionStore>>,
    context_identity: usize,
    server_context: *const Context,
    server_hit_counted: bool,
}

impl TlsConnection {
    fn active_mut(&mut self) -> TlsResult<&mut rustls::Connection> {
        self.inner.as_mut().ok_or_else(|| {
            (
                TLS_ERROR_WANT_READ,
                "TLS server handshake is waiting for ClientHello configuration".to_string(),
            )
        })
    }

    fn fill_pending_tls(&mut self) -> TlsResult<()> {
        if self.pending_tls_start == self.pending_tls.len() {
            self.pending_tls.clear();
            self.pending_tls_start = 0;
        }
        let Some(inner) = self.inner.as_mut() else {
            return Ok(());
        };
        while inner.wants_write() {
            let before = self.pending_tls.len();
            inner
                .write_tls(&mut self.pending_tls)
                .map_err(rustls_error)?;
            self.outgoing_observer.observe(
                &self.pending_tls[before..],
                true,
                &mut self.message_events,
            );
            if self.pending_tls.len() == before {
                break;
            }
        }
        Ok(())
    }

    fn compact_received_tls(&mut self) {
        if self.pending_received_tls_start == self.pending_received_tls.len() {
            self.pending_received_tls.clear();
            self.pending_received_tls_start = 0;
        } else if self.pending_received_tls_start >= 16 * 1024 {
            self.pending_received_tls
                .drain(..self.pending_received_tls_start);
            self.pending_received_tls_start = 0;
        }
    }

    /// Feed as much queued ciphertext as rustls can currently accept.  Its
    /// plaintext queue intentionally applies backpressure at 16 KiB, so an
    /// unread tail remains owned by this connection until Python drains data.
    fn process_received_tls(&mut self) -> TlsResult<()> {
        loop {
            if self.pending_received_tls_start == self.pending_received_tls.len() {
                self.compact_received_tls();
                return Ok(());
            }
            let Some(inner) = self.inner.as_mut() else {
                return Ok(());
            };
            let mut cursor =
                Cursor::new(&self.pending_received_tls[self.pending_received_tls_start..]);
            let read = match inner.read_tls(&mut cursor) {
                Ok(0) => return Ok(()),
                Ok(read) => read,
                Err(error)
                    if error.kind() == std::io::ErrorKind::Other
                        && error.to_string() == "received plaintext buffer full" =>
                {
                    return Ok(());
                }
                Err(error) => return Err(rustls_error(error)),
            };
            self.pending_received_tls_start += read;
            inner.process_new_packets().map_err(rustls_protocol_error)?;
            self.note_server_resumption();
            self.compact_received_tls();
        }
    }

    fn note_server_resumption(&mut self) {
        if self.server_hit_counted || self.server_context.is_null() {
            return;
        }
        let Some(inner) = self.inner.as_ref() else {
            return;
        };
        if inner.is_handshaking() {
            return;
        }
        if inner.handshake_kind() == Some(rustls::HandshakeKind::Resumed) {
            unsafe { &*self.server_context }
                .session_hits
                .fetch_add(1, Ordering::Relaxed);
        }
        self.server_hit_counted = true;
    }

    fn process_acceptor_tls(&mut self) -> TlsResult<()> {
        loop {
            if self.pending_received_tls_start == self.pending_received_tls.len() {
                self.compact_received_tls();
                return Ok(());
            }
            let acceptor = self.acceptor.as_mut().ok_or_else(|| {
                (
                    TLS_ERROR_SSL,
                    "[SSL] server acceptor is no longer available".to_string(),
                )
            })?;
            let mut cursor =
                Cursor::new(&self.pending_received_tls[self.pending_received_tls_start..]);
            let read = acceptor.read_tls(&mut cursor).map_err(rustls_error)?;
            if read == 0 {
                return Ok(());
            }
            self.pending_received_tls_start += read;
            match acceptor.accept() {
                Ok(Some(accepted)) => {
                    self.accepted = Some(accepted);
                    self.acceptor = None;
                    self.compact_received_tls();
                    return Ok(());
                }
                Ok(None) => self.compact_received_tls(),
                Err((error, mut alert)) => {
                    let before = self.pending_tls.len();
                    let _ = alert.write_all(&mut self.pending_tls);
                    self.outgoing_observer.observe(
                        &self.pending_tls[before..],
                        true,
                        &mut self.message_events,
                    );
                    self.acceptor = None;
                    return Err(rustls_protocol_error(error));
                }
            }
        }
    }
}

/// Create one rustls state machine. It has no knowledge of Python sockets or
/// BIO objects; the interpreter owns transport policy and explicitly moves TLS
/// records through the primitive functions below.
#[inline(never)]
pub unsafe fn connection_new(
    context: *const Context,
    server_side: bool,
    server_hostname: Option<&str>,
    session: *const NativeSession,
) -> NativeResult<*mut TlsConnection> {
    ensure_provider();
    let context = unsafe { &*context };
    if server_side && context.protocol == PROTOCOL_TLS_CLIENT {
        return Err((
            0,
            "Cannot create a server socket with a PROTOCOL_TLS_CLIENT context".to_string(),
        ));
    }
    if !server_side && context.protocol == PROTOCOL_TLS_SERVER {
        return Err((
            0,
            "Cannot create a client socket with a PROTOCOL_TLS_SERVER context".to_string(),
        ));
    }
    let mut retained_client_config = None;
    let mut retained_client_store = None;
    let (connection, acceptor, deferred_roots) = if server_side {
        (None, Some(rustls::server::Acceptor::default()), Vec::new())
    } else {
        let name = match server_hostname {
            Some(hostname) => rustls::pki_types::ServerName::try_from(hostname.to_string())
                .map_err(|error| (0, format!("invalid server hostname: {error}")))?,
            None => rustls::pki_types::ServerName::IpAddress(
                std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST).into(),
            ),
        };
        let store = Arc::new(CapturingClientSessionStore::new());
        let (mut config, deferred_roots) = if session.is_null() {
            client_config(context)?
        } else {
            let session = unsafe { &*session };
            if session.context_identity != context as *const Context as usize {
                return Err((0, "Session refers to a different SSLContext.".to_string()));
            }
            if session.server_name == name {
                store.seed(session);
            }
            ((*session.config).clone(), Vec::new())
        };
        config.resumption = rustls::client::Resumption::store(store.clone());
        let config = Arc::new(config);
        retained_client_config = Some(config.clone());
        retained_client_store = Some(store);
        (
            Some(rustls::Connection::Client(
                rustls::ClientConnection::new(config, name).map_err(rustls_error)?,
            )),
            None,
            deferred_roots,
        )
    };
    Ok(Box::into_raw(Box::new(TlsConnection {
        inner: connection,
        acceptor,
        accepted: None,
        pending_tls: Vec::new(),
        pending_tls_start: 0,
        deferred_roots,
        verified_deferred_root: None,
        incoming_observer: TlsRecordObserver::default(),
        outgoing_observer: TlsRecordObserver::default(),
        message_events: Vec::new(),
        pending_received_tls: Vec::new(),
        pending_received_tls_start: 0,
        client_config: retained_client_config,
        client_session_store: retained_client_store,
        context_identity: context as *const Context as usize,
        server_context: std::ptr::null(),
        server_hit_counted: false,
    })))
}

/// # Safety
/// `connection` must be null or a live pointer returned by `connection_new`.
#[inline(never)]
pub unsafe fn connection_free(connection: *mut TlsConnection) {
    if !connection.is_null() {
        unsafe { drop(Box::from_raw(connection)) };
    }
}

/// Feed encrypted TLS records into rustls and process every complete packet.
/// Returns the number of bytes accepted from `data`.
///
/// # Safety
/// `connection` must point to a live connection.
#[inline(never)]
pub unsafe fn connection_receive_tls(
    connection: *mut TlsConnection,
    data: &[u8],
) -> TlsResult<usize> {
    let connection = unsafe { &mut *connection };
    connection
        .incoming_observer
        .observe(data, false, &mut connection.message_events);
    connection.pending_received_tls.extend_from_slice(data);
    if connection.inner.is_none() {
        if connection.accepted.is_some() {
            return Ok(data.len());
        }
        connection.process_acceptor_tls()?;
        return Ok(data.len());
    }

    let was_handshaking = connection
        .inner
        .as_ref()
        .is_some_and(|inner| inner.is_handshaking());
    connection.process_received_tls()?;
    let inner = connection.inner.as_ref().expect("active connection");
    if was_handshaking && !inner.is_handshaking() {
        connection.verified_deferred_root =
            matching_deferred_root(inner, &connection.deferred_roots);
        connection.deferred_roots.clear();
    }
    Ok(data.len())
}

/// Whether a server-side connection has parsed ClientHello and is waiting for
/// the interpreter to run the Python SNI callback and choose a context.
///
/// # Safety
/// `connection` must point to a live connection.
#[inline(never)]
pub unsafe fn connection_waiting_for_server_config(connection: *const TlsConnection) -> bool {
    unsafe { (&*connection).accepted.is_some() }
}

/// The SNI DNS name from the accepted ClientHello.
///
/// # Safety
/// `connection` must point to a live connection.
#[inline(never)]
pub unsafe fn connection_server_name(connection: *const TlsConnection) -> Option<String> {
    unsafe { (&*connection).accepted.as_ref() }
        .and_then(|accepted| accepted.client_hello().server_name().map(ToOwned::to_owned))
}

/// Resume an accepted server ClientHello with the context selected by the
/// Python SNI callback.
///
/// # Safety
/// Both pointers must refer to live native values.
#[inline(never)]
pub unsafe fn connection_accept_server(
    connection: *mut TlsConnection,
    context: *const Context,
) -> TlsResult<()> {
    let connection = unsafe { &mut *connection };
    let accepted = connection.accepted.take().ok_or_else(|| {
        (
            TLS_ERROR_SSL,
            "[SSL] no accepted ClientHello is waiting for configuration".to_string(),
        )
    })?;
    let mut config = server_config(unsafe { &*context })?;
    if !config.alpn_protocols.is_empty()
        && let Some(offered) = accepted.client_hello().alpn()
        && !offered
            .into_iter()
            .any(|protocol| config.alpn_protocols.iter().any(|known| known == protocol))
    {
        // OpenSSL completes a handshake without ALPN when both peers offered
        // ALPN but their lists are disjoint. rustls defaults to a fatal
        // no_application_protocol alert, so clear only this connection's
        // immutable config in the disjoint case.
        config.alpn_protocols.clear();
    }
    let config = Arc::new(config);
    match accepted.into_connection(config) {
        Ok(server) => {
            connection.inner = Some(rustls::Connection::Server(server));
            connection.server_context = context;
            unsafe { &*context }
                .accept_count
                .fetch_add(1, Ordering::Relaxed);
            connection.process_received_tls()
        }
        Err((error, mut alert)) => {
            let _ = alert.write_all(&mut connection.pending_tls);
            Err(rustls_protocol_error(error))
        }
    }
}

/// Reject an accepted ClientHello with a caller-selected fatal TLS alert.
/// Alerts at this stage are plaintext TLS records by protocol definition.
///
/// # Safety
/// `connection` must point to a live connection waiting for server config.
#[inline(never)]
pub unsafe fn connection_reject_server(
    connection: *mut TlsConnection,
    alert_description: u8,
) -> TlsResult<()> {
    let connection = unsafe { &mut *connection };
    if connection.accepted.take().is_none() {
        return Err((
            TLS_ERROR_SSL,
            "[SSL] no accepted ClientHello is waiting for rejection".to_string(),
        ));
    }
    let alert = [21, 3, 3, 0, 2, 2, alert_description];
    connection.pending_tls.extend_from_slice(&alert);
    connection
        .outgoing_observer
        .observe(&alert, true, &mut connection.message_events);
    Ok(())
}

/// Drain protocol events observed since the previous call.
///
/// # Safety
/// `connection` must point to a live connection.
#[inline(never)]
pub unsafe fn connection_take_message_events(
    connection: *mut TlsConnection,
) -> Vec<TlsMessageEvent> {
    std::mem::take(unsafe { &mut (&mut *connection).message_events })
}

fn matching_deferred_root(
    connection: &rustls::Connection,
    candidates: &[Vec<u8>],
) -> Option<Vec<u8>> {
    let peer_chain = connection.peer_certificates()?;
    let tail = peer_chain.last()?;
    let (_, tail) = x509_parser::parse_x509_certificate(tail.as_ref()).ok()?;
    candidates.iter().find_map(|candidate| {
        let (_, root) = x509_parser::parse_x509_certificate(candidate).ok()?;
        (tail.issuer() == root.subject()).then(|| candidate.clone())
    })
}

/// Return, once, the trust anchor selected from an OpenSSL-style lazy CA
/// directory after a successful client handshake.
///
/// # Safety
/// `connection` must point to a live connection.
#[inline(never)]
pub unsafe fn connection_take_verified_root(connection: *mut TlsConnection) -> Option<Vec<u8>> {
    unsafe { (&mut *connection).verified_deferred_root.take() }
}

/// Drain all currently generated encrypted TLS records.
///
/// # Safety
/// `connection` must point to a live connection.
#[inline(never)]
pub unsafe fn connection_take_tls(connection: *mut TlsConnection) -> TlsResult<Vec<u8>> {
    let connection = unsafe { &mut *connection };
    connection.fill_pending_tls()?;
    let output = connection.pending_tls[connection.pending_tls_start..].to_vec();
    connection.pending_tls.clear();
    connection.pending_tls_start = 0;
    Ok(output)
}

/// Copy, but do not consume, generated TLS records. Socket transports use
/// this together with `connection_consume_tls` so partial non-blocking sends
/// cannot lose bytes already drained from rustls.
#[inline(never)]
pub unsafe fn connection_peek_tls(connection: *mut TlsConnection) -> TlsResult<Vec<u8>> {
    let connection = unsafe { &mut *connection };
    connection.fill_pending_tls()?;
    Ok(connection.pending_tls[connection.pending_tls_start..].to_vec())
}

#[inline(never)]
pub unsafe fn connection_consume_tls(connection: *mut TlsConnection, count: usize) {
    let connection = unsafe { &mut *connection };
    connection.pending_tls_start =
        (connection.pending_tls_start + count).min(connection.pending_tls.len());
}

/// # Safety
/// `connection` must point to a live connection.
#[inline(never)]
pub unsafe fn connection_is_handshaking(connection: *const TlsConnection) -> bool {
    unsafe { (&*connection).inner.as_ref() }.is_none_or(|inner| inner.is_handshaking())
}

/// # Safety
/// `connection` must point to a live connection.
#[inline(never)]
pub unsafe fn connection_wants_read(connection: *const TlsConnection) -> bool {
    let connection = unsafe { &*connection };
    connection
        .inner
        .as_ref()
        .map_or(connection.acceptor.is_some(), |inner| inner.wants_read())
}

/// # Safety
/// `connection` must point to a live connection.
#[inline(never)]
pub unsafe fn connection_write_plain(
    connection: *mut TlsConnection,
    data: &[u8],
) -> TlsResult<usize> {
    use std::io::Write;
    unsafe { (&mut *connection).active_mut()? }
        .writer()
        .write(data)
        .map_err(rustls_error)
}

/// # Safety
/// `connection` must point to a live connection.
#[inline(never)]
pub unsafe fn connection_read_plain(
    connection: *mut TlsConnection,
    size: usize,
) -> TlsResult<Vec<u8>> {
    use std::io::Read;
    let connection = unsafe { &mut *connection };
    connection.process_received_tls()?;
    let mut output = vec![0; size];
    match connection.active_mut()?.reader().read(&mut output) {
        // A clean close_notify is EOF at the Python stream layer. CPython's
        // SSL_read wrapper returns b"" here; SSLZeroReturnError is reserved
        // for lower-level error reporting paths, not ordinary recv().
        Ok(0) => Ok(Vec::new()),
        Ok(read) => {
            output.truncate(read);
            Ok(output)
        }
        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => Err((
            TLS_ERROR_WANT_READ,
            "The operation did not complete (read)".to_string(),
        )),
        Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => Err((
            TLS_ERROR_EOF,
            "EOF occurred in violation of protocol".to_string(),
        )),
        Err(error) => Err(rustls_error(error)),
    }
}

/// # Safety
/// `connection` must point to a live connection.
#[inline(never)]
pub unsafe fn connection_send_close_notify(connection: *mut TlsConnection) {
    if let Some(inner) = unsafe { (&mut *connection).inner.as_mut() } {
        inner.send_close_notify();
    }
}

/// # Safety
/// `connection` must point to a live connection.
#[inline(never)]
pub unsafe fn connection_pending_plaintext(connection: *mut TlsConnection) -> usize {
    unsafe { (&mut *connection).inner.as_mut() }
        .and_then(|inner| inner.process_new_packets().ok())
        .map(|state| state.plaintext_bytes_to_read())
        .unwrap_or(0)
}

/// # Safety
/// `connection` must point to a live connection.
#[inline(never)]
pub unsafe fn connection_peer_closed(connection: *mut TlsConnection) -> bool {
    unsafe { (&mut *connection).inner.as_mut() }
        .and_then(|inner| inner.process_new_packets().ok())
        .map(|state| state.peer_has_closed())
        .unwrap_or(false)
}

/// # Safety
/// `connection` must point to a live connection.
#[inline(never)]
pub unsafe fn connection_alpn(connection: *const TlsConnection) -> Option<Vec<u8>> {
    unsafe { (&*connection).inner.as_ref() }
        .and_then(|inner| inner.alpn_protocol())
        .map(ToOwned::to_owned)
}

/// # Safety
/// `connection` must point to a live connection.
#[inline(never)]
pub unsafe fn connection_version(connection: *const TlsConnection) -> Option<&'static str> {
    match unsafe { (&*connection).inner.as_ref() }.and_then(|inner| inner.protocol_version()) {
        Some(rustls::ProtocolVersion::TLSv1_2) => Some("TLSv1.2"),
        Some(rustls::ProtocolVersion::TLSv1_3) => Some("TLSv1.3"),
        _ => None,
    }
}

/// # Safety
/// `connection` must point to a live connection.
#[inline(never)]
pub unsafe fn connection_peer_certificate(connection: *const TlsConnection) -> Option<Vec<u8>> {
    unsafe { (&*connection).inner.as_ref() }
        .and_then(|inner| inner.peer_certificates())
        .and_then(|certs| certs.first())
        .map(|cert| cert.as_ref().to_vec())
}

#[inline(never)]
pub unsafe fn connection_session_reused(connection: *const TlsConnection) -> Option<bool> {
    let inner = unsafe { (&*connection).inner.as_ref() }?;
    if inner.is_handshaking() {
        return None;
    }
    Some(inner.handshake_kind() == Some(rustls::HandshakeKind::Resumed))
}

/// Snapshot the actual opaque rustls TLS 1.2 resumption value captured for
/// this connection. The returned session keeps its ClientConfig alive so the
/// verifier/credential identity checks performed by rustls remain valid.
///
/// # Safety
/// `connection` must point to a live connection.
#[inline(never)]
pub unsafe fn connection_session(connection: *const TlsConnection) -> *mut NativeSession {
    let connection = unsafe { &*connection };
    let Some(inner) = connection.inner.as_ref() else {
        return std::ptr::null_mut();
    };
    if inner.is_handshaking() || inner.protocol_version() != Some(rustls::ProtocolVersion::TLSv1_2)
    {
        return std::ptr::null_mut();
    }
    let (Some(store), Some(config)) = (
        connection.client_session_store.as_ref(),
        connection.client_config.as_ref(),
    ) else {
        return std::ptr::null_mut();
    };
    store
        .snapshot(connection.context_identity, config.clone())
        .map(|session| Box::into_raw(Box::new(session)))
        .unwrap_or(std::ptr::null_mut())
}

/// Return stable channel-binding bytes shared by both peers for TLS 1.2.
/// rustls does not expose the Finished verify_data used by RFC 5929's
/// historical `tls-unique` construction, so use its standard exporter with a
/// private compatibility label. This preserves the binding's required peer
/// equality and per-session uniqueness without exposing key material.
///
/// # Safety
/// `connection` must point to a live connection.
#[inline(never)]
pub unsafe fn connection_tls_unique(connection: *const TlsConnection) -> Option<Vec<u8>> {
    let inner = unsafe { (&*connection).inner.as_ref() }?;
    if inner.is_handshaking() || inner.protocol_version()? != rustls::ProtocolVersion::TLSv1_2 {
        return None;
    }
    let mut output = vec![0u8; 12];
    inner
        .export_keying_material(&mut output, b"EXPORTER-pyre-tls-unique", None)
        .ok()?;
    Some(output)
}

fn openssl_cipher_name(suite: rustls::SupportedCipherSuite) -> String {
    let name = format!("{:?}", suite.suite());
    if suite.tls13().is_some() {
        // rustls' enum uses a `TLS13_` disambiguator while IANA/OpenSSL call
        // these suites `TLS_AES_*` and `TLS_CHACHA20_*`.
        return name
            .strip_prefix("TLS13_")
            .map(|suffix| format!("TLS_{suffix}"))
            .unwrap_or(name);
    }
    name.strip_prefix("TLS_")
        .unwrap_or(&name)
        .replace("_WITH_", "-")
        .replace("AES_128", "AES128")
        .replace("AES_256", "AES256")
        .replace('_', "-")
}

/// Negotiated OpenSSL-style cipher name and effective key size.
///
/// # Safety
/// `connection` must point to a live connection.
#[inline(never)]
pub unsafe fn connection_cipher(connection: *const TlsConnection) -> Option<(String, i32)> {
    let suite = unsafe { (&*connection).inner.as_ref() }?.negotiated_cipher_suite()?;
    let name = openssl_cipher_name(suite);
    let bits = if name.contains("AES128") || name.contains("AES_128") {
        128
    } else {
        256
    };
    Some((name, bits))
}
