//! Rustls-backed `_ssl` module.
//!
//! The public policy and object wrappers remain the unmodified CPython
//! `ssl.py`.  This module supplies its low-level primitives while the actual
//! TLS engine lives in `pyre-native`, outside the translated interpreter.

use pyre_object::*;

const PROTOCOL_TLS: i32 = 2;
const PROTOCOL_TLS_CLIENT: i32 = 16;
const PROTOCOL_TLS_SERVER: i32 = 17;
const CERT_NONE: i32 = 0;
const CERT_OPTIONAL: i32 = 1;
const CERT_REQUIRED: i32 = 2;

/// The mapdict prefix is required because `ssl.SSLContext` is an app-level
/// subclass of this native type.  PyPy composes `MapdictStorageMixin` into
/// that allocation; pyre preserves the same native prefix.
#[crate::pyre_class("_ssl._SSLContext")]
#[derive(Default)]
pub struct W_SSLContext {
    pub map: *const u8,
    pub storage: *mut pyre_object::object_array::ItemsBlock,
    pub backend: *mut pyre_native::ssl::Context,
    pub sni_callback: PyObjectRef,
    pub msg_callback: PyObjectRef,
    pub keylog_filename: PyObjectRef,
    pub host_flags: i32,
    pub post_handshake_auth: bool,
    pub num_tickets: i32,
}

/// `ssl.MemoryBIO` is subclassable in CPython, so it uses the same mapdict
/// prefix rather than relying on a side table for subclass attributes.
#[crate::pyre_class("_ssl.MemoryBIO")]
#[derive(Default)]
pub struct W_MemoryBIO {
    pub map: *const u8,
    pub storage: *mut pyre_object::object_array::ItemsBlock,
    pub backend: *mut pyre_native::ssl::MemoryBio,
}

const _: () = assert!(
    std::mem::offset_of!(W_SSLContext, map)
        == std::mem::offset_of!(pyre_object::objectobject::W_ObjectObject, map),
    "W_SSLContext must keep W_ObjectObject's map offset"
);
const _: () = assert!(
    std::mem::offset_of!(W_SSLContext, storage)
        == std::mem::offset_of!(pyre_object::objectobject::W_ObjectObject, storage),
    "W_SSLContext must keep W_ObjectObject's storage offset"
);
const _: () = assert!(
    std::mem::offset_of!(W_MemoryBIO, map)
        == std::mem::offset_of!(pyre_object::objectobject::W_ObjectObject, map),
    "W_MemoryBIO must keep W_ObjectObject's map offset"
);
const _: () = assert!(
    std::mem::offset_of!(W_MemoryBIO, storage)
        == std::mem::offset_of!(pyre_object::objectobject::W_ObjectObject, storage),
    "W_MemoryBIO must keep W_ObjectObject's storage offset"
);

#[crate::pyre_class("_ssl.SSLSession")]
#[derive(Default)]
pub struct W_SSLSession {
    pub backend: *mut pyre_native::ssl::NativeSession,
}

/// One TLS state machine plus its Python-owned transport endpoints.  Rustls
/// remains opaque in `pyre-native`; these references preserve the same
/// per-object ownership shape as PyPy's `W_SSLObject`.
#[crate::pyre_class("_ssl._SSLSocket")]
#[derive(Default)]
pub struct W_SSLSocket {
    pub backend: *mut pyre_native::ssl::TlsConnection,
    pub context: PyObjectRef,
    pub socket: PyObjectRef,
    pub socket_send: PyObjectRef,
    pub socket_recv: PyObjectRef,
    pub incoming: PyObjectRef,
    pub outgoing: PyObjectRef,
    pub owner: PyObjectRef,
    pub server_hostname: PyObjectRef,
    pub server_side: bool,
    pub shutdown_started: bool,
    pub requested_session: *mut pyre_native::ssl::NativeSession,
}

#[crate::pyre_class("_ssl.Certificate")]
#[derive(Default)]
pub struct W_Certificate {
    pub der: PyObjectRef,
}

/// The `library` and `reason` an OpenSSL-shaped `[library: reason] text`
/// message carries.
fn split_library_reason(message: &str) -> Option<(&str, &str)> {
    let close = message.find(']')?;
    if !message.starts_with('[') {
        return None;
    }
    message[1..close].split_once(": ")
}

fn set_library_reason(exception: PyObjectRef, message: &str) {
    if let Some((library, reason)) = split_library_reason(message) {
        let _ = crate::baseobjspace::setattr_str(exception, "library", w_str_new(library));
        let _ = crate::baseobjspace::setattr_str(exception, "reason", w_str_new(reason));
    }
}

fn ssl_error(message: impl Into<String>) -> crate::PyError {
    let message = message.into();
    let mut err = crate::PyError::os_error(message.clone());
    if let Some(cls) = crate::builtins::lookup_exc_class("_ssl.SSLError")
        && let Ok(exc) =
            crate::builtins::exc_exception_new(&[cls, w_int_new(0), w_str_new(&message)])
    {
        set_library_reason(exc, &message);
        err.exc_object = exc;
    }
    err
}

/// CPython's `SSLError` is an `OSError` subclass, but its string form is the
/// TLS message rather than the generic two-item exception tuple.  Keep this
/// policy in the `_ssl` boundary instead of teaching the process-wide
/// exception formatter about an extension-module class.
fn ssl_error_str(args: &[PyObjectRef]) -> crate::PyResult {
    let w_args = unsafe { pyre_object::interp_exceptions::w_exception_get_args(args[0]) };
    let len = unsafe { pyre_object::w_tuple_len(w_args) };
    if len >= 2 {
        let message = unsafe {
            pyre_object::w_tuple_getitem(w_args, 1)
                .expect("SSLError argument tuple has a second element")
        };
        crate::builtins::builtin_str(&[message])
    } else {
        Ok(pyre_object::w_str_from_wtf8(unsafe {
            crate::display::base_exception_str_wtf8(args[0])?
        }))
    }
}

fn native_result<T>(result: Result<T, (i32, String)>) -> Result<T, crate::PyError> {
    result.map_err(|(errno, message)| {
        if errno != 0 {
            crate::PyError::os_error_with_errno(errno, message)
        } else {
            ssl_error(message)
        }
    })
}

fn tls_error(code: i32, message: String) -> crate::PyError {
    let verify_code = (code >= pyre_native::ssl::TLS_ERROR_CERT_VERIFY_BASE)
        .then_some(code - pyre_native::ssl::TLS_ERROR_CERT_VERIFY_BASE);
    let class_name = match code {
        pyre_native::ssl::TLS_ERROR_WANT_READ => "_ssl.SSLWantReadError",
        pyre_native::ssl::TLS_ERROR_WANT_WRITE => "_ssl.SSLWantWriteError",
        pyre_native::ssl::TLS_ERROR_ZERO_RETURN => "_ssl.SSLZeroReturnError",
        pyre_native::ssl::TLS_ERROR_EOF => "_ssl.SSLEOFError",
        _ if verify_code.is_some() => "_ssl.SSLCertVerificationError",
        _ => "_ssl.SSLError",
    };
    // `ssl_error` would build a complete `_ssl.SSLError` only for the
    // class-specific instance below to replace it, and WANT_READ/WANT_WRITE
    // travel this path on every non-blocking handshake step.
    let mut error = crate::PyError::os_error(message.clone());
    let public_errno = if verify_code.is_some() {
        pyre_native::ssl::TLS_ERROR_SSL
    } else {
        code
    };
    if let Some(class) = crate::builtins::lookup_exc_class(class_name)
        && let Ok(exception) = crate::builtins::exc_os_error_new(&[
            class,
            w_int_new(public_errno as i64),
            w_str_new(&message),
        ])
    {
        set_library_reason(exception, &message);
        if let Some(verify_code) = verify_code {
            let _ = crate::baseobjspace::setattr_str(
                exception,
                "verify_code",
                w_int_new(verify_code as i64),
            );
            let _ = crate::baseobjspace::setattr_str(
                exception,
                "verify_message",
                w_str_new(pyre_native::ssl::certificate_verify_message(verify_code)),
            );
            let _ = crate::baseobjspace::setattr_str(exception, "library", w_str_new("SSL"));
            let _ = crate::baseobjspace::setattr_str(
                exception,
                "reason",
                w_str_new("CERTIFICATE_VERIFY_FAILED"),
            );
        }
        error.exc_object = exception;
    }
    error
}

fn tls_result<T>(result: pyre_native::ssl::TlsResult<T>) -> Result<T, crate::PyError> {
    result.map_err(|(code, message)| tls_error(code, message))
}

fn path_string(obj: PyObjectRef) -> Result<String, crate::PyError> {
    if obj.is_null() || unsafe { is_none(obj) || is_bool(obj) } {
        return Err(crate::PyError::type_error(
            "path should be string, bytes, os.PathLike or integer, not NoneType",
        ));
    }
    Ok(crate::gateway::fspath_buf(obj)?
        .to_string_lossy()
        .into_owned())
}

fn password_bytes(obj: PyObjectRef) -> Result<Option<Vec<u8>>, crate::PyError> {
    if obj.is_null() || unsafe { is_none(obj) } {
        return Ok(None);
    }
    let callable = !unsafe { is_str(obj) }
        && !unsafe { bytesobject::is_bytes_like(obj) }
        && crate::baseobjspace::getattr_str(obj, "__call__").is_ok();
    let value = if callable {
        crate::call::call_function_impl_result(obj, &[])?
    } else {
        obj
    };
    let bytes = if unsafe { is_str(value) } {
        crate::baseobjspace::str_utf8_w(value)?.as_bytes().to_vec()
    } else if let Some(buffer) = crate::baseobjspace::simple_buffer_bytes(value)? {
        let bytes = buffer.as_bytes().to_vec();
        buffer.release();
        bytes
    } else {
        return Err(crate::PyError::type_error(if callable {
            "password callback must return a string"
        } else {
            "password should be a string or callable"
        }));
    };
    if bytes.len() > 1024 {
        return Err(crate::PyError::value_error(
            "password cannot be longer than 1024 bytes",
        ));
    }
    Ok(Some(bytes))
}

fn dict_from_pairs(items: &[(&str, PyObjectRef)]) -> PyObjectRef {
    let dict = w_dict_new();
    for (key, value) in items {
        unsafe { w_dict_setitem_str(dict, key, *value) };
    }
    dict
}

fn decoded_name(cert: *const pyre_native::ssl::DecodedCertificate, subject: bool) -> PyObjectRef {
    let rdn_count = unsafe { pyre_native::ssl::certificate_name_rdn_count(cert, subject) };
    let mut rdns = Vec::with_capacity(rdn_count);
    for rdn in 0..rdn_count {
        let attribute_count =
            unsafe { pyre_native::ssl::certificate_name_attribute_count(cert, subject, rdn) };
        let mut attributes = Vec::with_capacity(attribute_count);
        for attribute in 0..attribute_count {
            let key = unsafe {
                pyre_native::ssl::certificate_name_attribute_key(cert, subject, rdn, attribute)
            };
            let value = unsafe {
                pyre_native::ssl::certificate_name_attribute_value(cert, subject, rdn, attribute)
            };
            attributes.push(w_tuple_new(vec![w_str_new(&key), w_str_new(&value)]));
        }
        rdns.push(w_tuple_new(attributes));
    }
    w_tuple_new(rdns)
}

fn decoded_directory_name(
    cert: *const pyre_native::ssl::DecodedCertificate,
    san: usize,
) -> PyObjectRef {
    let rdn_count = unsafe { pyre_native::ssl::certificate_san_directory_rdn_count(cert, san) };
    let mut rdns = Vec::with_capacity(rdn_count);
    for rdn in 0..rdn_count {
        let count =
            unsafe { pyre_native::ssl::certificate_san_directory_attribute_count(cert, san, rdn) };
        let mut attributes = Vec::with_capacity(count);
        for attribute in 0..count {
            let key = unsafe {
                pyre_native::ssl::certificate_san_directory_attribute_key(cert, san, rdn, attribute)
            };
            let value = unsafe {
                pyre_native::ssl::certificate_san_directory_attribute_value(
                    cert, san, rdn, attribute,
                )
            };
            attributes.push(w_tuple_new(vec![w_str_new(&key), w_str_new(&value)]));
        }
        rdns.push(w_tuple_new(attributes));
    }
    w_tuple_new(rdns)
}

fn decoded_urls(
    cert: *const pyre_native::ssl::DecodedCertificate,
    kind: i32,
) -> Option<PyObjectRef> {
    let count = unsafe { pyre_native::ssl::certificate_url_count(cert, kind) };
    if count == 0 {
        return None;
    }
    Some(w_tuple_new(
        (0..count)
            .map(|index| {
                w_str_new(&unsafe { pyre_native::ssl::certificate_url(cert, kind, index) })
            })
            .collect(),
    ))
}

fn decoded_certificate_dict(cert: *mut pyre_native::ssl::DecodedCertificate) -> PyObjectRef {
    let dict = dict_from_pairs(&[
        ("issuer", decoded_name(cert, false)),
        (
            "notAfter",
            w_str_new(&unsafe { pyre_native::ssl::certificate_not_after(cert) }),
        ),
        (
            "notBefore",
            w_str_new(&unsafe { pyre_native::ssl::certificate_not_before(cert) }),
        ),
        (
            "serialNumber",
            w_str_new(&unsafe { pyre_native::ssl::certificate_serial_number(cert) }),
        ),
        ("subject", decoded_name(cert, true)),
        (
            "version",
            w_int_new(unsafe { pyre_native::ssl::certificate_version(cert) } as i64),
        ),
    ]);
    for (name, kind) in [("OCSP", 0), ("caIssuers", 1), ("crlDistributionPoints", 2)] {
        if let Some(urls) = decoded_urls(cert, kind) {
            unsafe { w_dict_setitem_str(dict, name, urls) };
        }
    }
    let san_count = unsafe { pyre_native::ssl::certificate_san_count(cert) };
    if san_count != 0 {
        let mut names = Vec::with_capacity(san_count);
        for index in 0..san_count {
            let kind = unsafe { pyre_native::ssl::certificate_san_kind(cert, index) };
            let value = if kind == "DirName" {
                decoded_directory_name(cert, index)
            } else {
                w_str_new(&unsafe { pyre_native::ssl::certificate_san_value(cert, index) })
            };
            names.push(w_tuple_new(vec![w_str_new(kind), value]));
        }
        unsafe { w_dict_setitem_str(dict, "subjectAltName", w_tuple_new(names)) };
    }
    unsafe { pyre_native::ssl::certificate_free(cert) };
    dict
}

fn parse_server_hostname(
    value: Option<PyObjectRef>,
) -> Result<(Option<String>, PyObjectRef), crate::PyError> {
    let Some(value) = value.filter(|value| !unsafe { is_none(*value) }) else {
        return Ok((None, w_none()));
    };
    if !unsafe { is_str(value) } {
        return Err(crate::PyError::type_error(
            "server_hostname must be a string",
        ));
    }
    let hostname = crate::baseobjspace::str_utf8_w(value)?.to_string();
    if hostname.as_bytes().contains(&0) {
        return Err(crate::PyError::type_error(
            "argument must be encoded string without null bytes",
        ));
    }
    if hostname.is_empty() || hostname.starts_with('.') {
        return Err(crate::PyError::value_error(
            "server_hostname cannot be an empty string or start with a leading dot",
        ));
    }
    if !hostname.is_ascii() {
        return Err(crate::PyError::value_error(
            "server_hostname must contain ASCII only",
        ));
    }
    Ok((Some(hostname.clone()), w_str_new(&hostname)))
}

fn import_module(name: &str) -> Result<PyObjectRef, crate::PyError> {
    crate::importing::dunder_import(
        name,
        w_none(),
        w_none(),
        w_none(),
        0,
        crate::call::getexecutioncontext(),
    )?;
    crate::importing::get_sys_module(name).ok_or_else(|| {
        crate::PyError::new(
            crate::PyErrorKind::ImportError,
            format!("No module named '{name}'"),
        )
    })
}

fn allocate_ssl_socket(
    backend: *mut pyre_native::ssl::TlsConnection,
    context: PyObjectRef,
    socket: PyObjectRef,
    socket_send: PyObjectRef,
    socket_recv: PyObjectRef,
    incoming: PyObjectRef,
    outgoing: PyObjectRef,
    owner: PyObjectRef,
    server_hostname: PyObjectRef,
    server_side: bool,
    requested_session: *mut pyre_native::ssl::NativeSession,
) -> PyObjectRef {
    // Offsets into the pinned-root array below.  The weakref boxing further
    // down rewrites two of these slots by position, so a reorder of the array
    // has to move these constants with it.
    const CONTEXT_SLOT: usize = 0;
    const SOCKET_SLOT: usize = 1;
    const SOCKET_SEND_SLOT: usize = 2;
    const SOCKET_RECV_SLOT: usize = 3;
    const INCOMING_SLOT: usize = 4;
    const OUTGOING_SLOT: usize = 5;
    const OWNER_SLOT: usize = 6;
    const SERVER_HOSTNAME_SLOT: usize = 7;
    const SLOT_COUNT: usize = 8;

    let _ = ssl_socket_methods::type_object();
    let _roots = pyre_object::gc_roots::push_roots();
    for value in [
        context,
        socket,
        socket_send,
        socket_recv,
        incoming,
        outgoing,
        owner,
        server_hostname,
    ] {
        pyre_object::gc_roots::pin_root(value);
    }
    let first = pyre_object::gc_roots::shadow_stack_len() - SLOT_COUNT;
    for slot in [SOCKET_SLOT, OWNER_SLOT] {
        let rooted = unsafe { pyre_object::gc_roots::shadow_stack_get(first + slot) };
        if !unsafe { is_none(rooted) } {
            let weak = pyre_object::weakref::w_gc_weakref_box_new_or_strong(rooted);
            pyre_object::gc_roots::shadow_stack_set(first + slot, weak);
        }
    }
    W_SSLSocket::allocate_stable(W_SSLSocket {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        backend,
        context: unsafe { pyre_object::gc_roots::shadow_stack_get(first + CONTEXT_SLOT) },
        socket: unsafe { pyre_object::gc_roots::shadow_stack_get(first + SOCKET_SLOT) },
        socket_send: unsafe { pyre_object::gc_roots::shadow_stack_get(first + SOCKET_SEND_SLOT) },
        socket_recv: unsafe { pyre_object::gc_roots::shadow_stack_get(first + SOCKET_RECV_SLOT) },
        incoming: unsafe { pyre_object::gc_roots::shadow_stack_get(first + INCOMING_SLOT) },
        outgoing: unsafe { pyre_object::gc_roots::shadow_stack_get(first + OUTGOING_SLOT) },
        owner: unsafe { pyre_object::gc_roots::shadow_stack_get(first + OWNER_SLOT) },
        server_hostname: unsafe {
            pyre_object::gc_roots::shadow_stack_get(first + SERVER_HOSTNAME_SLOT)
        },
        server_side,
        shutdown_started: false,
        requested_session,
    })
}

fn allocate_ssl_session(backend: *mut pyre_native::ssl::NativeSession) -> PyObjectRef {
    let _ = ssl_session_methods::type_object();
    W_SSLSession::allocate_stable(W_SSLSession {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        backend,
    })
}

fn clone_requested_session(
    value: Option<PyObjectRef>,
    context: *mut pyre_native::ssl::Context,
) -> Result<*mut pyre_native::ssl::NativeSession, crate::PyError> {
    let Some(value) = value.filter(|value| !unsafe { is_none(*value) }) else {
        return Ok(std::ptr::null_mut());
    };
    let session = W_SSLSession::from_obj(value)
        .ok_or_else(|| crate::PyError::type_error("Value is not a SSLSession."))?;
    if unsafe { pyre_native::ssl::session_context_identity(session.backend) }
        != unsafe { pyre_native::ssl::context_identity(context) }
    {
        return Err(crate::PyError::value_error(
            "Session refers to a different SSLContext.",
        ));
    }
    Ok(unsafe { pyre_native::ssl::session_clone(session.backend) })
}

mod context_methods {
    use super::*;

    fn pin_and_allocate_context(
        cls: PyObjectRef,
        backend: *mut pyre_native::ssl::Context,
    ) -> PyObjectRef {
        let _roots = pyre_object::gc_roots::push_roots();
        pyre_object::gc_roots::pin_root(cls);
        let cls_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let obj = W_SSLContext::allocate_stable(W_SSLContext {
            ob: PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            map: std::ptr::null(),
            storage: std::ptr::null_mut(),
            backend,
            sni_callback: w_none(),
            msg_callback: w_none(),
            keylog_filename: w_none(),
            host_flags: 0,
            post_handshake_auth: false,
            num_tickets: 2,
        });
        crate::typedef::tag_subclass_instance(obj, unsafe {
            pyre_object::gc_roots::shadow_stack_get(cls_slot)
        })
    }

    #[crate::pyre_methods(doc = "An SSLContext holds TLS configuration and state.")]
    impl W_SSLContext {
        #[staticmethod]
        fn __new__(cls: PyObjectRef, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
            if crate::builtins::has_real_kwargs(kwargs) {
                return Err(crate::PyError::type_error(
                    "_SSLContext.__new__() takes no keyword arguments",
                ));
            }
            let protocol_obj = positional.get(1).copied().ok_or_else(|| {
                crate::PyError::type_error(
                    "_SSLContext.__new__() missing required argument 'protocol'",
                )
            })?;
            if positional.len() != 2 {
                return Err(crate::PyError::type_error(format!(
                    "_SSLContext.__new__() takes 2 arguments ({} given)",
                    positional.len()
                )));
            }
            crate::typedef::check_user_subclass(type_object(), cls)?;
            let protocol = crate::baseobjspace::int_w(protocol_obj)?;
            let protocol = i32::try_from(protocol).map_err(|_| {
                crate::PyError::value_error("invalid or unsupported protocol version")
            })?;
            let backend = pyre_native::ssl::context_new(protocol)
                .map_err(|message| crate::PyError::value_error(message))?;
            if matches!(protocol, PROTOCOL_TLS | 3 | 4 | 5) {
                crate::warn::warn_deprecation(&format!(
                    "ssl.PROTOCOL_{} is deprecated",
                    match protocol {
                        PROTOCOL_TLS => "TLS",
                        3 => "TLSv1",
                        4 => "TLSv1_1",
                        5 => "TLSv1_2",
                        _ => unreachable!(),
                    }
                ))?;
            }
            Ok(pin_and_allocate_context(cls, backend))
        }

        #[getter]
        fn protocol(&self) -> i64 {
            unsafe { pyre_native::ssl::context_protocol(self.backend) as i64 }
        }

        #[getter]
        fn check_hostname(&self) -> bool {
            unsafe { pyre_native::ssl::context_check_hostname(self.backend) }
        }

        #[setter]
        fn set_check_hostname(&mut self, value: bool) {
            unsafe {
                pyre_native::ssl::context_set_check_hostname(self.backend, value);
                if value && pyre_native::ssl::context_verify_mode(self.backend) == CERT_NONE {
                    pyre_native::ssl::context_set_verify_mode(self.backend, CERT_REQUIRED);
                }
            }
        }

        #[getter]
        fn verify_mode(&self) -> i64 {
            unsafe { pyre_native::ssl::context_verify_mode(self.backend) as i64 }
        }

        #[setter]
        fn set_verify_mode(&mut self, value: PyObjectRef) -> Result<(), crate::PyError> {
            let mode = crate::baseobjspace::int_w(value)?;
            if !(CERT_NONE as i64..=CERT_REQUIRED as i64).contains(&mode) {
                return Err(crate::PyError::value_error("invalid verify mode"));
            }
            if mode == CERT_NONE as i64 && self.check_hostname() {
                return Err(crate::PyError::value_error(
                    "Cannot set verify_mode to CERT_NONE when check_hostname is enabled",
                ));
            }
            unsafe { pyre_native::ssl::context_set_verify_mode(self.backend, mode as i32) };
            Ok(())
        }

        #[getter]
        fn verify_flags(&self) -> i64 {
            unsafe { pyre_native::ssl::context_verify_flags(self.backend) as i64 }
        }

        #[setter]
        fn set_verify_flags(&mut self, value: PyObjectRef) -> Result<(), crate::PyError> {
            let value = crate::baseobjspace::int_w(value)?;
            let value = i32::try_from(value)
                .map_err(|_| crate::PyError::overflow_error("verify_flags out of range"))?;
            unsafe { pyre_native::ssl::context_set_verify_flags(self.backend, value) };
            Ok(())
        }

        #[getter]
        fn options(&self) -> i64 {
            unsafe { pyre_native::ssl::context_options(self.backend) as i64 }
        }

        #[setter]
        fn set_options(&mut self, value: PyObjectRef) -> Result<(), crate::PyError> {
            let value = crate::baseobjspace::uint_w(value)?;
            let old = unsafe { pyre_native::ssl::context_options(self.backend) };
            const DEPRECATED: u64 =
                0x0200_0000 | 0x0400_0000 | 0x1000_0000 | 0x0800_0000 | 0x2000_0000;
            if (!old & value) & DEPRECATED != 0 {
                crate::warn::warn_deprecation(
                    "ssl.OP_NO_SSL*/ssl.OP_NO_TLS* options are deprecated",
                )?;
            }
            unsafe { pyre_native::ssl::context_set_options(self.backend, value) };
            Ok(())
        }

        #[getter]
        fn minimum_version(&self) -> i64 {
            unsafe { pyre_native::ssl::context_minimum_version(self.backend) as i64 }
        }

        #[setter]
        fn set_minimum_version(&mut self, value: PyObjectRef) -> Result<(), crate::PyError> {
            let value = validate_tls_version(crate::baseobjspace::int_w(value)?)?;
            let value = if value == -1 { 0x304 } else { value };
            unsafe { pyre_native::ssl::context_set_minimum_version(self.backend, value) };
            Ok(())
        }

        #[getter]
        fn maximum_version(&self) -> i64 {
            unsafe { pyre_native::ssl::context_maximum_version(self.backend) as i64 }
        }

        #[setter]
        fn set_maximum_version(&mut self, value: PyObjectRef) -> Result<(), crate::PyError> {
            let value = validate_tls_version(crate::baseobjspace::int_w(value)?)?;
            let value = if value == -2 { 0x303 } else { value };
            unsafe { pyre_native::ssl::context_set_maximum_version(self.backend, value) };
            Ok(())
        }

        #[getter]
        fn _host_flags(&self) -> i64 {
            self.host_flags as i64
        }

        #[setter("_host_flags")]
        fn set_host_flags(&mut self, value: PyObjectRef) -> Result<(), crate::PyError> {
            let value = crate::baseobjspace::int_w(value)?;
            self.host_flags = i32::try_from(value)
                .map_err(|_| crate::PyError::overflow_error("_host_flags out of range"))?;
            Ok(())
        }

        #[getter]
        fn post_handshake_auth(&self) -> bool {
            self.post_handshake_auth
        }

        #[setter]
        fn set_post_handshake_auth(&mut self, value: bool) {
            self.post_handshake_auth = value;
        }

        #[getter]
        fn num_tickets(&self) -> i64 {
            self.num_tickets as i64
        }

        #[setter]
        fn set_num_tickets(&mut self, value: PyObjectRef) -> Result<(), crate::PyError> {
            let value = crate::baseobjspace::int_w(value)?;
            if value < 0 {
                return Err(crate::PyError::value_error(
                    "num_tickets must be a non-negative integer",
                ));
            }
            if self.protocol() != PROTOCOL_TLS_SERVER as i64 {
                return Err(crate::PyError::value_error(
                    "num_tickets can only be set on server-side contexts",
                ));
            }
            self.num_tickets = i32::try_from(value)
                .map_err(|_| crate::PyError::overflow_error("num_tickets out of range"))?;
            unsafe {
                pyre_native::ssl::context_set_num_tickets(self.backend, self.num_tickets as usize)
            };
            Ok(())
        }

        #[getter]
        fn sni_callback(&self) -> PyObjectRef {
            self.sni_callback
        }

        #[setter]
        fn set_sni_callback(&mut self, value: PyObjectRef) -> Result<(), crate::PyError> {
            if !unsafe { is_none(value) }
                && crate::baseobjspace::getattr_str(value, "__call__").is_err()
            {
                return Err(crate::PyError::type_error("not a callable object"));
            }
            pyre_object::gc_hook::try_gc_write_barrier(self as *mut W_SSLContext as *mut u8);
            self.sni_callback = value;
            Ok(())
        }

        #[getter]
        fn _msg_callback(&self) -> PyObjectRef {
            self.msg_callback
        }

        #[setter("_msg_callback")]
        fn set_msg_callback(&mut self, value: PyObjectRef) -> Result<(), crate::PyError> {
            if !unsafe { is_none(value) }
                && crate::baseobjspace::getattr_str(value, "__call__").is_err()
            {
                return Err(crate::PyError::type_error(format!(
                    "{} is not callable.",
                    crate::type_methods::arg_type_name(value)
                )));
            }
            pyre_object::gc_hook::try_gc_write_barrier(self as *mut W_SSLContext as *mut u8);
            self.msg_callback = value;
            Ok(())
        }

        fn load_cert_chain(&mut self, args: &[PyObjectRef]) -> Result<(), crate::PyError> {
            const KEYWORDS: &[&str] = &["certfile", "keyfile", "password"];
            let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
            let user = positional.get(1..).unwrap_or(&[]);
            if user.len() > 3 {
                return Err(crate::PyError::type_error(
                    "load_cert_chain() takes at most 3 arguments",
                ));
            }
            let cert =
                crate::builtins::bind_pos_or_kw(user, kwargs, 0, "certfile", "load_cert_chain", 1)?
                    .ok_or_else(|| {
                        crate::PyError::type_error(
                            "load_cert_chain() missing required argument 'certfile'",
                        )
                    })?;
            let key =
                crate::builtins::bind_pos_or_kw(user, kwargs, 1, "keyfile", "load_cert_chain", 2)?;
            let password =
                crate::builtins::bind_pos_or_kw(user, kwargs, 2, "password", "load_cert_chain", 3)?;
            crate::builtins::kwarg_reject_unknown(kwargs, KEYWORDS, "load_cert_chain")?;
            let cert_path = path_string(cert)?;
            let key_path = match key {
                Some(value) if !unsafe { is_none(value) } => path_string(value)?,
                _ => cert_path.clone(),
            };
            // OpenSSL asks its callback only after parsing discovers an
            // encrypted key. Preserve that observable ordering: an irrelevant
            // callback must not run for an unencrypted key file.  The backend
            // reports which case it hit, so the container decides this rather
            // than a scan of the file for header text.
            let loaded = native_result(unsafe {
                pyre_native::ssl::context_load_cert_chain(self.backend, &cert_path, &key_path, None)
            })?;
            if loaded {
                return Ok(());
            }
            let Some(password) = password_bytes(password.unwrap_or_else(w_none))? else {
                return Err(ssl_error("[SSL] PEM routines: bad password read"));
            };
            native_result(unsafe {
                pyre_native::ssl::context_load_cert_chain(
                    self.backend,
                    &cert_path,
                    &key_path,
                    Some(&password),
                )
            })
            .map(|_| ())
        }

        fn load_verify_locations(&mut self, args: &[PyObjectRef]) -> Result<(), crate::PyError> {
            const KEYWORDS: &[&str] = &["cafile", "capath", "cadata"];
            let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
            let user = positional.get(1..).unwrap_or(&[]);
            if user.len() > 3 {
                return Err(crate::PyError::type_error(
                    "load_verify_locations() takes at most 3 arguments",
                ));
            }
            let cafile = crate::builtins::bind_pos_or_kw(
                user,
                kwargs,
                0,
                "cafile",
                "load_verify_locations",
                1,
            )?;
            let capath = crate::builtins::bind_pos_or_kw(
                user,
                kwargs,
                1,
                "capath",
                "load_verify_locations",
                2,
            )?;
            let cadata = crate::builtins::bind_pos_or_kw(
                user,
                kwargs,
                2,
                "cadata",
                "load_verify_locations",
                3,
            )?;
            crate::builtins::kwarg_reject_unknown(kwargs, KEYWORDS, "load_verify_locations")?;
            let cafile = cafile.filter(|value| !unsafe { is_none(*value) });
            let capath = capath.filter(|value| !unsafe { is_none(*value) });
            let cadata = cadata.filter(|value| !unsafe { is_none(*value) });
            if cafile.is_none() && capath.is_none() && cadata.is_none() {
                return Err(crate::PyError::type_error(
                    "cafile, capath and cadata cannot be all omitted",
                ));
            }
            if let Some(cafile) = cafile {
                let path = path_string(cafile)?;
                native_result(unsafe {
                    pyre_native::ssl::context_load_verify_file(self.backend, &path)
                })?;
            }
            if let Some(capath) = capath {
                let path = path_string(capath)?;
                if !std::path::Path::new(&path).is_dir() {
                    return Err(crate::PyError::os_error_with_errno(
                        libc::ENOENT,
                        "CA directory does not exist",
                    ));
                }
                unsafe { pyre_native::ssl::context_add_verify_dir(self.backend, &path) };
            }
            if let Some(cadata) = cadata {
                if unsafe { is_str(cadata) } {
                    let data = crate::baseobjspace::str_utf8_w(cadata)?;
                    unsafe {
                        pyre_native::ssl::context_load_verify_data(
                            self.backend,
                            data.as_bytes(),
                            true,
                        )
                    }
                    .map_err(|_| {
                        ssl_error("no start line: cadata does not contain a certificate")
                    })?;
                } else {
                    let buffer =
                        crate::baseobjspace::simple_buffer_bytes(cadata)?.ok_or_else(|| {
                            crate::PyError::type_error(
                                "cadata should be an ASCII string or a bytes-like object",
                            )
                        })?;
                    let result = unsafe {
                        pyre_native::ssl::context_load_verify_data(
                            self.backend,
                            buffer.as_bytes(),
                            false,
                        )
                    }
                    .map_err(|_| {
                        ssl_error("not enough data: cadata does not contain a certificate")
                    });
                    buffer.release();
                    result?;
                }
            }
            Ok(())
        }

        /// `SSL_CTX_set_default_verify_paths` loads two independent sources: a
        /// bundle file and a hashed directory.  `SSL_CERT_FILE` overrides only
        /// the first and `SSL_CERT_DIR` only the second, so neither variable
        /// may suppress the other's source.
        fn set_default_verify_paths(&mut self) -> Result<(), crate::PyError> {
            let env_path = |name: &[u8]| {
                crate::host_seam::getenv(name)
                    .ok()
                    .flatten()
                    .map(|value| String::from_utf8_lossy(&value).into_owned())
            };
            let cert_file =
                env_path(b"SSL_CERT_FILE").filter(|path| std::path::Path::new(path).is_file());
            match cert_file {
                Some(path) => native_result(unsafe {
                    pyre_native::ssl::context_load_verify_file(self.backend, &path)
                })
                .map(|_| ())?,
                None => native_result(unsafe {
                    pyre_native::ssl::context_load_native_roots(self.backend)
                })
                .map(|_| ())?,
            }
            let (_, default_dir) = pyre_native::ssl::default_verify_paths();
            let cert_dir = env_path(b"SSL_CERT_DIR").unwrap_or(default_dir);
            if std::path::Path::new(&cert_dir).is_dir() {
                // OpenSSL defers hashed directory loading until chain lookup,
                // so it does not contribute to cert_store_stats here.
                unsafe { pyre_native::ssl::context_add_verify_dir(self.backend, &cert_dir) };
            }
            Ok(())
        }

        fn cert_store_stats(&self) -> PyObjectRef {
            let (x509, ca) = unsafe { pyre_native::ssl::context_cert_store_stats(self.backend) };
            dict_from_pairs(&[
                ("x509", w_int_new(x509 as i64)),
                ("crl", w_int_new(0)),
                ("x509_ca", w_int_new(ca as i64)),
            ])
        }

        fn get_ca_certs(
            &self,
            #[default(false)] binary_form: bool,
        ) -> Result<PyObjectRef, crate::PyError> {
            let certs = unsafe { pyre_native::ssl::context_ca_certs(self.backend) };
            let mut result = Vec::with_capacity(certs.len());
            for cert in certs {
                result.push(if binary_form {
                    w_bytes_from_bytes(&cert)
                } else {
                    let decoded = native_result(pyre_native::ssl::certificate_decode_der(&cert))?;
                    decoded_certificate_dict(decoded)
                });
            }
            Ok(w_list_new(result))
        }

        fn set_ciphers(&mut self, cipherlist: PyObjectRef) -> Result<(), crate::PyError> {
            if !unsafe { is_str(cipherlist) } {
                return Err(crate::PyError::type_error("cipherlist must be a string"));
            }
            let cipherlist = crate::baseobjspace::str_utf8_w(cipherlist)?;
            unsafe { pyre_native::ssl::context_set_cipher_list(self.backend, cipherlist.as_ref()) }
                .map_err(ssl_error)
        }

        /// The suites this context can actually negotiate, so a `set_ciphers()`
        /// filter is visible to configuration auditing rather than only to
        /// connection creation.
        fn get_ciphers(&self) -> PyObjectRef {
            let mut ciphers = Vec::with_capacity(pyre_native::ssl::cipher_count());
            for index in 0..pyre_native::ssl::cipher_count() {
                if !unsafe { pyre_native::ssl::context_cipher_enabled(self.backend, index) } {
                    continue;
                }
                let name = pyre_native::ssl::cipher_name(index);
                let bits = pyre_native::ssl::cipher_bits(index);
                ciphers.push(dict_from_pairs(&[
                    ("id", w_int_new(index as i64)),
                    ("name", w_str_new(name)),
                    (
                        "protocol",
                        w_str_new(pyre_native::ssl::cipher_protocol(index)),
                    ),
                    ("description", w_str_new(name)),
                    ("strength_bits", w_int_new(bits as i64)),
                    ("alg_bits", w_int_new(bits as i64)),
                    ("aead", w_bool_from(pyre_native::ssl::cipher_aead(index))),
                    (
                        "symmetric",
                        w_str_new(pyre_native::ssl::cipher_symmetric(index)),
                    ),
                    ("digest", w_str_new(pyre_native::ssl::cipher_digest(index))),
                    ("kea", w_str_new(pyre_native::ssl::cipher_kea(index))),
                    ("auth", w_str_new(pyre_native::ssl::cipher_auth(index))),
                ]));
            }
            w_list_new(ciphers)
        }

        fn session_stats(&self) -> PyObjectRef {
            let (accept, hits) = unsafe { pyre_native::ssl::context_session_stats(self.backend) };
            dict_from_pairs(&[
                ("number", w_int_new(0)),
                ("connect", w_int_new(0)),
                ("connect_good", w_int_new(0)),
                ("connect_renegotiate", w_int_new(0)),
                ("accept", w_int_new(accept as i64)),
                ("accept_good", w_int_new(accept as i64)),
                ("accept_renegotiate", w_int_new(0)),
                ("hits", w_int_new(hits as i64)),
                ("misses", w_int_new(0)),
                ("timeouts", w_int_new(0)),
                ("cache_full", w_int_new(0)),
            ])
        }

        /// Validate the file the way `SSL_CTX_set_tmp_dh` would, then discard
        /// the group.  rustls negotiates ECDHE only, so no finite-field DH
        /// parameter can ever take effect; a server that set one still gets a
        /// forward-secret key exchange, and `supports_kx_alias` reports no DHE
        /// so callers can detect the gap.  Raising here instead would reject
        /// `load_dh_params` outright for every caller.
        fn load_dh_params(&self, path: PyObjectRef) -> Result<(), crate::PyError> {
            if path.is_null() || unsafe { is_none(path) } {
                return Err(crate::PyError::type_error(
                    "load_dh_params() missing required path argument",
                ));
            }
            let path = path_string(path)?;
            let data = std::fs::read(&path).map_err(|error| {
                crate::PyError::os_error_with_errno(
                    error.raw_os_error().unwrap_or(libc::EIO),
                    error.to_string(),
                )
            })?;
            if !data
                .windows(b"BEGIN DH PARAMETERS".len())
                .any(|window| window == b"BEGIN DH PARAMETERS")
            {
                return Err(ssl_error("[PEM: NO_START_LINE] no start line"));
            }
            Ok(())
        }

        fn set_ecdh_curve(&self, curve: PyObjectRef) -> Result<(), crate::PyError> {
            let curve = if unsafe { is_str(curve) } {
                crate::baseobjspace::str_utf8_w(curve)?.to_string()
            } else if unsafe { bytesobject::is_bytes_like(curve) } {
                String::from_utf8(unsafe { bytesobject::bytes_like_data(curve) }.to_vec())
                    .map_err(|_| crate::PyError::value_error("invalid curve name"))?
            } else {
                return Err(crate::PyError::type_error("curve_name must be a string"));
            };
            unsafe { pyre_native::ssl::context_set_ecdh_curve(self.backend, &curve) }
                .map_err(crate::PyError::value_error)
        }

        #[getter]
        fn security_level(&self) -> i64 {
            2
        }

        fn _wrap_bio(&self, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            const KEYWORDS: &[&str] = &[
                "incoming",
                "outgoing",
                "server_side",
                "server_hostname",
                "owner",
                "session",
            ];
            let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
            let user = positional.get(1..).unwrap_or(&[]);
            if user.len() > KEYWORDS.len() {
                return Err(crate::PyError::type_error(
                    "_wrap_bio() takes at most 6 arguments",
                ));
            }
            let incoming =
                crate::builtins::bind_pos_or_kw(user, kwargs, 0, "incoming", "_wrap_bio", 1)?
                    .ok_or_else(|| {
                        crate::PyError::type_error(
                            "_wrap_bio() missing required argument 'incoming'",
                        )
                    })?;
            let outgoing =
                crate::builtins::bind_pos_or_kw(user, kwargs, 1, "outgoing", "_wrap_bio", 2)?
                    .ok_or_else(|| {
                        crate::PyError::type_error(
                            "_wrap_bio() missing required argument 'outgoing'",
                        )
                    })?;
            let server_side =
                crate::builtins::bind_pos_or_kw(user, kwargs, 2, "server_side", "_wrap_bio", 3)?
                    .map(crate::baseobjspace::is_true)
                    .transpose()?
                    .unwrap_or(false);
            let hostname = crate::builtins::bind_pos_or_kw(
                user,
                kwargs,
                3,
                "server_hostname",
                "_wrap_bio",
                4,
            )?;
            let owner = crate::builtins::bind_pos_or_kw(user, kwargs, 4, "owner", "_wrap_bio", 5)?
                .unwrap_or_else(w_none);
            let session =
                crate::builtins::bind_pos_or_kw(user, kwargs, 5, "session", "_wrap_bio", 6)?;
            crate::builtins::kwarg_reject_unknown(kwargs, KEYWORDS, "_wrap_bio")?;
            if W_MemoryBIO::from_obj(incoming).is_none()
                || W_MemoryBIO::from_obj(outgoing).is_none()
            {
                return Err(crate::PyError::type_error(
                    "incoming and outgoing must be MemoryBIO objects",
                ));
            }
            if let Some(session) = session.filter(|value| !unsafe { is_none(*value) })
                && W_SSLSession::from_obj(session).is_none()
            {
                return Err(crate::PyError::type_error("Value is not a SSLSession."));
            }
            let (hostname, hostname_obj) = parse_server_hostname(hostname)?;
            if !server_side
                && unsafe { pyre_native::ssl::context_check_hostname(self.backend) }
                && hostname.is_none()
            {
                return Err(crate::PyError::value_error(
                    "check_hostname requires server_hostname",
                ));
            }
            let protocol = unsafe { pyre_native::ssl::context_protocol(self.backend) };
            if server_side && protocol == PROTOCOL_TLS_CLIENT {
                return Err(ssl_error(
                    "Cannot create a server socket with a PROTOCOL_TLS_CLIENT context",
                ));
            }
            if !server_side && protocol == PROTOCOL_TLS_SERVER {
                return Err(ssl_error(
                    "Cannot create a client socket with a PROTOCOL_TLS_SERVER context",
                ));
            }
            let requested_session = clone_requested_session(session, self.backend)?;
            Ok(allocate_ssl_socket(
                std::ptr::null_mut(),
                self as *const W_SSLContext as PyObjectRef,
                w_none(),
                w_none(),
                w_none(),
                incoming,
                outgoing,
                owner,
                hostname_obj,
                server_side,
                requested_session,
            ))
        }

        fn _wrap_socket(&self, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            const KEYWORDS: &[&str] =
                &["sock", "server_side", "server_hostname", "owner", "session"];
            let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
            let user = positional.get(1..).unwrap_or(&[]);
            if user.len() > KEYWORDS.len() {
                return Err(crate::PyError::type_error(
                    "_wrap_socket() takes at most 5 arguments",
                ));
            }
            let socket =
                crate::builtins::bind_pos_or_kw(user, kwargs, 0, "sock", "_wrap_socket", 1)?
                    .ok_or_else(|| {
                        crate::PyError::type_error(
                            "_wrap_socket() missing required argument 'sock'",
                        )
                    })?;
            let server_side =
                crate::builtins::bind_pos_or_kw(user, kwargs, 1, "server_side", "_wrap_socket", 2)?
                    .map(crate::baseobjspace::is_true)
                    .transpose()?
                    .unwrap_or(false);
            let hostname = crate::builtins::bind_pos_or_kw(
                user,
                kwargs,
                2,
                "server_hostname",
                "_wrap_socket",
                3,
            )?;
            let owner =
                crate::builtins::bind_pos_or_kw(user, kwargs, 3, "owner", "_wrap_socket", 4)?
                    .unwrap_or_else(w_none);
            let session =
                crate::builtins::bind_pos_or_kw(user, kwargs, 4, "session", "_wrap_socket", 5)?;
            crate::builtins::kwarg_reject_unknown(kwargs, KEYWORDS, "_wrap_socket")?;
            if let Some(session) = session.filter(|value| !unsafe { is_none(*value) })
                && W_SSLSession::from_obj(session).is_none()
            {
                return Err(crate::PyError::type_error("Value is not a SSLSession."));
            }
            let (hostname, hostname_obj) = parse_server_hostname(hostname)?;
            if !server_side
                && unsafe { pyre_native::ssl::context_check_hostname(self.backend) }
                && hostname.is_none()
            {
                return Err(crate::PyError::value_error(
                    "check_hostname requires server_hostname",
                ));
            }
            let protocol = unsafe { pyre_native::ssl::context_protocol(self.backend) };
            if server_side && protocol == PROTOCOL_TLS_CLIENT {
                return Err(ssl_error(
                    "Cannot create a server socket with a PROTOCOL_TLS_CLIENT context",
                ));
            }
            if !server_side && protocol == PROTOCOL_TLS_SERVER {
                return Err(ssl_error(
                    "Cannot create a client socket with a PROTOCOL_TLS_SERVER context",
                ));
            }
            let requested_session = clone_requested_session(session, self.backend)?;
            let socket_module = import_module("socket")?;
            let socket_class = crate::baseobjspace::getattr_str(socket_module, "socket")?;
            let socket_send = crate::baseobjspace::getattr_str(socket_class, "send")?;
            let socket_recv = crate::baseobjspace::getattr_str(socket_class, "recv")?;
            Ok(allocate_ssl_socket(
                std::ptr::null_mut(),
                self as *const W_SSLContext as PyObjectRef,
                socket,
                socket_send,
                socket_recv,
                w_none(),
                w_none(),
                owner,
                hostname_obj,
                server_side,
                requested_session,
            ))
        }

        fn _set_alpn_protocols(&mut self, data: PyObjectRef) -> Result<(), crate::PyError> {
            let buffer = crate::baseobjspace::simple_buffer_bytes(data)?
                .ok_or_else(|| crate::PyError::type_error("a bytes-like object is required"))?;
            // `PyBuffer_Release` before propagating: a parse failure must not
            // leave a `bytearray` argument permanently exported.
            let parsed = parse_length_prefixed_protocols(buffer.as_bytes());
            buffer.release();
            let protocols = parsed?;
            unsafe { pyre_native::ssl::context_set_alpn(self.backend, protocols) };
            Ok(())
        }

        fn _set_npn_protocols(&mut self, _data: PyObjectRef) -> Result<(), crate::PyError> {
            Err(ssl_error("NPN is not supported by rustls"))
        }
    }

    fn validate_tls_version(value: i64) -> Result<i32, crate::PyError> {
        let value = i32::try_from(value).map_err(|_| {
            crate::PyError::value_error(format!("invalid protocol version: {value}"))
        })?;
        if value != -2 && value != -1 && !(0x300..=0x304).contains(&value) {
            return Err(crate::PyError::value_error(format!(
                "invalid protocol version: {value}"
            )));
        }
        Ok(value)
    }

    fn parse_length_prefixed_protocols(data: &[u8]) -> Result<Vec<Vec<u8>>, crate::PyError> {
        let mut protocols = Vec::new();
        let mut offset = 0usize;
        while offset < data.len() {
            let len = data[offset] as usize;
            offset += 1;
            if len == 0 || offset + len > data.len() {
                return Err(ssl_error("invalid ALPN protocol list"));
            }
            protocols.push(data[offset..offset + len].to_vec());
            offset += len;
        }
        Ok(protocols)
    }
} // context_methods

mod memory_bio_methods {
    use super::*;

    #[crate::pyre_methods(doc = "Memory BIO for TLS protocol data.")]
    impl W_MemoryBIO {
        #[staticmethod]
        fn __new__(cls: PyObjectRef, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
            if positional.len() != 1 || crate::builtins::has_real_kwargs(kwargs) {
                return Err(crate::PyError::type_error("MemoryBIO() takes no arguments"));
            }
            crate::typedef::check_user_subclass(type_object(), cls)?;
            let _roots = pyre_object::gc_roots::push_roots();
            pyre_object::gc_roots::pin_root(cls);
            let cls_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            let obj = W_MemoryBIO::allocate_stable(W_MemoryBIO {
                ob: PyObject {
                    ob_type: std::ptr::null(),
                    w_class: std::ptr::null_mut(),
                },
                map: std::ptr::null(),
                storage: std::ptr::null_mut(),
                backend: pyre_native::ssl::memory_bio_new(),
            });
            Ok(crate::typedef::tag_subclass_instance(obj, unsafe {
                pyre_object::gc_roots::shadow_stack_get(cls_slot)
            }))
        }

        fn read(&mut self, #[default(-1)] size: i64) -> Result<PyObjectRef, crate::PyError> {
            let size = if size < 0 {
                unsafe { pyre_native::ssl::memory_bio_pending(self.backend) }
            } else {
                usize::try_from(size)
                    .map_err(|_| crate::PyError::overflow_error("read size out of range"))?
            };
            let bytes = unsafe { pyre_native::ssl::memory_bio_read(self.backend, size) };
            Ok(w_bytes_from_bytes(&bytes))
        }

        fn write(&mut self, data: PyObjectRef) -> Result<usize, crate::PyError> {
            let buffer = crate::baseobjspace::simple_buffer_bytes(data)?
                .ok_or_else(|| crate::PyError::type_error("a bytes-like object is required"))?;
            let result =
                unsafe { pyre_native::ssl::memory_bio_write(self.backend, buffer.as_bytes()) }
                    .map_err(ssl_error);
            buffer.release();
            result
        }

        fn write_eof(&mut self) {
            unsafe { pyre_native::ssl::memory_bio_write_eof(self.backend) };
        }

        #[getter]
        fn pending(&self) -> usize {
            unsafe { pyre_native::ssl::memory_bio_pending(self.backend) }
        }

        #[getter]
        fn eof(&self) -> bool {
            unsafe { pyre_native::ssl::memory_bio_eof(self.backend) }
        }
    }
} // memory_bio_methods

mod ssl_socket_methods {
    use super::*;

    fn call_transport(
        callable: PyObjectRef,
        socket: PyObjectRef,
        args: &[PyObjectRef],
    ) -> Result<PyObjectRef, crate::PyError> {
        let mut call_args = Vec::with_capacity(args.len() + 1);
        call_args.push(socket);
        call_args.extend_from_slice(args);
        crate::call::call_function_impl_result(callable, &call_args)
    }

    fn is_blocking_error(error: &crate::PyError) -> bool {
        let Some(class) = crate::builtins::lookup_exc_class("BlockingIOError") else {
            return false;
        };
        !error.exc_object.is_null()
            && unsafe { crate::baseobjspace::isinstance_w(error.exc_object, class) }
    }

    fn ensure_connection(socket: &mut W_SSLSocket) -> Result<(), crate::PyError> {
        if !socket.backend.is_null() {
            return Ok(());
        }
        let context =
            W_SSLContext::from_obj(socket.context).expect("SSL socket owns a live SSLContext");
        let hostname = if unsafe { is_none(socket.server_hostname) } {
            None
        } else {
            Some(crate::baseobjspace::str_utf8_w(socket.server_hostname)?.to_string())
        };
        socket.backend = native_result(unsafe {
            pyre_native::ssl::connection_new(
                context.backend,
                socket.server_side,
                hostname.as_deref(),
                socket.requested_session,
            )
        })?;
        Ok(())
    }

    fn transport_socket(socket: &W_SSLSocket) -> Result<PyObjectRef, crate::PyError> {
        let value =
            unsafe { pyre_object::weakref::w_gc_weakref_box_or_strong_deref(socket.socket) };
        if value.is_null() {
            Err(ssl_error("underlying socket has been collected"))
        } else {
            Ok(value)
        }
    }

    fn receive_tls(socket: &mut W_SSLSocket, data: &[u8]) -> Result<usize, crate::PyError> {
        let read = match unsafe { pyre_native::ssl::connection_receive_tls(socket.backend, data) } {
            Ok(read) => read,
            Err(error) => {
                // rustls may have queued the fatal alert describing this
                // protocol error. Send it before unwinding the Python call so
                // the peer cannot remain blocked in its handshake.
                let _ = flush_transport(socket);
                return tls_result(Err(error));
            }
        };
        run_message_callbacks(socket)?;
        if let Some(root) =
            unsafe { pyre_native::ssl::connection_take_verified_root(socket.backend) }
        {
            let context =
                W_SSLContext::from_obj(socket.context).expect("SSL socket owns its context");
            native_result(unsafe {
                pyre_native::ssl::context_add_verified_root(context.backend, root)
            })?;
        }
        Ok(read)
    }

    fn run_message_callbacks(socket: &mut W_SSLSocket) -> Result<(), crate::PyError> {
        let events = unsafe { pyre_native::ssl::connection_take_message_events(socket.backend) };
        if events.is_empty() {
            return Ok(());
        }
        let context = W_SSLContext::from_obj(socket.context).expect("SSL socket owns its context");
        if unsafe { is_none(context.msg_callback) } {
            return Ok(());
        }
        let owner = unsafe { pyre_object::weakref::w_gc_weakref_box_or_strong_deref(socket.owner) };
        if owner.is_null() {
            return Ok(());
        }
        for event in events {
            crate::call::call_function_impl_result(
                context.msg_callback,
                &[
                    owner,
                    w_str_new(if event.write { "write" } else { "read" }),
                    w_int_new(event.version as i64),
                    w_int_new(event.content_type as i64),
                    w_int_new(event.message_type as i64),
                    w_bytes_from_bytes(&event.data),
                ],
            )?;
        }
        Ok(())
    }

    fn flush_transport(socket: &mut W_SSLSocket) -> Result<(), crate::PyError> {
        ensure_connection(socket)?;
        if !unsafe { is_none(socket.outgoing) } {
            let data =
                tls_result(unsafe { pyre_native::ssl::connection_take_tls(socket.backend) })?;
            run_message_callbacks(socket)?;
            if !data.is_empty() {
                let outgoing = W_MemoryBIO::from_obj(socket.outgoing)
                    .expect("_wrap_bio validated its outgoing MemoryBIO");
                unsafe { pyre_native::ssl::memory_bio_write(outgoing.backend, &data) }
                    .map_err(ssl_error)?;
            }
            return Ok(());
        }

        loop {
            let data =
                tls_result(unsafe { pyre_native::ssl::connection_peek_tls(socket.backend) })?;
            run_message_callbacks(socket)?;
            if data.is_empty() {
                return Ok(());
            }
            let sent = match call_transport(
                socket.socket_send,
                transport_socket(socket)?,
                &[w_bytes_from_bytes(&data)],
            ) {
                Ok(value) => crate::baseobjspace::int_w(value)?,
                Err(error) if is_blocking_error(&error) => {
                    return Err(tls_error(
                        pyre_native::ssl::TLS_ERROR_WANT_WRITE,
                        "The operation did not complete (write)".to_string(),
                    ));
                }
                Err(error) => return Err(error),
            };
            if sent <= 0 {
                return Err(tls_error(
                    pyre_native::ssl::TLS_ERROR_WANT_WRITE,
                    "The operation did not complete (write)".to_string(),
                ));
            }
            unsafe { pyre_native::ssl::connection_consume_tls(socket.backend, sent as usize) };
        }
    }

    fn receive_transport(socket: &mut W_SSLSocket) -> Result<(), crate::PyError> {
        ensure_connection(socket)?;
        if !unsafe { is_none(socket.incoming) } {
            let incoming = W_MemoryBIO::from_obj(socket.incoming)
                .expect("_wrap_bio validated its incoming MemoryBIO");
            let pending = unsafe { pyre_native::ssl::memory_bio_pending(incoming.backend) };
            if pending == 0 {
                if unsafe { pyre_native::ssl::memory_bio_eof(incoming.backend) } {
                    let _ = receive_tls(socket, &[]);
                    return Err(tls_error(
                        pyre_native::ssl::TLS_ERROR_EOF,
                        "EOF occurred in violation of protocol".to_string(),
                    ));
                }
                return Err(tls_error(
                    pyre_native::ssl::TLS_ERROR_WANT_READ,
                    "The operation did not complete (read)".to_string(),
                ));
            }
            let data = unsafe { pyre_native::ssl::memory_bio_read(incoming.backend, pending) };
            receive_tls(socket, &data)?;
            return Ok(());
        }

        let value = match call_transport(
            socket.socket_recv,
            transport_socket(socket)?,
            &[w_int_new(32 * 1024)],
        ) {
            Ok(value) => value,
            Err(error) if is_blocking_error(&error) => {
                return Err(tls_error(
                    pyre_native::ssl::TLS_ERROR_WANT_READ,
                    "The operation did not complete (read)".to_string(),
                ));
            }
            Err(error) => return Err(error),
        };
        let buffer = crate::baseobjspace::simple_buffer_bytes(value)?.ok_or_else(|| {
            crate::PyError::type_error("socket recv() returned a non-bytes value")
        })?;
        if buffer.as_bytes().is_empty() {
            buffer.release();
            return Err(tls_error(
                pyre_native::ssl::TLS_ERROR_EOF,
                "EOF occurred in violation of protocol".to_string(),
            ));
        }
        let result = receive_tls(socket, buffer.as_bytes());
        buffer.release();
        result.map(|_| ())
    }

    fn reject_sni(socket: &mut W_SSLSocket, alert: u8, reason: &str) -> Result<(), crate::PyError> {
        tls_result(unsafe { pyre_native::ssl::connection_reject_server(socket.backend, alert) })?;
        // The peer must receive the fatal alert before the server-side
        // operation reports its callback failure.
        flush_transport(socket)?;
        Err(tls_error(
            pyre_native::ssl::TLS_ERROR_SSL,
            format!("[SSL: {reason}] SNI callback failed"),
        ))
    }

    /// Finish a server configuration after rustls has parsed ClientHello.
    ///
    /// This must operate through a raw pointer: the Python SNI callback may
    /// re-enter `_SSLSocket.context` and mutate this exact object. Keeping a
    /// Rust `&mut W_SSLSocket` live across that call would violate its unique
    /// alias and lets optimized builds reuse the pre-callback context. PyPy's
    /// `servername_callback` likewise reloads the SSL object's context after
    /// the app-level callback returns.
    #[inline(never)]
    unsafe fn configure_accepted_server(socket: *mut W_SSLSocket) -> Result<(), crate::PyError> {
        let backend = unsafe { (*socket).backend };
        if !unsafe { (*socket).server_side }
            || !unsafe { pyre_native::ssl::connection_waiting_for_server_config(backend) }
        {
            return Ok(());
        }

        let initial_context = unsafe { (*socket).context };
        let context = W_SSLContext::from_obj(initial_context)
            .expect("SSL socket owns a live initial context");
        let callback = context.sni_callback;
        if !unsafe { is_none(callback) } {
            let owner =
                unsafe { pyre_object::weakref::w_gc_weakref_box_or_strong_deref((*socket).owner) };
            if owner.is_null() {
                return Err(tls_error(
                    pyre_native::ssl::TLS_ERROR_SSL,
                    "[SSL: PARSE_TLSEXT] SNI callback owner is no longer available".to_string(),
                ));
            }
            let server_name = unsafe {
                pyre_native::ssl::connection_server_name(backend)
                    .map(|name| w_str_new(&name))
                    .unwrap_or_else(w_none)
            };
            let result = match crate::call::call_function_impl_result(
                callback,
                &[owner, server_name, initial_context],
            ) {
                Ok(result) => result,
                Err(mut error) => {
                    error.write_unraisable(
                        w_none(),
                        rustpython_wtf8::Wtf8::new("in SNI callback"),
                        callback,
                    );
                    return reject_sni(unsafe { &mut *socket }, 40, "CALLBACK_FAILED");
                }
            };
            if !unsafe { is_none(result) } {
                let alert = match crate::baseobjspace::int_w(result) {
                    Ok(alert) if (0..=u8::MAX as i64).contains(&alert) => alert as u8,
                    _ => {
                        let mut error = crate::PyError::type_error(
                            "SNI callback must return None or a TLS alert integer",
                        );
                        error.write_unraisable(
                            w_none(),
                            rustpython_wtf8::Wtf8::new("in SNI callback"),
                            callback,
                        );
                        return reject_sni(unsafe { &mut *socket }, 80, "CALLBACK_FAILED");
                    }
                };
                return reject_sni(unsafe { &mut *socket }, alert, "CALLBACK_FAILED");
            }
        }

        // The callback is allowed to assign ssl_sock.context. Reload through
        // the raw object pointer after re-entry; `initial_context` is only the
        // third callback argument, never the selected configuration anchor.
        let selected = W_SSLContext::from_obj(unsafe { (*socket).context })
            .expect("SNI callback preserves an SSLContext on the socket");
        match unsafe { pyre_native::ssl::connection_accept_server(backend, selected.backend) } {
            Ok(()) => Ok(()),
            Err(error) => {
                // Accepted::into_connection returns the alert bytes alongside
                // the error. Deliver those bytes before reporting failure.
                let _ = flush_transport(unsafe { &mut *socket });
                tls_result(Err(error))
            }
        }
    }

    #[crate::pyre_methods]
    impl W_SSLSocket {
        #[staticmethod]
        fn __new__(_cls: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
            Err(crate::PyError::type_error(
                "cannot create '_ssl._SSLSocket' instances",
            ))
        }

        #[getter]
        fn owner(&self) -> PyObjectRef {
            let owner =
                unsafe { pyre_object::weakref::w_gc_weakref_box_or_strong_deref(self.owner) };
            if owner.is_null() { w_none() } else { owner }
        }

        #[getter]
        fn server_side(&self) -> bool {
            self.server_side
        }

        #[getter]
        fn server_hostname(&self) -> PyObjectRef {
            self.server_hostname
        }

        #[getter]
        fn context(&self) -> PyObjectRef {
            self.context
        }

        #[setter]
        fn set_context(&mut self, value: PyObjectRef) -> Result<(), crate::PyError> {
            if W_SSLContext::from_obj(value).is_none() {
                return Err(crate::PyError::type_error("context must be an SSLContext"));
            }
            pyre_object::gc_hook::try_gc_write_barrier(self as *mut W_SSLSocket as *mut u8);
            self.context = value;
            Ok(())
        }

        #[getter]
        fn session_reused(&self) -> PyObjectRef {
            if self.backend.is_null() {
                return w_none();
            }
            unsafe { pyre_native::ssl::connection_session_reused(self.backend) }
                .map(w_bool_from)
                .unwrap_or_else(w_none)
        }

        #[getter]
        fn session(&self) -> PyObjectRef {
            if self.server_side || self.backend.is_null() {
                return w_none();
            }
            let backend = unsafe { pyre_native::ssl::connection_session(self.backend) };
            if backend.is_null() {
                w_none()
            } else {
                allocate_ssl_session(backend)
            }
        }

        #[setter]
        fn set_session(&mut self, value: PyObjectRef) -> Result<(), crate::PyError> {
            if !unsafe { is_none(value) } && W_SSLSession::from_obj(value).is_none() {
                return Err(crate::PyError::type_error("Value is not a SSLSession."));
            }
            if !self.backend.is_null()
                && !unsafe { pyre_native::ssl::connection_is_handshaking(self.backend) }
            {
                return Err(crate::PyError::value_error(
                    "Cannot set session after handshake.",
                ));
            }
            let replacement = clone_requested_session(Some(value), {
                let context =
                    W_SSLContext::from_obj(self.context).expect("SSL socket owns its context");
                context.backend
            })?;
            unsafe { pyre_native::ssl::session_free(self.requested_session) };
            self.requested_session = replacement;
            Ok(())
        }

        fn do_handshake(&mut self) -> Result<(), crate::PyError> {
            ensure_connection(self)?;
            if self.server_side {
                let context =
                    W_SSLContext::from_obj(self.context).expect("SSL socket owns its context");
                let owner =
                    unsafe { pyre_object::weakref::w_gc_weakref_box_or_strong_deref(self.owner) };
                if !unsafe { is_none(context.sni_callback) } && owner.is_null() {
                    return Err(tls_error(
                        pyre_native::ssl::TLS_ERROR_SSL,
                        "[SSL: PARSE_TLSEXT] SNI callback owner is no longer available".to_string(),
                    ));
                }
            }
            loop {
                unsafe { configure_accepted_server(self as *mut W_SSLSocket) }?;
                flush_transport(self)?;
                if !unsafe { pyre_native::ssl::connection_is_handshaking(self.backend) } {
                    return Ok(());
                }
                receive_transport(self)?;
                unsafe { configure_accepted_server(self as *mut W_SSLSocket) }?;
                if !unsafe { is_none(self.incoming) } {
                    flush_transport(self)?;
                    if unsafe { pyre_native::ssl::connection_is_handshaking(self.backend) } {
                        return Err(tls_error(
                            pyre_native::ssl::TLS_ERROR_WANT_READ,
                            "The operation did not complete (read)".to_string(),
                        ));
                    }
                    return Ok(());
                }
            }
        }

        fn write(&mut self, data: PyObjectRef) -> Result<usize, crate::PyError> {
            if self.shutdown_started {
                return Err(ssl_error("TLS/SSL connection has been closed"));
            }
            ensure_connection(self)?;
            let buffer = crate::baseobjspace::simple_buffer_bytes(data)?
                .ok_or_else(|| crate::PyError::type_error("a bytes-like object is required"))?;
            let result = tls_result(unsafe {
                pyre_native::ssl::connection_write_plain(self.backend, buffer.as_bytes())
            });
            buffer.release();
            let written = result?;
            flush_transport(self)?;
            Ok(written)
        }

        fn read(&mut self, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            ensure_connection(self)?;
            let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
            if crate::builtins::has_real_kwargs(kwargs) || positional.len() > 3 {
                return Err(crate::PyError::type_error(
                    "read() takes at most 2 arguments",
                ));
            }
            let requested = positional
                .get(1)
                .copied()
                .map(crate::baseobjspace::int_w)
                .transpose()?
                .unwrap_or(1024);
            let output_buffer = positional.get(2).copied();
            if requested < 0 && output_buffer.is_none() {
                return Err(crate::PyError::value_error("size should not be negative"));
            }
            if requested == 0 && output_buffer.is_none() {
                return Ok(w_bytes_from_bytes(&[]));
            }
            let mut writable = if let Some(buffer) = output_buffer {
                Some(
                    unsafe { crate::builtins::WritableBuffer::acquire(buffer) }.map_err(|_| {
                        crate::PyError::type_error("read() argument 2 must be a writable buffer")
                    })?,
                )
            } else {
                None
            };
            let size = if let Some(buffer) = writable.as_mut() {
                let capacity = unsafe { buffer.as_mut_slice() }.len();
                if requested < 0 {
                    capacity
                } else {
                    capacity.min(requested as usize)
                }
            } else {
                requested as usize
            };
            let data = loop {
                match unsafe { pyre_native::ssl::connection_read_plain(self.backend, size) } {
                    Ok(data) => break data,
                    Err((code, _)) if code == pyre_native::ssl::TLS_ERROR_WANT_READ => {
                        receive_transport(self)?;
                        flush_transport(self)?;
                    }
                    Err((code, message)) => return Err(tls_error(code, message)),
                }
            };
            if let Some(buffer) = writable.as_mut() {
                let target = unsafe { buffer.as_mut_slice() };
                target[..data.len()].copy_from_slice(&data);
                Ok(w_int_new(data.len() as i64))
            } else {
                Ok(w_bytes_from_bytes(&data))
            }
        }

        fn pending(&mut self) -> usize {
            if self.backend.is_null() {
                0
            } else {
                unsafe { pyre_native::ssl::connection_pending_plaintext(self.backend) }
            }
        }

        fn selected_alpn_protocol(&self) -> PyObjectRef {
            if self.backend.is_null() {
                return w_none();
            }
            unsafe { pyre_native::ssl::connection_alpn(self.backend) }
                .map(|value| w_str_new(&String::from_utf8_lossy(&value)))
                .unwrap_or_else(w_none)
        }

        fn version(&self) -> PyObjectRef {
            if self.backend.is_null() {
                return w_none();
            }
            unsafe { pyre_native::ssl::connection_version(self.backend) }
                .map(w_str_new)
                .unwrap_or_else(w_none)
        }

        fn compression(&self) -> PyObjectRef {
            w_none()
        }

        fn shared_ciphers(&self) -> PyObjectRef {
            if !self.server_side || self.backend.is_null() {
                return w_none();
            }
            let Some((name, bits)) = (unsafe { pyre_native::ssl::connection_cipher(self.backend) })
            else {
                return w_none();
            };
            let version =
                unsafe { pyre_native::ssl::connection_version(self.backend) }.unwrap_or("unknown");
            w_list_new(vec![w_tuple_new(vec![
                w_str_new(&name),
                w_str_new(version),
                w_int_new(bits as i64),
            ])])
        }

        fn cipher(&self) -> PyObjectRef {
            if self.backend.is_null() {
                return w_none();
            }
            let Some(version) = (unsafe { pyre_native::ssl::connection_version(self.backend) })
            else {
                return w_none();
            };
            let Some((name, bits)) = (unsafe { pyre_native::ssl::connection_cipher(self.backend) })
            else {
                return w_none();
            };
            w_tuple_new(vec![
                w_str_new(&name),
                w_str_new(version),
                w_int_new(bits as i64),
            ])
        }

        fn getpeercert(
            &self,
            #[default(false)] binary_form: bool,
        ) -> Result<PyObjectRef, crate::PyError> {
            if self.backend.is_null()
                || unsafe { pyre_native::ssl::connection_is_handshaking(self.backend) }
            {
                return Err(crate::PyError::value_error("handshake not done yet"));
            }
            let Some(der) =
                (unsafe { pyre_native::ssl::connection_peer_certificate(self.backend) })
            else {
                return Ok(w_none());
            };
            if binary_form {
                return Ok(w_bytes_from_bytes(&der));
            }
            let context =
                W_SSLContext::from_obj(self.context).expect("SSL socket owns its context");
            if unsafe { pyre_native::ssl::context_verify_mode(context.backend) } == CERT_NONE {
                return Ok(w_dict_new());
            }
            let decoded = native_result(pyre_native::ssl::certificate_decode_der(&der))?;
            Ok(decoded_certificate_dict(decoded))
        }

        fn get_channel_binding(
            &self,
            #[default("tls-unique")] kind: &str,
        ) -> Result<PyObjectRef, crate::PyError> {
            if kind != "tls-unique" {
                return Err(crate::PyError::value_error(
                    "'channel_binding_type' must be 'tls-unique'",
                ));
            }
            // CPython permits querying a lazily-created SSLObject before its
            // first handshake step.  There is no TLS channel to bind yet.
            if self.backend.is_null()
                || unsafe { pyre_native::ssl::connection_is_handshaking(self.backend) }
            {
                return Ok(w_none());
            }
            // RFC 5929 `tls-unique` is the TLS 1.2 Finished verify_data, which
            // rustls does not expose.  Any substitute would agree only with
            // another pyre peer, so a binding negotiated against an OpenSSL,
            // Java, or Go implementation would fail authentication with no
            // indication of why.  Report the gap instead of a private value.
            Err(crate::PyError::value_error(
                "'tls-unique' channel binding type not implemented",
            ))
        }

        fn shutdown(&mut self) -> Result<PyObjectRef, crate::PyError> {
            ensure_connection(self)?;
            if !self.shutdown_started {
                unsafe { pyre_native::ssl::connection_send_close_notify(self.backend) };
                self.shutdown_started = true;
            }
            flush_transport(self)?;
            if !unsafe { is_none(self.incoming) } {
                receive_transport(self)?;
                if !unsafe { pyre_native::ssl::connection_peer_closed(self.backend) } {
                    return Err(tls_error(
                        pyre_native::ssl::TLS_ERROR_WANT_READ,
                        "The operation did not complete (read)".to_string(),
                    ));
                }
            }
            if unsafe { is_none(self.socket) } {
                Ok(w_none())
            } else {
                transport_socket(self)
            }
        }
    }
} // ssl_socket_methods

mod ssl_session_methods {
    use super::*;

    #[crate::pyre_methods]
    impl W_SSLSession {
        #[staticmethod]
        fn __new__(_cls: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
            Err(crate::PyError::type_error(
                "cannot create '_ssl.SSLSession' instances",
            ))
        }

        #[getter]
        fn id(&self) -> PyObjectRef {
            w_bytes_from_bytes(&unsafe { pyre_native::ssl::session_id(self.backend) })
        }

        #[getter]
        fn time(&self) -> i64 {
            unsafe { pyre_native::ssl::session_creation_time(self.backend) as i64 }
        }

        #[getter]
        fn timeout(&self) -> i64 {
            unsafe { pyre_native::ssl::session_timeout(self.backend) as i64 }
        }

        #[getter]
        fn ticket_lifetime_hint(&self) -> i64 {
            unsafe { pyre_native::ssl::session_timeout(self.backend) as i64 }
        }

        #[getter]
        fn has_ticket(&self) -> bool {
            true
        }

        fn __eq__(&self, other: PyObjectRef) -> PyObjectRef {
            let Some(other) = W_SSLSession::from_obj(other) else {
                return pyre_object::w_not_implemented();
            };
            w_bool_from(
                unsafe { pyre_native::ssl::session_id(self.backend) }
                    == unsafe { pyre_native::ssl::session_id(other.backend) },
            )
        }

        fn __repr__(&self) -> PyObjectRef {
            w_str_new("<_ssl.SSLSession>")
        }
    }
} // ssl_session_methods

mod certificate_methods {
    use super::*;

    #[crate::pyre_methods]
    impl W_Certificate {
        #[staticmethod]
        fn __new__(_cls: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
            Err(crate::PyError::type_error(
                "cannot create '_ssl.Certificate' instances",
            ))
        }

        fn public_bytes(&self, #[default(2)] encoding: i64) -> Result<PyObjectRef, crate::PyError> {
            if encoding == 2 {
                return Ok(self.der);
            }
            Err(crate::PyError::value_error(
                "unsupported certificate encoding",
            ))
        }
    }
} // certificate_methods

/// Sweep-time cleanup for opaque native TLS state.
///
/// # Safety
/// `obj` must be a GC-dead instance of the named class.
pub unsafe fn w_ssl_context_dealloc(obj: PyObjectRef) {
    if let Some(context) = W_SSLContext::from_obj(obj) {
        unsafe { pyre_native::ssl::context_free(context.backend) };
        context.backend = std::ptr::null_mut();
    }
}

/// # Safety
/// `obj` must be a GC-dead `W_MemoryBIO`.
pub unsafe fn w_memory_bio_dealloc(obj: PyObjectRef) {
    if let Some(bio) = W_MemoryBIO::from_obj(obj) {
        unsafe { pyre_native::ssl::memory_bio_free(bio.backend) };
        bio.backend = std::ptr::null_mut();
    }
}

/// # Safety
/// `obj` must be a GC-dead `W_SSLSession`.
pub unsafe fn w_ssl_session_dealloc(obj: PyObjectRef) {
    if let Some(session) = W_SSLSession::from_obj(obj) {
        unsafe { pyre_native::ssl::session_free(session.backend) };
        session.backend = std::ptr::null_mut();
    }
}

/// # Safety
/// `obj` must be a GC-dead `W_SSLSocket`.
pub unsafe fn w_ssl_socket_dealloc(obj: PyObjectRef) {
    if let Some(socket) = W_SSLSocket::from_obj(obj) {
        unsafe { pyre_native::ssl::connection_free(socket.backend) };
        unsafe { pyre_native::ssl::session_free(socket.requested_session) };
        socket.backend = std::ptr::null_mut();
        socket.requested_session = std::ptr::null_mut();
    }
}

fn rand_status(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(w_bool_from(true))
}

fn rand_add(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(w_none())
}

fn rand_bytes(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let size = crate::baseobjspace::int_w(args[0])?;
    if size < 0 {
        return Err(crate::PyError::value_error("num must be positive"));
    }
    let size = usize::try_from(size)
        .map_err(|_| crate::PyError::overflow_error("RAND_bytes size out of range"))?;
    let bytes = crate::importing::host::os::urandom(size)
        .map_err(|err| ssl_error(format!("RAND_bytes failed: {err}")))?;
    Ok(w_bytes_from_bytes(&bytes))
}

#[derive(Clone, Copy)]
struct OidEntry {
    nid: i32,
    short_name: &'static str,
    long_name: &'static str,
    oid: &'static str,
}

const OIDS: &[OidEntry] = &[
    OidEntry {
        nid: 13,
        short_name: "CN",
        long_name: "commonName",
        oid: "2.5.4.3",
    },
    OidEntry {
        nid: 14,
        short_name: "C",
        long_name: "countryName",
        oid: "2.5.4.6",
    },
    OidEntry {
        nid: 15,
        short_name: "L",
        long_name: "localityName",
        oid: "2.5.4.7",
    },
    OidEntry {
        nid: 16,
        short_name: "ST",
        long_name: "stateOrProvinceName",
        oid: "2.5.4.8",
    },
    OidEntry {
        nid: 17,
        short_name: "O",
        long_name: "organizationName",
        oid: "2.5.4.10",
    },
    OidEntry {
        nid: 18,
        short_name: "OU",
        long_name: "organizationalUnitName",
        oid: "2.5.4.11",
    },
    OidEntry {
        nid: 129,
        short_name: "serverAuth",
        long_name: "TLS Web Server Authentication",
        oid: "1.3.6.1.5.5.7.3.1",
    },
    OidEntry {
        nid: 130,
        short_name: "clientAuth",
        long_name: "TLS Web Client Authentication",
        oid: "1.3.6.1.5.5.7.3.2",
    },
];

fn oid_tuple(entry: OidEntry) -> PyObjectRef {
    w_tuple_new(vec![
        w_int_new(entry.nid as i64),
        w_str_new(entry.short_name),
        w_str_new(entry.long_name),
        w_str_new(entry.oid),
    ])
}

fn txt2obj(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    if positional.is_empty() || positional.len() > 2 {
        return Err(crate::PyError::type_error(
            "txt2obj() takes 1 or 2 arguments",
        ));
    }
    let value = crate::baseobjspace::str_utf8_w(positional[0])?;
    let name_arg = crate::builtins::bind_pos_or_kw(positional, kwargs, 1, "name", "txt2obj", 2)?;
    crate::builtins::kwarg_reject_unknown(kwargs, &["name"], "txt2obj")?;
    let allow_names = name_arg
        .map(crate::baseobjspace::is_true)
        .transpose()?
        .unwrap_or(false);
    let name = value.as_ref();
    let entry = OIDS.iter().copied().find(|entry| {
        entry.oid == name
            || (allow_names
                && (entry.short_name == name || entry.long_name.eq_ignore_ascii_case(name)))
    });
    entry
        .map(oid_tuple)
        .ok_or_else(|| crate::PyError::value_error(format!("unknown object '{name}'")))
}

fn nid2obj(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let nid = crate::baseobjspace::int_w(args[0])?;
    OIDS.iter()
        .copied()
        .find(|entry| entry.nid as i64 == nid)
        .map(oid_tuple)
        .ok_or_else(|| crate::PyError::value_error(format!("unknown NID {nid}")))
}

fn get_default_verify_paths(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (cert_file, cert_dir) = pyre_native::ssl::default_verify_paths();
    Ok(w_tuple_new(vec![
        w_str_new("SSL_CERT_FILE"),
        w_str_new(&cert_file),
        w_str_new("SSL_CERT_DIR"),
        w_str_new(&cert_dir),
    ]))
}

fn test_decode_cert(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let path = path_string(args[0])?;
    let cert = native_result(pyre_native::ssl::certificate_decode_file(&path))?;
    Ok(decoded_certificate_dict(cert))
}

crate::py_module! {
    "_ssl",
    interpleveldefs: {
        "_SSLContext" => context_methods::type_object(),
        "MemoryBIO" => memory_bio_methods::type_object(),
        "SSLSession" => ssl_session_methods::type_object(),
        "_SSLSocket" => ssl_socket_methods::type_object(),
        "Certificate" => certificate_methods::type_object(),
        "SSLError" => crate::builtins::make_exc_type(
            "_ssl.SSLError",
            crate::builtins::exc_os_error_new,
            crate::builtins::lookup_exc_class("OSError").expect("OSError installed"),
        ),
        "SSLZeroReturnError" => crate::builtins::make_exc_type(
            "_ssl.SSLZeroReturnError",
            crate::builtins::exc_os_error_new,
            crate::builtins::lookup_exc_class("_ssl.SSLError").expect("SSLError installed"),
        ),
        "SSLWantReadError" => crate::builtins::make_exc_type(
            "_ssl.SSLWantReadError",
            crate::builtins::exc_os_error_new,
            crate::builtins::lookup_exc_class("_ssl.SSLError").expect("SSLError installed"),
        ),
        "SSLWantWriteError" => crate::builtins::make_exc_type(
            "_ssl.SSLWantWriteError",
            crate::builtins::exc_os_error_new,
            crate::builtins::lookup_exc_class("_ssl.SSLError").expect("SSLError installed"),
        ),
        "SSLSyscallError" => crate::builtins::make_exc_type(
            "_ssl.SSLSyscallError",
            crate::builtins::exc_os_error_new,
            crate::builtins::lookup_exc_class("_ssl.SSLError").expect("SSLError installed"),
        ),
        "SSLEOFError" => crate::builtins::make_exc_type(
            "_ssl.SSLEOFError",
            crate::builtins::exc_os_error_new,
            crate::builtins::lookup_exc_class("_ssl.SSLError").expect("SSLError installed"),
        ),
        // OpenSSL-shaped compatibility fields are required by `ssl.py` and
        // consumers such as urllib3, while the suffix names the real backend.
        "OPENSSL_VERSION" => w_str_new("OpenSSL 3.0.0-compatible (AWS-LC/rustls 0.23)"),
        "OPENSSL_VERSION_NUMBER" => w_int_new(0x3000_0000),
        "OPENSSL_VERSION_INFO" => w_tuple_new(vec![w_int_new(3), w_int_new(0), w_int_new(0), w_int_new(0), w_int_new(15)]),
        "_OPENSSL_API_VERSION" => w_tuple_new(vec![w_int_new(3), w_int_new(0), w_int_new(0), w_int_new(0), w_int_new(15)]),
        "_DEFAULT_CIPHERS" => w_str_new("rustls default cipher suites"),
        "HAS_SNI" => w_bool_from(true),
        "HAS_ECDH" => w_bool_from(true),
        "HAS_NPN" => w_bool_from(false),
        "HAS_ALPN" => w_bool_from(true),
        "HAS_SSLv2" => w_bool_from(false),
        "HAS_SSLv3" => w_bool_from(false),
        "HAS_TLSv1" => w_bool_from(false),
        "HAS_TLSv1_1" => w_bool_from(false),
        "HAS_TLSv1_2" => w_bool_from(true),
        "HAS_TLSv1_3" => w_bool_from(true),
        "HAS_PSK" => w_bool_from(false),
        "HAS_PHA" => w_bool_from(false)
    },
    int_constants: {
        "PROTOCOL_SSLv23" => PROTOCOL_TLS,
        "PROTOCOL_TLS" => PROTOCOL_TLS,
        "PROTOCOL_TLS_CLIENT" => PROTOCOL_TLS_CLIENT,
        "PROTOCOL_TLS_SERVER" => PROTOCOL_TLS_SERVER,
        "PROTOCOL_TLSv1" => 3,
        "PROTOCOL_TLSv1_1" => 4,
        "PROTOCOL_TLSv1_2" => 5,
        "PROTOCOL_TLSv1_3" => 6,
        "PROTO_MINIMUM_SUPPORTED" => -2,
        "PROTO_MAXIMUM_SUPPORTED" => -1,
        "PROTO_SSLv3" => 0x300,
        "PROTO_TLSv1" => 0x301,
        "PROTO_TLSv1_1" => 0x302,
        "PROTO_TLSv1_2" => 0x303,
        "PROTO_TLSv1_3" => 0x304,
        "CERT_NONE" => CERT_NONE,
        "CERT_OPTIONAL" => CERT_OPTIONAL,
        "CERT_REQUIRED" => CERT_REQUIRED,
        "VERIFY_DEFAULT" => 0,
        "VERIFY_CRL_CHECK_LEAF" => 4,
        "VERIFY_CRL_CHECK_CHAIN" => 12,
        "VERIFY_X509_STRICT" => 32,
        "VERIFY_ALLOW_PROXY_CERTS" => 64,
        "VERIFY_X509_TRUSTED_FIRST" => 32768,
        "VERIFY_X509_PARTIAL_CHAIN" => 0x80000,
        "OP_ALL" => 0x00000bfb,
        "OP_NO_SSLv2" => 0,
        "OP_NO_SSLv3" => 0x02000000,
        "OP_NO_TLSv1" => 0x04000000,
        "OP_NO_TLSv1_1" => 0x10000000,
        "OP_NO_TLSv1_2" => 0x08000000,
        "OP_NO_TLSv1_3" => 0x20000000,
        "OP_NO_COMPRESSION" => 0x00020000,
        "OP_CIPHER_SERVER_PREFERENCE" => 0x00400000,
        "OP_SINGLE_DH_USE" => 0,
        "OP_SINGLE_ECDH_USE" => 0,
        "OP_NO_TICKET" => 0x00004000,
        "OP_LEGACY_SERVER_CONNECT" => 4,
        "OP_NO_RENEGOTIATION" => 0x40000000,
        "OP_IGNORE_UNEXPECTED_EOF" => 0x80,
        "OP_ENABLE_MIDDLEBOX_COMPAT" => 0x00100000,
        "SSL_ERROR_NONE" => 0,
        "SSL_ERROR_SSL" => 1,
        "SSL_ERROR_WANT_READ" => 2,
        "SSL_ERROR_WANT_WRITE" => 3,
        "SSL_ERROR_WANT_X509_LOOKUP" => 4,
        "SSL_ERROR_SYSCALL" => 5,
        "SSL_ERROR_ZERO_RETURN" => 6,
        "SSL_ERROR_WANT_CONNECT" => 7,
        "SSL_ERROR_EOF" => 8,
        "SSL_ERROR_INVALID_ERROR_CODE" => 10,
        "HOSTFLAG_NEVER_CHECK_SUBJECT" => 0x20,
        "ENCODING_PEM" => 1,
        "ENCODING_DER" => 2,
        "ENCODING_PEM_AUX" => 0x101
    },
    functions: {
        "RAND_status" / 0 = rand_status,
        "RAND_add" / * = rand_add,
        "RAND_bytes" / 1 = rand_bytes,
        "txt2obj" / * = txt2obj,
        "nid2obj" / 1 = nid2obj,
        "get_default_verify_paths" / 0 = get_default_verify_paths,
        "_test_decode_cert" / 1 = test_decode_cert
    },
    extra_init: |ns| {
        let ssl_error = crate::builtins::lookup_exc_class("_ssl.SSLError")
            .expect("SSLError installed");
        let ssl_error_dict =
            unsafe { pyre_object::w_type_get_dict_ptr(ssl_error) as PyObjectRef };
        unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ssl_error_dict,
                "__str__",
                crate::make_builtin_function_with_arity("__str__", ssl_error_str, 1),
            );
        }
        let value_error = crate::builtins::lookup_exc_class("ValueError")
            .expect("ValueError installed");
        let cert_error = crate::builtins::make_exc_type_multi(
            "_ssl.SSLCertVerificationError",
            crate::builtins::exc_exception_new,
            &[ssl_error, value_error],
        );
        crate::module_ns_store(ns, "SSLCertVerificationError", cert_error);
        for (name, value) in [
            ("ALERT_DESCRIPTION_CLOSE_NOTIFY", 0),
            ("ALERT_DESCRIPTION_UNEXPECTED_MESSAGE", 10),
            ("ALERT_DESCRIPTION_BAD_RECORD_MAC", 20),
            ("ALERT_DESCRIPTION_DECRYPTION_FAILED", 21),
            ("ALERT_DESCRIPTION_RECORD_OVERFLOW", 22),
            ("ALERT_DESCRIPTION_DECOMPRESSION_FAILURE", 30),
            ("ALERT_DESCRIPTION_HANDSHAKE_FAILURE", 40),
            ("ALERT_DESCRIPTION_NO_CERTIFICATE", 41),
            ("ALERT_DESCRIPTION_BAD_CERTIFICATE", 42),
            ("ALERT_DESCRIPTION_UNSUPPORTED_CERTIFICATE", 43),
            ("ALERT_DESCRIPTION_CERTIFICATE_REVOKED", 44),
            ("ALERT_DESCRIPTION_CERTIFICATE_EXPIRED", 45),
            ("ALERT_DESCRIPTION_CERTIFICATE_UNKNOWN", 46),
            ("ALERT_DESCRIPTION_ILLEGAL_PARAMETER", 47),
            ("ALERT_DESCRIPTION_UNKNOWN_CA", 48),
            ("ALERT_DESCRIPTION_ACCESS_DENIED", 49),
            ("ALERT_DESCRIPTION_DECODE_ERROR", 50),
            ("ALERT_DESCRIPTION_DECRYPT_ERROR", 51),
            ("ALERT_DESCRIPTION_EXPORT_RESTRICTION", 60),
            ("ALERT_DESCRIPTION_PROTOCOL_VERSION", 70),
            ("ALERT_DESCRIPTION_INSUFFICIENT_SECURITY", 71),
            ("ALERT_DESCRIPTION_INTERNAL_ERROR", 80),
            ("ALERT_DESCRIPTION_INAPPROPRIATE_FALLBACK", 86),
            ("ALERT_DESCRIPTION_USER_CANCELLED", 90),
            ("ALERT_DESCRIPTION_NO_RENEGOTIATION", 100),
            ("ALERT_DESCRIPTION_MISSING_EXTENSION", 109),
            ("ALERT_DESCRIPTION_UNSUPPORTED_EXTENSION", 110),
            ("ALERT_DESCRIPTION_CERTIFICATE_UNOBTAINABLE", 111),
            ("ALERT_DESCRIPTION_UNRECOGNIZED_NAME", 112),
            ("ALERT_DESCRIPTION_BAD_CERTIFICATE_STATUS_RESPONSE", 113),
            ("ALERT_DESCRIPTION_BAD_CERTIFICATE_HASH_VALUE", 114),
            ("ALERT_DESCRIPTION_UNKNOWN_PSK_IDENTITY", 115),
            ("ALERT_DESCRIPTION_CERTIFICATE_REQUIRED", 116),
            ("ALERT_DESCRIPTION_NO_APPLICATION_PROTOCOL", 120),
        ] {
            crate::module_ns_store(ns, name, w_int_new(value));
        }
    },
}
