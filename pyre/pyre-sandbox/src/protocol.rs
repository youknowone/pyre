//! Shared sandbox protocol types: the error taxonomy exchanged over the wire and
//! the result-kind tags that drive client-side decoding.
//!
//! The error codes are the contract between `rsandbox.reraise_error`
//! (`rpython/translator/sandbox/rsandbox.py:90-108`) and `sandlib.write_exception`
//! / `EXCEPTION_TABLE` (`rpython/translator/sandbox/sandlib.py:69-94`). The two
//! tables MUST stay in sync; they are unified here.

/// An error raised across the sandbox boundary.
///
/// The numeric `Os` payload is the errno carried as a second marshalled int after
/// the exception code (`rsandbox.py:92`). `Protocol` is local to this port: it
/// covers malformed marshal data / EOF on the pipe, which RPython surfaces as an
/// `IOError`/`ValueError` from the loader.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SandboxError {
    /// `OSError(errno, ...)` — code 1, followed by the errno int.
    Os(i32),
    /// `IOError` — code 2.
    Io,
    /// `OverflowError` — code 3.
    Overflow,
    /// `ValueError` — code 4.
    Value,
    /// `ZeroDivisionError` — code 5.
    ZeroDivision,
    /// `MemoryError` — code 6.
    Memory,
    /// `KeyError` — code 7.
    Key,
    /// `IndexError` — code 8.
    Index,
    /// `RuntimeError` — code 9 (and the catch-all).
    Runtime,
    /// Malformed wire data / unexpected EOF (loader-level failure).
    Protocol(String),
}

impl std::fmt::Display for SandboxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SandboxError::Os(e) => write!(f, "OSError({e})"),
            SandboxError::Io => f.write_str("IOError"),
            SandboxError::Overflow => f.write_str("OverflowError"),
            SandboxError::Value => f.write_str("ValueError"),
            SandboxError::ZeroDivision => f.write_str("ZeroDivisionError"),
            SandboxError::Memory => f.write_str("MemoryError"),
            SandboxError::Key => f.write_str("KeyError"),
            SandboxError::Index => f.write_str("IndexError"),
            SandboxError::Runtime => f.write_str("RuntimeError"),
            SandboxError::Protocol(msg) => write!(f, "protocol error: {msg}"),
        }
    }
}

impl std::error::Error for SandboxError {}

pub type SandboxResult<T> = Result<T, SandboxError>;

/// Map an exception code (the first int of a reply) to a [`SandboxError`].
///
/// For code 1 (`OSError`) the caller must subsequently read the errno int and
/// patch it into the returned `Os(0)`. Mirrors `rsandbox.reraise_error`.
pub fn error_from_code(code: i64) -> SandboxError {
    match code {
        1 => SandboxError::Os(0),
        2 => SandboxError::Io,
        3 => SandboxError::Overflow,
        4 => SandboxError::Value,
        5 => SandboxError::ZeroDivision,
        6 => SandboxError::Memory,
        7 => SandboxError::Key,
        8 => SandboxError::Index,
        _ => SandboxError::Runtime,
    }
}

/// Map a [`SandboxError`] to its wire code. Mirrors `sandlib.EXCEPTION_TABLE`.
pub fn code_for_error(err: &SandboxError) -> i64 {
    match err {
        SandboxError::Os(_) => 1,
        SandboxError::Io => 2,
        SandboxError::Overflow => 3,
        SandboxError::Value => 4,
        SandboxError::ZeroDivision => 5,
        SandboxError::Memory => 6,
        SandboxError::Key => 7,
        SandboxError::Index => 8,
        SandboxError::Runtime | SandboxError::Protocol(_) => 9,
    }
}

/// The static shape of a reply value, so the client knows which loader to run
/// after the success code. Each variant corresponds to a `load_result`
/// specialization in `rsandbox.make_sandbox_trampoline`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResultKind {
    /// `None` reply (e.g. `ll_os_close`).
    None,
    /// A plain int (`'i'`/`'I'`).
    Int,
    /// A forced 64-bit int (`RESULTTYPE_LONGLONG`, e.g. `ll_os_lseek`).
    LongLong,
    /// A bool (`'T'`/`'F'`).
    Bool,
    /// A byte string (`ll_os_read`, `ll_os_getcwd`, ...).
    Str,
    /// `str | None` (`ll_os_getenv`).
    OptStr,
    /// A list of byte strings (`ll_os_listdir`).
    ListStr,
    /// A list of `(str, str)` pairs (`ll_os_envitems`).
    EnvItems,
    /// A hand-packed stat result (`RESULTTYPE_STATRESULT`).
    StatResult,
    /// A float (`ll_time_time` / `ll_time_clock`).
    Float,
}
