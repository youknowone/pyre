//! RPython `rpython/translator/sandbox/sandlib.py`.

use crate::translator::sandbox::_marshal::{MarshalError, MarshalValue, dump, load};

pub fn create_log() -> Vec<String> {
    Vec::new()
}

pub fn read_message(input: &[u8]) -> Result<MarshalValue, MarshalError> {
    let mut cursor = std::io::Cursor::new(input);
    load(&mut cursor)
}

pub fn write_message(
    out: &mut Vec<u8>,
    msg: &MarshalValue,
    _resulttype: Option<&str>,
) -> Result<(), MarshalError> {
    dump(msg, out, crate::translator::sandbox::_marshal::version)
}

/// External-call exception classes framed for the sandboxed child. Keep the
/// table in sync with `rsandbox::reraise_error()`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExcClass {
    OSError,
    IOError,
    OverflowError,
    ValueError,
    ZeroDivisionError,
    MemoryError,
    KeyError,
    IndexError,
    RuntimeError,
}

pub const EXCEPTION_TABLE: &[(i64, ExcClass)] = &[
    (1, ExcClass::OSError),
    (2, ExcClass::IOError),
    (3, ExcClass::OverflowError),
    (4, ExcClass::ValueError),
    (5, ExcClass::ZeroDivisionError),
    (6, ExcClass::MemoryError),
    (7, ExcClass::KeyError),
    (8, ExcClass::IndexError),
    (9, ExcClass::RuntimeError),
];

const EPERM: i32 = 1;

/// Frame an external-call exception for the child: a marshalled error code from
/// `EXCEPTION_TABLE`, plus the marshalled errno for `OSError` (defaulting to
/// `EPERM`). An exception with no table entry returns `Err`, mirroring the
/// upstream `raise exception.__class__, exception, tb` re-raise.
pub fn write_exception(
    out: &mut Vec<u8>,
    exception: ExcClass,
    errno: Option<i32>,
) -> Result<(), MarshalError> {
    for &(code, excclass) in EXCEPTION_TABLE {
        if exception == excclass {
            write_message(out, &MarshalValue::Int(code), None)?;
            if excclass == ExcClass::OSError {
                let error = errno.unwrap_or(EPERM);
                write_message(out, &MarshalValue::Int(error as i64), None)?;
            }
            return Ok(());
        }
    }
    // just re-raise the exception
    Err(MarshalError::new(
        "sandlib.py: write_exception: exception not in EXCEPTION_TABLE",
    ))
}

pub fn shortrepr(x: &str) -> String {
    const MAX: usize = 80;
    if x.len() <= MAX {
        x.to_string()
    } else {
        format!("{}...", &x[..MAX])
    }
}

pub fn signal_name(n: i32) -> String {
    format!("signal {n}")
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SandboxedProc {
    pub args: Vec<String>,
    pub os_level_sandboxing: bool,
}

impl SandboxedProc {
    pub fn new(args: Vec<String>) -> Self {
        Self {
            args,
            os_level_sandboxing: false,
        }
    }

    pub fn interact(&self) -> Result<(), String> {
        Err("sandlib.py: SandboxedProc.interact subprocess control is not ported yet".to_string())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SimpleIOSandboxedProc {
    pub proc: SandboxedProc,
}

impl SimpleIOSandboxedProc {
    pub fn new(args: Vec<String>) -> Self {
        Self {
            proc: SandboxedProc::new(args),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VirtualizedSandboxedProc {
    pub proc: SandboxedProc,
}

impl VirtualizedSandboxedProc {
    pub fn new(args: Vec<String>) -> Self {
        Self {
            proc: SandboxedProc::new(args),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VirtualizedSocketProc {
    pub proc: VirtualizedSandboxedProc,
}

impl VirtualizedSocketProc {
    pub fn new(args: Vec<String>) -> Self {
        Self {
            proc: VirtualizedSandboxedProc::new(args),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_round_trip_uses_marshal_surface() {
        let msg = MarshalValue::String("ll_os.ll_os_open".to_string());
        let mut out = Vec::new();
        write_message(&mut out, &msg, None).unwrap();
        assert_eq!(read_message(&out).unwrap(), msg);
    }

    #[test]
    fn write_exception_frames_oserror_with_errno() {
        let mut out = Vec::new();
        write_exception(&mut out, ExcClass::OSError, Some(2)).unwrap();
        let mut cursor = std::io::Cursor::new(out);
        assert_eq!(load(&mut cursor).unwrap(), MarshalValue::Int(1));
        assert_eq!(load(&mut cursor).unwrap(), MarshalValue::Int(2));
    }

    #[test]
    fn write_exception_frames_plain_code() {
        let mut out = Vec::new();
        write_exception(&mut out, ExcClass::ValueError, None).unwrap();
        assert_eq!(read_message(&out).unwrap(), MarshalValue::Int(4));
    }
}
