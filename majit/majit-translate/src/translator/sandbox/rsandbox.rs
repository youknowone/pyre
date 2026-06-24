//! RPython `rpython/translator/sandbox/rsandbox.py`.

use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SandboxError {
    OSError(i64),
    IOError,
    OverflowError,
    ValueError,
    ZeroDivisionError,
    MemoryError,
    KeyError,
    IndexError,
    RuntimeError,
}

impl fmt::Display for SandboxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for SandboxError {}

pub fn writeall_not_sandboxed(fd: i32, buf: &[u8], out: &mut Vec<u8>) -> Result<(), SandboxError> {
    if fd < 0 {
        return Err(SandboxError::IOError);
    }
    out.extend_from_slice(buf);
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FdLoader {
    pub fd: i32,
    pub buf: Vec<u8>,
    pub buflen: usize,
}

impl FdLoader {
    pub fn new(fd: i32) -> Self {
        Self {
            fd,
            buf: Vec::new(),
            buflen: 4096,
        }
    }
}

pub fn sandboxed_io(buf: &[u8]) -> Result<FdLoader, SandboxError> {
    let mut stdout = Vec::new();
    writeall_not_sandboxed(1, buf, &mut stdout)?;
    Ok(FdLoader::new(0))
}

pub fn reraise_error(error: i64) -> Result<(), SandboxError> {
    match error {
        0 => Ok(()),
        1 => Err(SandboxError::OSError(0)),
        2 => Err(SandboxError::IOError),
        3 => Err(SandboxError::OverflowError),
        4 => Err(SandboxError::ValueError),
        5 => Err(SandboxError::ZeroDivisionError),
        6 => Err(SandboxError::MemoryError),
        7 => Err(SandboxError::KeyError),
        8 => Err(SandboxError::IndexError),
        _ => Err(SandboxError::RuntimeError),
    }
}

pub fn not_implemented_stub(msg: &str) -> Result<(), SandboxError> {
    let _ = msg;
    Err(SandboxError::RuntimeError)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SandboxStub {
    pub fnname: String,
    pub msg: String,
    pub __name__: String,
}

pub fn make_stub(fnname: &str, msg: &str) -> SandboxStub {
    SandboxStub {
        fnname: fnname.to_string(),
        msg: msg.to_string(),
        __name__: format!("sandboxed_{fnname}"),
    }
}

pub fn sig_ll(fnobj_name: &str) -> Result<(Vec<String>, String), SandboxError> {
    let _ = fnobj_name;
    Err(SandboxError::RuntimeError)
}

pub fn get_sandbox_stub(fnname: &str) -> SandboxStub {
    let msg = format!("Not implemented: sandboxing for external function '{fnname}'");
    make_stub(fnname, &msg)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SandboxTrampoline {
    pub fnname: String,
    pub __name__: String,
    pub args_s: Vec<String>,
    pub s_result: String,
}

pub fn make_sandbox_trampoline(
    fnname: &str,
    args_s: Vec<String>,
    s_result: String,
) -> SandboxTrampoline {
    SandboxTrampoline {
        fnname: fnname.to_string(),
        __name__: format!("sandboxed_{fnname}"),
        args_s,
        s_result,
    }
}

pub fn _annotate(
    _rtyper: &str,
    f: SandboxTrampoline,
    _args_s: &[String],
    _s_result: &str,
) -> SandboxTrampoline {
    f
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reraise_error_maps_protocol_codes() {
        assert_eq!(reraise_error(0), Ok(()));
        assert_eq!(reraise_error(2), Err(SandboxError::IOError));
        assert_eq!(reraise_error(8), Err(SandboxError::IndexError));
        assert_eq!(reraise_error(99), Err(SandboxError::RuntimeError));
    }

    #[test]
    fn make_sandbox_trampoline_sets_upstream_name() {
        let trampoline =
            make_sandbox_trampoline("os_open", vec!["int".to_string()], "str".to_string());
        assert_eq!(trampoline.__name__, "sandboxed_os_open");
    }
}
