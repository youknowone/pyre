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

pub fn write_exception(out: &mut Vec<u8>, exception: &str, tb: Option<&str>) {
    out.extend_from_slice(exception.as_bytes());
    if let Some(tb) = tb {
        out.extend_from_slice(tb.as_bytes());
    }
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
}
