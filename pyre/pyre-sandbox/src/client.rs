//! The untrusted-client half of the sandbox — port of
//! `rpython/translator/sandbox/rsandbox.py`.
//!
//! Inside `pyre --features sandbox`, every external OS call is replaced by a
//! trampoline that marshals `(fnname, args)` to STDOUT and reads the reply from
//! STDIN using only the raw, *un-sandboxed* `read`/`write` syscalls on fds 0/1
//! (so the protocol I/O is not itself re-sandboxed). [`syscall`] is the single
//! generic trampoline body that `host_seam::TrampolineHost` calls; RPython
//! instead emits one `make_sandbox_trampoline` closure per external, but the
//! runtime behaviour is identical.

use std::os::raw::c_void;

use crate::protocol::{ResultKind, SandboxError, SandboxResult, error_from_code};
use crate::rmarshal::{
    IntFlavor, Loader, MarshalValue, NeedMore, dump_string, dump_tuple, load_bool, load_float,
    load_int, load_longlong, load_statresult, load_string, load_value,
};
use crate::vfs::StatResult;

/// A decoded reply payload — the typed Rust value a trampoline returns, keyed by
/// the [`ResultKind`] requested. (The wire `MarshalValue` cannot represent a stat
/// result, so decoding lands in this richer enum rather than `MarshalValue`.)
#[derive(Clone, Debug, PartialEq)]
pub enum SyscallResult {
    None,
    Int(i64),
    Bool(bool),
    Str(Vec<u8>),
    OptStr(Option<Vec<u8>>),
    ListStr(Vec<Vec<u8>>),
    EnvItems(Vec<(Vec<u8>, Vec<u8>)>),
    Stat(StatResult),
    Float(f64),
}

// rsandbox.py:73 — the protocol talks over the raw process fds.
const STDIN_FD: i32 = 0;
const STDOUT_FD: i32 = 1;

/// A [`NeedMore`] that refills from a raw fd via the un-sandboxed `read(2)` — the
/// reply pipe (STDIN). `buflen` starts at 4096 and doubles on every refill,
/// matching `rsandbox.FdLoader.need_more_data` (rsandbox.py:54-70). A `FdLoader`
/// is therefore `Loader<FdNeedMore>`.
pub struct FdNeedMore {
    fd: i32,
    buflen: usize,
}

impl FdNeedMore {
    pub fn new(fd: i32) -> Self {
        FdNeedMore { fd, buflen: 4096 }
    }
}

impl NeedMore for FdNeedMore {
    // rsandbox.py:60 `need_more_data`.
    fn need_more(&mut self) -> SandboxResult<Vec<u8>> {
        let mut buf = vec![0u8; self.buflen];
        // SAFETY: read(2) into a buffer we own and sized; raw un-sandboxed read.
        // Retry on EINTR so a delivered signal does not spuriously fail the
        // protocol read.
        let count = loop {
            let n = unsafe { libc::read(self.fd, buf.as_mut_ptr() as *mut c_void, self.buflen) };
            if n < 0 && std::io::Error::last_os_error().raw_os_error() == Some(libc::EINTR) {
                continue;
            }
            break n;
        };
        if count <= 0 {
            return Err(SandboxError::Io);
        }
        buf.truncate(count as usize);
        self.buflen *= 2;
        Ok(buf)
    }
}

/// `FdLoader(fd)` — a marshal loader fed by the raw fd.
pub fn fd_loader(fd: i32) -> Loader<FdNeedMore> {
    Loader::new(Vec::new(), FdNeedMore::new(fd))
}

/// rsandbox.py:42 `writeall_not_sandboxed` — write the whole buffer with the raw
/// un-sandboxed `write(2)`, looping over partial writes, IOError on `count <= 0`.
pub fn writeall_not_sandboxed(fd: i32, mut buf: &[u8]) -> SandboxResult<()> {
    while !buf.is_empty() {
        // SAFETY: write(2) from a slice we hold for the duration of the call.
        // Retry on EINTR so a delivered signal does not spuriously fail the write.
        let count = loop {
            let n = unsafe { libc::write(fd, buf.as_ptr() as *const c_void, buf.len()) };
            if n < 0 && std::io::Error::last_os_error().raw_os_error() == Some(libc::EINTR) {
                continue;
            }
            break n;
        };
        if count <= 0 {
            return Err(SandboxError::Io);
        }
        buf = &buf[count as usize..];
    }
    Ok(())
}

/// rsandbox.py:90 `reraise_error` — map the leading exception code to a
/// [`SandboxError`]. Code 1 (`OSError`) reads a second int as the errno;
/// 2..=8 map via [`error_from_code`]; anything else is `RuntimeError`.
fn reraise_error<N: NeedMore>(error: i64, loader: &mut Loader<N>) -> SandboxError {
    if error == 1 {
        match load_int(loader) {
            Ok(errno) => SandboxError::Os(errno as i32),
            Err(e) => e,
        }
    } else {
        error_from_code(error)
    }
}

/// rsandbox.py:111 `not_implemented_stub` — raise `RuntimeError` for an external
/// whose signature cannot be marshalled. The RPython original also writes `msg`
/// to fd 2; this port omits that write (matching the majit sibling mirror) so
/// the untrusted child never writes directly to an inherited host fd.
pub fn not_implemented_stub(msg: &str) -> SandboxError {
    let _ = msg;
    SandboxError::Runtime
}

/// rsandbox.py:158-161 — marshal `fnname` (TYPE_STRING) followed by the argument
/// TUPLE, with the client int convention (`IntFlavor::Rmarshal` = every int as
/// TYPE_INT64). Frozen by `rmarshal::tests::golden_request_open`.
pub fn encode_request(fnname: &str, args: &[MarshalValue]) -> Vec<u8> {
    let mut buf = Vec::new();
    dump_string(&mut buf, fnname.as_bytes());
    dump_tuple(&mut buf, args, IntFlavor::Rmarshal);
    buf
}

// Project a marshalled string value to raw bytes.
fn as_bytes(v: MarshalValue) -> SandboxResult<Vec<u8>> {
    match v {
        MarshalValue::Str(s) => Ok(s),
        _ => Err(SandboxError::Protocol("expected string element".into())),
    }
}

/// rsandbox.py:152,165 `load_result = rmarshal.get_loader(s_result)` — decode the
/// typed reply payload by dispatching on [`ResultKind`]. The reply shapes are
/// exactly what `sandlib::encode_reply` emits.
pub fn load_result<N: NeedMore>(
    loader: &mut Loader<N>,
    kind: ResultKind,
) -> SandboxResult<SyscallResult> {
    match kind {
        ResultKind::None => match load_value(loader)? {
            MarshalValue::None => Ok(SyscallResult::None),
            _ => Err(SandboxError::Protocol("expected None reply".into())),
        },
        ResultKind::Int => Ok(SyscallResult::Int(load_int(loader)?)),
        ResultKind::LongLong => Ok(SyscallResult::Int(load_longlong(loader)?)),
        ResultKind::Bool => Ok(SyscallResult::Bool(load_bool(loader)?)),
        ResultKind::Str => Ok(SyscallResult::Str(load_string(loader)?)),
        ResultKind::OptStr => match load_value(loader)? {
            MarshalValue::None => Ok(SyscallResult::OptStr(None)),
            MarshalValue::Str(s) => Ok(SyscallResult::OptStr(Some(s))),
            _ => Err(SandboxError::Protocol("expected str|None reply".into())),
        },
        ResultKind::ListStr => match load_value(loader)? {
            MarshalValue::List(items) => Ok(SyscallResult::ListStr(
                items
                    .into_iter()
                    .map(as_bytes)
                    .collect::<SandboxResult<_>>()?,
            )),
            _ => Err(SandboxError::Protocol("expected list[str] reply".into())),
        },
        ResultKind::EnvItems => match load_value(loader)? {
            MarshalValue::List(items) => {
                let mut pairs = Vec::with_capacity(items.len());
                for item in items {
                    match item {
                        MarshalValue::Tuple(mut kv) if kv.len() == 2 => {
                            let value = as_bytes(kv.pop().unwrap())?;
                            let key = as_bytes(kv.pop().unwrap())?;
                            pairs.push((key, value));
                        }
                        _ => {
                            return Err(SandboxError::Protocol(
                                "expected (str,str) env item".into(),
                            ));
                        }
                    }
                }
                Ok(SyscallResult::EnvItems(pairs))
            }
            _ => Err(SandboxError::Protocol(
                "expected list[(str,str)] reply".into(),
            )),
        },
        ResultKind::StatResult => Ok(SyscallResult::Stat(load_statresult(loader)?)),
        ResultKind::Float => Ok(SyscallResult::Float(load_float(loader)?)),
    }
}

/// rsandbox.py:82-88 `sandboxed_io` — write the request, build a loader on the
/// reply pipe, read the leading code, and either raise or return the loader
/// positioned at the result payload.
fn sandboxed_io(buf: &[u8]) -> SandboxResult<Loader<FdNeedMore>> {
    writeall_not_sandboxed(STDOUT_FD, buf)?;
    let mut loader = fd_loader(STDIN_FD);
    let error = load_int(&mut loader)?;
    if error != 0 {
        return Err(reraise_error(error, &mut loader));
    }
    Ok(loader)
}

/// The generic trampoline body (rsandbox.py:157-167 `execute`). Marshals the
/// request, performs the round-trip, decodes the typed result, and asserts the
/// reply was fully consumed.
pub fn syscall(
    fnname: &str,
    args: &[MarshalValue],
    kind: ResultKind,
) -> SandboxResult<SyscallResult> {
    let buf = encode_request(fnname, args);
    let mut loader = sandboxed_io(&buf)?;
    let result = load_result(&mut loader, kind)?;
    loader.check_finished()?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sandlib::{Console, SandboxPolicy};
    use crate::vfs::{Dir, File, FsNode};
    use indexmap::IndexMap;
    use std::rc::Rc;

    // Round-trip the CLIENT encode/decode against the already-verified
    // controller, with no real fds: client encodes a request, the controller
    // services it, and the client decodes the controller's reply bytes. This
    // proves encode_request + load_result + reraise_error agree with
    // sandlib::encode_reply on the wire.
    fn files_root() -> FsNode {
        let mut map: IndexMap<String, FsNode> = IndexMap::new();
        map.insert("hi.txt".into(), Rc::new(File::new("Hello, world!\n")));
        Rc::new(Dir::new(map))
    }

    fn serve(policy: &mut SandboxPolicy, request: &[u8]) -> Vec<u8> {
        let mut replies = Vec::new();
        let (i, mut o, mut e) = (Vec::new(), Vec::new(), Vec::new());
        let mut console = Console {
            input: &mut i.as_slice(),
            output: &mut o,
            error: &mut e,
            input_isatty: false,
        };
        policy
            .handle_until_return(request, &mut replies, &mut console)
            .unwrap();
        replies
    }

    fn decode(replies: Vec<u8>, kind: ResultKind) -> SandboxResult<SyscallResult> {
        let mut loader = Loader::from_bytes(replies);
        let error = load_int(&mut loader)?;
        if error != 0 {
            return Err(reraise_error(error, &mut loader));
        }
        let result = load_result(&mut loader, kind)?;
        loader.check_finished()?;
        Ok(result)
    }

    #[test]
    fn client_getcwd_roundtrip() {
        let mut p = SandboxPolicy::new(files_root(), "/tmp", vec![], false);
        let req = encode_request("ll_os.ll_os_getcwd", &[]);
        let reply = serve(&mut p, &req);
        assert_eq!(
            decode(reply, ResultKind::Str).unwrap(),
            SyscallResult::Str(b"/tmp".to_vec())
        );
    }

    #[test]
    fn client_open_read_roundtrip() {
        let mut p = SandboxPolicy::new(files_root(), "/", vec![], false);
        let open = encode_request(
            "ll_os.ll_os_open",
            &[
                MarshalValue::Str(b"/hi.txt".to_vec()),
                MarshalValue::Int(libc::O_RDONLY as i64),
                MarshalValue::Int(0o777),
            ],
        );
        let fd = match decode(serve(&mut p, &open), ResultKind::Int).unwrap() {
            SyscallResult::Int(fd) => fd,
            other => panic!("fd: {other:?}"),
        };
        let read = encode_request(
            "ll_os.ll_os_read",
            &[MarshalValue::Int(fd), MarshalValue::Int(100)],
        );
        assert_eq!(
            decode(serve(&mut p, &read), ResultKind::Str).unwrap(),
            SyscallResult::Str(b"Hello, world!\n".to_vec())
        );
    }

    #[test]
    fn client_decodes_oserror_with_errno() {
        // open an existing file for write -> controller raises OSError(EPERM);
        // the client must surface SandboxError::Os(EPERM).
        let mut p = SandboxPolicy::new(files_root(), "/", vec![], false);
        let req = encode_request(
            "ll_os.ll_os_open",
            &[
                MarshalValue::Str(b"/hi.txt".to_vec()),
                MarshalValue::Int(libc::O_WRONLY as i64),
                MarshalValue::Int(0o666),
            ],
        );
        let err = decode(serve(&mut p, &req), ResultKind::Int).unwrap_err();
        assert_eq!(err, SandboxError::Os(libc::EPERM));
    }

    #[test]
    fn client_decodes_statresult() {
        let mut p = SandboxPolicy::new(files_root(), "/", vec![], false);
        let req = encode_request(
            "ll_os.ll_os_stat",
            &[MarshalValue::Str(b"/hi.txt".to_vec())],
        );
        match decode(serve(&mut p, &req), ResultKind::StatResult).unwrap() {
            SyscallResult::Stat(st) => assert_eq!(st.st_size, 14),
            other => panic!("stat: {other:?}"),
        }
    }
}
