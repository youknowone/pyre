//! The OS-call seam — the single indirection through which the interpreter
//! reaches the operating system, so a sandbox build can swap real syscalls for
//! marshalling trampolines at compile time.
//!
//! This is the faithful Rust analog of RPython's `--sandbox` translation
//! (`rpython/translator/sandbox/rsandbox.py: make_sandbox_trampoline`): there,
//! the translator replaces each external C call with a trampoline; here,
//! `#[cfg(feature = "sandbox")]` selects [`TrampolineHost`] (which marshals to
//! the controller via `pyre_sandbox::client`) instead of [`RealHost`] (the real
//! libc/std bodies). Because pyre's builtins are non-capturing `fn` pointers
//! (`gateway.rs` `BuiltinCodeFn = fn(...)`), the seam cannot be a runtime-held
//! object; it is reached purely by module path through the free functions in
//! [`ops`], and the real-vs-trampoline choice is baked in by cfg.
//!
//! This seam is a *selective* compile-out, not RPython's whole-program one:
//! genc rewrites every external in an exhaustive translation pass, whereas here
//! the guarantee reaches only code that names `libc`/fenced `std` through the
//! seam and the CI fence. A host call outside that surface (a dependency's own
//! FFI, a raw `syscall!`, an un-rerouted site) still compiles; only the runtime
//! `seccomp` backstop (Linux) catches it. See `pyre_sandbox`'s crate-root
//! "Structural constraint" note for the full guarantee model.

// Used only by RealHost's byte<->OsStr conversions (the non-sandbox build).
#[cfg(not(feature = "sandbox"))]
use std::os::unix::ffi::{OsStrExt, OsStringExt};

#[cfg(feature = "sandbox")]
use pyre_sandbox::client::{self, SyscallResult};
#[cfg(feature = "sandbox")]
use pyre_sandbox::protocol::SandboxError;
#[cfg(feature = "sandbox")]
use pyre_sandbox::rmarshal::MarshalValue;

/// `host_seam::sys` — the libc surface a sandbox-compiled module is allowed to
/// name. Off sandbox it is `libc` verbatim; under sandbox it re-exports only
/// TYPES, CONSTANTS, and the curated *pure* (no syscall, no host I/O) functions,
/// never a syscall function. A module that does `use crate::host_seam::sys as
/// libc;` therefore turns any direct syscall *call* outside this seam into a
/// compile error — the fails-closed analog of RPython leaving unsupported
/// externals unlinkable. Add an entry when a sandbox-reachable module needs it;
/// a missing one is a loud compile error, fails-closed in the safe direction.
#[cfg(not(feature = "sandbox"))]
pub use ::libc as sys;

#[cfg(feature = "sandbox")]
pub mod sys {
    // Types (zero-cost to name; cross no boundary).
    pub use ::libc::{
        c_char, c_int, c_long, c_uint, c_void, clockid_t, gid_t, mode_t, off_t, pid_t, rusage,
        size_t, time_t, timespec, timeval, tm, uid_t,
    };
    // The struct `sched_setparam`/`sched_setscheduler` fill in before handing it
    // to host_env. Naming it allocates nothing and calls nothing; the calls
    // themselves are overwritten by the sandbox stubs in interp_posix.
    #[cfg(any(
        target_os = "android",
        target_os = "freebsd",
        target_os = "linux",
        target_os = "netbsd"
    ))]
    pub use ::libc::sched_param;
    // Calendar/formatting on a caller-supplied value: `gmtime_r` reads only glibc's
    // timezone cache — which the seccomp backstop primes before lockdown
    // (see `pyre_sandbox::seccomp`), so at runtime it doesn't open a host file.
    pub use ::libc::gmtime_r;
    // Pure functions: wait-status decoders are bit-twiddling on a caller-supplied
    // integer; they make no syscall and read no host state.
    pub use ::libc::{WEXITSTATUS, WIFEXITED, WIFSIGNALED, WIFSTOPPED, WSTOPSIG, WTERMSIG};
    // Type-only re-exports for names that libc also defines as a function, so
    // the type resolves but the syscall call does not.
    #[allow(non_camel_case_types)]
    pub type stat = ::libc::stat;
    pub use ::libc::winsize;
    // Constants (added as sandbox-reachable modules need them).
    pub use ::libc::{
        AT_FDCWD, CODESET, EBADF, EINTR, EINVAL, F_OK, LC_ALL, LC_COLLATE, LC_CTYPE, LC_MESSAGES,
        LC_MONETARY, LC_NUMERIC, LC_TIME, O_ACCMODE, O_APPEND, O_ASYNC, O_CLOEXEC, O_CREAT,
        O_DIRECTORY, O_DSYNC, O_EXCL, O_FSYNC, O_NOCTTY, O_NOFOLLOW, O_NONBLOCK, O_RDONLY, O_RDWR,
        O_SYNC, O_TRUNC, O_WRONLY, PRIO_PGRP, PRIO_PROCESS, PRIO_USER, R_OK, RTLD_GLOBAL,
        RTLD_LAZY, RTLD_LOCAL, RTLD_NODELETE, RTLD_NOLOAD, RTLD_NOW, RUSAGE_SELF, S_IFDIR, S_IFMT,
        S_IFREG, SEEK_CUR, SEEK_END, SEEK_SET, ST_NOSUID, ST_RDONLY, TIOCGWINSZ, W_OK, WCONTINUED,
        WEXITED, WNOHANG, WNOWAIT, WSTOPPED, WUNTRACED, X_OK,
    };
    // The `<fcntl.h>` flags only one platform declares, carrying the same gates
    // as the table that publishes them.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    pub use ::libc::{O_DIRECT, O_LARGEFILE, O_NOATIME, O_PATH, O_RSYNC, O_TMPFILE};
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub use ::libc::{O_EVTONLY, O_EXEC, O_EXLOCK, O_NOFOLLOW_ANY, O_SEARCH, O_SHLOCK, O_SYMLINK};
    // `lockf`'s commands and `waitid`'s two vocabularies — which process it is
    // asked about, and what it reports happened. All are numbers the module
    // publishes; the calls themselves are refused under sandbox.
    pub use ::libc::{
        CLD_CONTINUED, CLD_DUMPED, CLD_EXITED, CLD_KILLED, CLD_STOPPED, CLD_TRAPPED, F_LOCK,
        F_TEST, F_TLOCK, F_ULOCK, P_ALL, P_PGID, P_PID,
    };
    // `posix`'s `<sched.h>` and `<dlfcn.h>` names, which carry the same gates
    // there as the table that publishes them: the Apple targets declare the
    // scheduling policies outside the `libc` crate, `SCHED_BATCH`/`SCHED_IDLE`
    // are Linux's own, and `RTLD_DEEPBIND` is glibc's.
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    pub use ::libc::RTLD_DEEPBIND;
    #[cfg(any(target_os = "linux", target_os = "android"))]
    pub use ::libc::{SCHED_BATCH, SCHED_IDLE};
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    pub use ::libc::{SCHED_FIFO, SCHED_OTHER, SCHED_RR};
    // `SEEK_HOLE`/`SEEK_DATA` carry the same host list as the table that
    // publishes them: they exist only where the header answers the question.
    #[cfg(any(
        target_os = "macos",
        target_os = "ios",
        target_os = "linux",
        target_os = "android",
        target_os = "freebsd",
        target_os = "dragonfly",
        target_os = "solaris",
        target_os = "illumos",
        target_os = "hurd",
    ))]
    pub use ::libc::{SEEK_DATA, SEEK_HOLE};
    // `posix.sysconf`'s name table names one constant per entry and calls
    // nothing, so the whole table resolves through the seam.
    #[cfg(all(unix, not(target_os = "redox")))]
    pub use ::libc::{
        _SC_2_C_BIND, _SC_2_C_DEV, _SC_2_CHAR_TERM, _SC_2_FORT_DEV, _SC_2_FORT_RUN,
        _SC_2_LOCALEDEF, _SC_2_SW_DEV, _SC_2_UPE, _SC_2_VERSION, _SC_AIO_LISTIO_MAX, _SC_AIO_MAX,
        _SC_AIO_PRIO_DELTA_MAX, _SC_ARG_MAX, _SC_ASYNCHRONOUS_IO, _SC_ATEXIT_MAX, _SC_BC_BASE_MAX,
        _SC_BC_DIM_MAX, _SC_BC_SCALE_MAX, _SC_BC_STRING_MAX, _SC_CHILD_MAX, _SC_CLK_TCK,
        _SC_COLL_WEIGHTS_MAX, _SC_DELAYTIMER_MAX, _SC_EXPR_NEST_MAX, _SC_FSYNC,
        _SC_GETGR_R_SIZE_MAX, _SC_GETPW_R_SIZE_MAX, _SC_IOV_MAX, _SC_JOB_CONTROL, _SC_LINE_MAX,
        _SC_LOGIN_NAME_MAX, _SC_MAPPED_FILES, _SC_MEMLOCK, _SC_MEMLOCK_RANGE,
        _SC_MEMORY_PROTECTION, _SC_MESSAGE_PASSING, _SC_MQ_OPEN_MAX, _SC_MQ_PRIO_MAX,
        _SC_NGROUPS_MAX, _SC_NPROCESSORS_CONF, _SC_NPROCESSORS_ONLN, _SC_OPEN_MAX, _SC_PAGE_SIZE,
        _SC_PASS_MAX, _SC_PHYS_PAGES, _SC_PRIORITIZED_IO, _SC_PRIORITY_SCHEDULING, _SC_RE_DUP_MAX,
        _SC_REALTIME_SIGNALS, _SC_RTSIG_MAX, _SC_SAVED_IDS, _SC_SEM_NSEMS_MAX, _SC_SEM_VALUE_MAX,
        _SC_SEMAPHORES, _SC_SHARED_MEMORY_OBJECTS, _SC_SIGQUEUE_MAX, _SC_STREAM_MAX,
        _SC_SYNCHRONIZED_IO, _SC_THREAD_ATTR_STACKADDR, _SC_THREAD_ATTR_STACKSIZE,
        _SC_THREAD_DESTRUCTOR_ITERATIONS, _SC_THREAD_KEYS_MAX, _SC_THREAD_PRIO_INHERIT,
        _SC_THREAD_PRIO_PROTECT, _SC_THREAD_PRIORITY_SCHEDULING, _SC_THREAD_PROCESS_SHARED,
        _SC_THREAD_SAFE_FUNCTIONS, _SC_THREAD_STACK_MIN, _SC_THREAD_THREADS_MAX, _SC_THREADS,
        _SC_TIMER_MAX, _SC_TIMERS, _SC_TTY_NAME_MAX, _SC_TZNAME_MAX, _SC_VERSION,
        _SC_XBS5_ILP32_OFF32, _SC_XBS5_ILP32_OFFBIG, _SC_XBS5_LP64_OFF64, _SC_XBS5_LPBIG_OFFBIG,
        _SC_XOPEN_CRYPT, _SC_XOPEN_ENH_I18N, _SC_XOPEN_LEGACY, _SC_XOPEN_REALTIME,
        _SC_XOPEN_REALTIME_THREADS, _SC_XOPEN_SHM, _SC_XOPEN_UNIX, _SC_XOPEN_VERSION,
        _SC_XOPEN_XCU_VERSION,
    };
    // `posix.pathconf_names`' `_PC_*` table. `libc` exports these on the BSD
    // family only, so the table names them there and spells the glibc values
    // out for Linux — this re-export carries the same gate.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub use ::libc::{
        _PC_ALLOC_SIZE_MIN, _PC_ASYNC_IO, _PC_CHOWN_RESTRICTED, _PC_FILESIZEBITS, _PC_LINK_MAX,
        _PC_MAX_CANON, _PC_MAX_INPUT, _PC_MIN_HOLE_SIZE, _PC_NAME_MAX, _PC_NO_TRUNC, _PC_PATH_MAX,
        _PC_PIPE_BUF, _PC_PRIO_IO, _PC_REC_INCR_XFER_SIZE, _PC_REC_MAX_XFER_SIZE,
        _PC_REC_MIN_XFER_SIZE, _PC_REC_XFER_ALIGN, _PC_SYMLINK_MAX, _PC_SYNC_IO, _PC_VDISABLE,
    };
}

/// An error from an OS seam operation. Self-contained in the interpreter so the
/// non-sandbox build does not depend on `pyre-sandbox`. Numeric codes mirror the
/// sandbox `EXCEPTION_TABLE`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SeamError {
    /// `OSError(errno)`.
    Os(i32),
    Io,
    Overflow,
    Value,
    ZeroDivision,
    Memory,
    Key,
    Index,
    Runtime,
}

#[cfg(feature = "sandbox")]
impl From<SandboxError> for SeamError {
    fn from(e: SandboxError) -> Self {
        match e {
            SandboxError::Os(errno) => SeamError::Os(errno),
            SandboxError::Io => SeamError::Io,
            SandboxError::Overflow => SeamError::Overflow,
            SandboxError::Value => SeamError::Value,
            SandboxError::ZeroDivision => SeamError::ZeroDivision,
            SandboxError::Memory => SeamError::Memory,
            SandboxError::Key => SeamError::Key,
            SandboxError::Index => SeamError::Index,
            SandboxError::Runtime | SandboxError::Protocol(_) => SeamError::Runtime,
        }
    }
}

pub type SeamResult<T> = Result<T, SeamError>;

/// Map a [`SeamError`] onto the interpreter's `PyError`, mirroring the
/// `io_err`/`fd_io_err` conversions the real-syscall call sites use. `context`
/// is the path that becomes the `OSError`'s `filename` (empty = omit, matching
/// the `""` the fd call sites pass). Both real and trampoline-backed opens
/// use this conversion so they expose the same Python exception shape.
pub fn seam_os_err(e: SeamError, context: &str) -> crate::PyError {
    let w_filename = if context.is_empty() {
        pyre_object::PY_NULL
    } else {
        pyre_object::w_str_new(context)
    };
    seam_os_err_with_filename(e, w_filename)
}

/// `wrap_oserror2(space, e, w_path)` in `interp_posix.py`: retain the exact
/// argument object for `OSError.filename` instead of rebuilding its spelling.
pub fn seam_os_err_with_filename(
    e: SeamError,
    w_filename: pyre_object::PyObjectRef,
) -> crate::PyError {
    match e {
        SeamError::Os(errno) => crate::PyError::os_error_syscall(errno, w_filename),
        SeamError::Io => crate::PyError::os_error_with_errno(libc::EIO, "I/O error"),
        SeamError::Value => crate::PyError::value_error("embedded null in path"),
        SeamError::Overflow => crate::PyError::overflow_error("integer overflow"),
        SeamError::ZeroDivision => crate::PyError::runtime_error("division by zero"),
        SeamError::Memory => crate::PyError::memory_error("out of memory"),
        SeamError::Key => crate::PyError::key_error("key error"),
        SeamError::Index => crate::PyError::index_error("index out of range"),
        SeamError::Runtime => crate::PyError::runtime_error("sandbox runtime error"),
    }
}

/// The not-implemented stub for OS surface that the sandbox controller does not
/// service (signal/socket/dup/ftruncate/…). Port of `rsandbox.py`'s
/// `get_sandbox_stub`/`not_implemented_stub`: raise `RuntimeError` rather than
/// touch the OS (`not_implemented_stub` does `raise RuntimeError(msg)`).
#[cfg(feature = "sandbox")]
pub fn stub(fnname: &str) -> crate::PyError {
    crate::PyError::runtime_error(format!("{fnname} is not available in the sandbox"))
}

/// The raw stat fields `make_stat_result` consumes. `RealHost` fills every field
/// from a `libc::stat`; under sandbox the wire `os.stat_result` carries only the
/// 10 protocol fields (mode/ino/dev/nlink/uid/gid/size + integer atime/mtime/
/// ctime) and the remaining fields are zeroed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StatBuf {
    pub mode: u32,
    pub ino: u64,
    pub dev: u64,
    pub nlink: u64,
    pub uid: u32,
    pub gid: u32,
    pub size: u64,
    pub atime: i64,
    pub mtime: i64,
    pub ctime: i64,
    pub atime_nsec: i64,
    pub mtime_nsec: i64,
    pub ctime_nsec: i64,
    pub blksize: u64,
    pub blocks: u64,
    pub rdev: u64,
    pub st_flags: u32,
}

/// The real-syscall host (selected when the `sandbox` feature is off).
pub struct RealHost;

/// The marshalling-trampoline host (selected when the `sandbox` feature is on);
/// every method round-trips through `pyre_sandbox::client::syscall`.
#[cfg(feature = "sandbox")]
pub struct TrampolineHost;

// ── arg/return token -> Rust type ────────────────────────────────────────────

macro_rules! seam_arg_ty {
    (bytes) => {
        &[u8]
    };
    (i32) => {
        i32
    };
    (u32) => {
        u32
    };
    (i64) => {
        i64
    };
    (f64) => {
        f64
    };
}

macro_rules! seam_ret_ty {
    (unit) => { () };
    (i32) => { i32 };
    (i64) => { i64 };
    (longlong) => { i64 };
    (bytes) => { Vec<u8> };
    (bool) => { bool };
    (optbytes) => { Option<Vec<u8>> };
    (liststr) => { Vec<Vec<u8>> };
    (envitems) => { Vec<(Vec<u8>, Vec<u8>)> };
    (stat) => { StatBuf };
    (f64) => { f64 };
}

// The protocol ResultKind a return token decodes as. `longlong` forces 'I'+8
// (RESULTTYPE_LONGLONG, e.g. lseek); plain `i64` accepts 'i'/'I'.
#[cfg(feature = "sandbox")]
macro_rules! seam_result_kind {
    (unit) => {
        pyre_sandbox::protocol::ResultKind::None
    };
    (i32) => {
        pyre_sandbox::protocol::ResultKind::Int
    };
    (i64) => {
        pyre_sandbox::protocol::ResultKind::Int
    };
    (longlong) => {
        pyre_sandbox::protocol::ResultKind::LongLong
    };
    (bytes) => {
        pyre_sandbox::protocol::ResultKind::Str
    };
    (bool) => {
        pyre_sandbox::protocol::ResultKind::Bool
    };
    (optbytes) => {
        pyre_sandbox::protocol::ResultKind::OptStr
    };
    (liststr) => {
        pyre_sandbox::protocol::ResultKind::ListStr
    };
    (envitems) => {
        pyre_sandbox::protocol::ResultKind::EnvItems
    };
    (stat) => {
        pyre_sandbox::protocol::ResultKind::StatResult
    };
    (f64) => {
        pyre_sandbox::protocol::ResultKind::Float
    };
}

// Marshal one argument into a request MarshalValue (client int convention).
#[cfg(feature = "sandbox")]
macro_rules! seam_marshal {
    (bytes, $v:expr) => {
        MarshalValue::Str($v.to_vec())
    };
    (i32, $v:expr) => {
        MarshalValue::Int($v as i64)
    };
    (u32, $v:expr) => {
        MarshalValue::Int($v as i64)
    };
    (i64, $v:expr) => {
        MarshalValue::Int($v)
    };
    (f64, $v:expr) => {
        MarshalValue::Float($v)
    };
}

// Project a decoded SyscallResult back to the typed return value.
#[cfg(feature = "sandbox")]
macro_rules! seam_unwrap {
    (unit, $r:expr) => {
        match $r {
            SyscallResult::None => Ok(()),
            _ => Err(SeamError::Runtime),
        }
    };
    (i32, $r:expr) => {
        match $r {
            SyscallResult::Int(v) => Ok(v as i32),
            _ => Err(SeamError::Runtime),
        }
    };
    (i64, $r:expr) => {
        match $r {
            SyscallResult::Int(v) => Ok(v),
            _ => Err(SeamError::Runtime),
        }
    };
    (longlong, $r:expr) => {
        match $r {
            SyscallResult::Int(v) => Ok(v),
            _ => Err(SeamError::Runtime),
        }
    };
    (bytes, $r:expr) => {
        match $r {
            SyscallResult::Str(v) => Ok(v),
            _ => Err(SeamError::Runtime),
        }
    };
    (bool, $r:expr) => {
        match $r {
            SyscallResult::Bool(v) => Ok(v),
            _ => Err(SeamError::Runtime),
        }
    };
    (optbytes, $r:expr) => {
        match $r {
            SyscallResult::OptStr(v) => Ok(v),
            _ => Err(SeamError::Runtime),
        }
    };
    (liststr, $r:expr) => {
        match $r {
            SyscallResult::ListStr(v) => Ok(v),
            _ => Err(SeamError::Runtime),
        }
    };
    (envitems, $r:expr) => {
        match $r {
            SyscallResult::EnvItems(v) => Ok(v),
            _ => Err(SeamError::Runtime),
        }
    };
    (stat, $r:expr) => {
        match $r {
            SyscallResult::Stat(s) => Ok(StatBuf::from_wire(&s)),
            _ => Err(SeamError::Runtime),
        }
    };
    (f64, $r:expr) => {
        match $r {
            SyscallResult::Float(v) => Ok(v),
            _ => Err(SeamError::Runtime),
        }
    };
}

/// Declare the whole OS surface once: it generates the [`SandboxableHost`] trait,
/// the cfg-gated [`TrampolineHost`] impl (each body marshalling to the
/// controller), and the [`ops`] free-function forwarders. `RealHost` implements
/// the trait by hand below.
macro_rules! declare_seam {
    ($(
        $name:ident ( $($a:ident : $aty:tt),* ) -> $rty:tt = $ll:literal ;
    )*) => {
        /// The OS surface the sandbox protocol mediates. Implemented by
        /// [`RealHost`] (real syscalls) and [`TrampolineHost`] (marshalling).
        pub trait SandboxableHost {
            $(
                fn $name($($a : seam_arg_ty!($aty)),*) -> SeamResult<seam_ret_ty!($rty)>;
            )*
        }

        #[cfg(feature = "sandbox")]
        impl SandboxableHost for TrampolineHost {
            $(
                fn $name($($a : seam_arg_ty!($aty)),*) -> SeamResult<seam_ret_ty!($rty)> {
                    let args = [ $( seam_marshal!($aty, $a) ),* ];
                    // Every trampolined op is a round trip to the controller
                    // and blocks for its reply.
                    let result = {
                        let _blocked = crate::module::thread::before_external_block();
                        client::syscall($ll, &args, seam_result_kind!($rty))
                    }?;
                    seam_unwrap!($rty, result)
                }
            )*
        }

        /// The free-function seam. Every OS call site reaches the host by naming
        /// `host_seam::ops::*`; the real-vs-trampoline body is chosen by cfg.
        pub mod ops {
            use super::*;

            #[cfg(not(feature = "sandbox"))]
            type Host = RealHost;
            #[cfg(feature = "sandbox")]
            type Host = TrampolineHost;

            $(
                pub fn $name($($a : seam_arg_ty!($aty)),*) -> SeamResult<seam_ret_ty!($rty)> {
                    <Host as SandboxableHost>::$name($($a),*)
                }
            )*
        }
    };
}

declare_seam! {
    open(path: bytes, flags: i32, mode: u32) -> i32 = "ll_os.ll_os_open";
    close(fd: i32) -> unit = "ll_os.ll_os_close";
    read(fd: i32, size: i64) -> bytes = "ll_os.ll_os_read";
    write(fd: i32, data: bytes) -> i64 = "ll_os.ll_os_write";
    lseek(fd: i32, pos: i64, how: i32) -> longlong = "ll_os.ll_os_lseek";
    stat(path: bytes) -> stat = "ll_os.ll_os_stat";
    lstat(path: bytes) -> stat = "ll_os.ll_os_lstat";
    fstat(fd: i32) -> stat = "ll_os.ll_os_fstat";
    access(path: bytes, mode: i32) -> bool = "ll_os.ll_os_access";
    isatty(fd: i32) -> bool = "ll_os.ll_os_isatty";
    getcwd() -> bytes = "ll_os.ll_os_getcwd";
    listdir(path: bytes) -> liststr = "ll_os.ll_os_listdir";
    getenv(name: bytes) -> optbytes = "ll_os.ll_os_getenv";
    envitems() -> envitems = "ll_os.ll_os_envitems";
    strerror(code: i32) -> bytes = "ll_os.ll_os_strerror";
    getuid() -> i64 = "ll_os.ll_os_getuid";
    geteuid() -> i64 = "ll_os.ll_os_geteuid";
    getgid() -> i64 = "ll_os.ll_os_getgid";
    getegid() -> i64 = "ll_os.ll_os_getegid";
    unlink(path: bytes) -> unit = "ll_os.ll_os_unlink";
    mkdir(path: bytes, mode: u32) -> unit = "ll_os.ll_os_mkdir";
    urandom(size: i64) -> bytes = "ll_os.ll_os_urandom";
    time() -> f64 = "ll_time.ll_time_time";
    clock() -> f64 = "ll_time.ll_time_clock";
    sleep(seconds: f64) -> unit = "ll_time.ll_time_sleep";
}

/// Read one process-environment value through the host seam.
///
/// Keep this target-independent entry point at the module root: Charon's
/// wasm-layout extraction does not retain the macro-generated `ops` module,
/// while ordinary Unix builds still delegate to the same real/sandbox host
/// selected by [`ops::getenv`].
pub fn getenv(name: &[u8]) -> SeamResult<Option<Vec<u8>>> {
    ops::getenv(name)
}

// ── Interpreter stdio ────────────────────────────────────────────────────────
//
// Diagnostic output (tracebacks, warnings, the interactive displayhook) reaches
// fd 1/2 through these two helpers so it obeys the same seam as `sys.stdout`.
// Under sandbox fd 1 is the marshalling pipe, so a raw write would corrupt the
// protocol: route through `ll_os_write(1|2,…)` and let the controller relay it.
// Best-effort — a failed relay is dropped, matching a closed real stream.

/// Emit bytes to the interpreter's stdout (fd 1).
///
/// `dont_look_inside` for the same reason `ll_os_write` is an external: the
/// non-sandbox arm reaches `std::io::stdout()`, a host handle with no lifted
/// form, so a caller that can reach it loses its own jitcode and so does every
/// caller above it.
#[majit_macros::dont_look_inside]
pub fn emit_stdout(bytes: &[u8]) {
    if crate::print_hook_emit_bytes(bytes) {
        return;
    }
    #[cfg(not(feature = "sandbox"))]
    {
        use std::io::Write;
        let _ = std::io::stdout().write_all(bytes);
    }
    #[cfg(feature = "sandbox")]
    {
        let _ = ops::write(1, bytes);
    }
    flush_stdout_when_unbuffered();
}

/// Settle fd 1 when `-u` / PYTHONUNBUFFERED asked for no buffering.
///
/// `std::io::Stdout` is a `LineWriter`, so without this a value carrying no
/// newline waits in its buffer until process exit — which is the opposite of
/// what the flag requests, and differs from the block buffering an unflagged
/// run should get. Under sandbox fd 1 is already written through unbuffered
/// `ll_os_write` requests, so there is nothing to settle.
pub fn flush_stdout_when_unbuffered() {
    #[cfg(not(feature = "sandbox"))]
    if crate::importing::unbuffered_flag() {
        use std::io::Write;
        let _ = std::io::stdout().flush();
    }
}

/// Emit bytes to the interpreter's stderr (fd 2).
///
/// `dont_look_inside` — see [`emit_stdout`].  This one is on the warning path:
/// `warn::warn` is the pre-`_warnings` fallback reachable from
/// `warn::warn_category_w`, so an opaque body here is what keeps
/// `warn_category_w` liftable.
#[majit_macros::dont_look_inside]
pub fn emit_stderr(bytes: &[u8]) {
    if crate::stderr_hook_emit(bytes) {
        return;
    }
    #[cfg(not(feature = "sandbox"))]
    {
        use std::io::Write;
        let _ = std::io::stderr().write_all(bytes);
    }
    #[cfg(feature = "sandbox")]
    {
        let _ = ops::write(2, bytes);
    }
}

/// Flush the interpreter's stdout (fd 1). Under sandbox fd 1 is written through
/// unbuffered `ll_os_write` syscalls, so there is nothing to flush.
pub fn flush_stdout() {
    #[cfg(not(feature = "sandbox"))]
    {
        use std::io::Write;
        let _ = std::io::stdout().flush();
    }
}

// ── StatBuf constructors ─────────────────────────────────────────────────────

impl StatBuf {
    /// Build from a `libc::stat` (the real-host path). `st_flags` exists only on
    /// the BSD-derived platforms; elsewhere it is 0.
    #[cfg(not(feature = "sandbox"))]
    fn from_libc(st: &libc::stat) -> Self {
        #[cfg(any(target_os = "macos", target_os = "ios", target_os = "freebsd"))]
        let st_flags = st.st_flags as u32;
        #[cfg(not(any(target_os = "macos", target_os = "ios", target_os = "freebsd")))]
        let st_flags = 0u32;
        StatBuf {
            mode: st.st_mode as u32,
            ino: st.st_ino as u64,
            dev: st.st_dev as u64,
            nlink: st.st_nlink as u64,
            uid: st.st_uid as u32,
            gid: st.st_gid as u32,
            size: st.st_size as u64,
            atime: st.st_atime as i64,
            mtime: st.st_mtime as i64,
            ctime: st.st_ctime as i64,
            atime_nsec: st.st_atime_nsec as i64,
            mtime_nsec: st.st_mtime_nsec as i64,
            ctime_nsec: st.st_ctime_nsec as i64,
            blksize: st.st_blksize as u64,
            blocks: st.st_blocks as u64,
            rdev: st.st_rdev as u64,
            st_flags,
        }
    }

    /// Build from the 10-field wire `os.stat_result` (the trampoline path); the
    /// non-protocol fields (nsec, blksize, blocks, rdev, st_flags) are zero.
    #[cfg(feature = "sandbox")]
    fn from_wire(st: &pyre_sandbox::vfs::StatResult) -> Self {
        StatBuf {
            mode: st.st_mode,
            ino: st.st_ino,
            dev: st.st_dev,
            nlink: st.st_nlink,
            uid: st.st_uid,
            gid: st.st_gid,
            size: st.st_size,
            atime: st.st_atime,
            mtime: st.st_mtime,
            ctime: st.st_ctime,
            ..StatBuf::default()
        }
    }
}

// ── RealHost: the real-syscall bodies (non-sandbox build) ─────────────────────

#[cfg(not(feature = "sandbox"))]
mod real {
    use super::*;
    use std::ffi::{CStr, CString};
    use std::os::raw::c_void;

    fn last_os_error() -> SeamError {
        SeamError::Os(
            std::io::Error::last_os_error()
                .raw_os_error()
                .unwrap_or(libc::EIO),
        )
    }

    fn cstr(path: &[u8]) -> SeamResult<CString> {
        CString::new(path).map_err(|_| SeamError::Value)
    }

    /// `rffi.py:193-211 call_external_function` for the real-host bodies: an OS
    /// call blocks, so drop the GIL around it.  `last_os_error` is read inside
    /// `f`, which keeps it ahead of the re-acquire the way `_errno_after`
    /// precedes `rgil.acquire()`.
    #[inline]
    fn blocking<R>(f: impl FnOnce() -> R) -> R {
        let _blocked = crate::module::thread::before_external_block();
        f()
    }

    fn real_stat(path: &[u8], symlink: bool) -> SeamResult<StatBuf> {
        let c = cstr(path)?;
        blocking(|| {
            // SAFETY: stat(2)/lstat(2) into a zeroed, owned `libc::stat`.
            let mut st: libc::stat = unsafe { std::mem::zeroed() };
            let r = unsafe {
                if symlink {
                    libc::lstat(c.as_ptr(), &mut st)
                } else {
                    libc::stat(c.as_ptr(), &mut st)
                }
            };
            if r < 0 {
                return Err(last_os_error());
            }
            Ok(StatBuf::from_libc(&st))
        })
    }

    impl SandboxableHost for RealHost {
        fn open(path: &[u8], flags: i32, mode: u32) -> SeamResult<i32> {
            let c = cstr(path)?;
            blocking(|| {
                // SAFETY: open(2) with an owned NUL-terminated path.
                let fd = unsafe { libc::open(c.as_ptr(), flags, mode as libc::c_uint) };
                if fd < 0 { Err(last_os_error()) } else { Ok(fd) }
            })
        }

        fn close(fd: i32) -> SeamResult<()> {
            blocking(|| {
                if unsafe { libc::close(fd) } < 0 {
                    Err(last_os_error())
                } else {
                    Ok(())
                }
            })
        }

        fn read(fd: i32, size: i64) -> SeamResult<Vec<u8>> {
            let n = size.max(0) as usize;
            let mut buf = vec![0u8; n];
            // SAFETY: read(2) into a buffer we own and sized to `n`.
            let (got, err) = blocking(|| {
                let got = unsafe { libc::read(fd, buf.as_mut_ptr() as *mut c_void, n) };
                // Read `errno` before the guard hands the GIL back: the
                // re-acquire can enter the stealer loop, whose mutex and
                // condvar waits overwrite it.
                (got, last_os_error())
            });
            if got < 0 {
                return Err(err);
            }
            buf.truncate(got as usize);
            Ok(buf)
        }

        fn urandom(size: i64) -> SeamResult<Vec<u8>> {
            use std::io::Read;
            let n = size.max(0) as usize;
            let mut buf = vec![0u8; n];
            blocking(|| {
                std::fs::File::open("/dev/urandom")
                    .and_then(|mut f| f.read_exact(&mut buf))
                    .map_err(|_| last_os_error())
            })?;
            Ok(buf)
        }

        fn write(fd: i32, data: &[u8]) -> SeamResult<i64> {
            blocking(|| {
                // SAFETY: write(2) from a slice held for the call.
                let n = unsafe { libc::write(fd, data.as_ptr() as *const c_void, data.len()) };
                if n < 0 {
                    Err(last_os_error())
                } else {
                    Ok(n as i64)
                }
            })
        }

        fn lseek(fd: i32, pos: i64, how: i32) -> SeamResult<i64> {
            let r = unsafe { libc::lseek(fd, pos as libc::off_t, how) };
            if r < 0 {
                Err(last_os_error())
            } else {
                Ok(r as i64)
            }
        }

        fn stat(path: &[u8]) -> SeamResult<StatBuf> {
            real_stat(path, false)
        }

        fn lstat(path: &[u8]) -> SeamResult<StatBuf> {
            real_stat(path, true)
        }

        fn fstat(fd: i32) -> SeamResult<StatBuf> {
            blocking(|| {
                // SAFETY: fstat(2) into a zeroed, owned `libc::stat`.
                let mut st: libc::stat = unsafe { std::mem::zeroed() };
                if unsafe { libc::fstat(fd, &mut st) } < 0 {
                    return Err(last_os_error());
                }
                Ok(StatBuf::from_libc(&st))
            })
        }

        fn access(path: &[u8], mode: i32) -> SeamResult<bool> {
            let c = cstr(path)?;
            Ok(blocking(|| unsafe { libc::access(c.as_ptr(), mode) }) == 0)
        }

        fn isatty(fd: i32) -> SeamResult<bool> {
            Ok(unsafe { libc::isatty(fd) } == 1)
        }

        fn getcwd() -> SeamResult<Vec<u8>> {
            blocking(|| {
                std::env::current_dir()
                    .map(|p| p.into_os_string().into_vec())
                    .map_err(|_| last_os_error())
            })
        }

        fn listdir(path: &[u8]) -> SeamResult<Vec<Vec<u8>>> {
            let p = std::path::Path::new(std::ffi::OsStr::from_bytes(path));
            blocking(|| {
                let mut names = Vec::new();
                for entry in std::fs::read_dir(p).map_err(|_| last_os_error())? {
                    let entry = entry.map_err(|_| last_os_error())?;
                    names.push(entry.file_name().into_vec());
                }
                Ok(names)
            })
        }

        fn getenv(name: &[u8]) -> SeamResult<Option<Vec<u8>>> {
            Ok(std::env::var_os(std::ffi::OsStr::from_bytes(name)).map(|v| v.into_vec()))
        }

        fn envitems() -> SeamResult<Vec<(Vec<u8>, Vec<u8>)>> {
            Ok(std::env::vars_os()
                .map(|(k, v)| (k.into_vec(), v.into_vec()))
                .collect())
        }

        fn strerror(code: i32) -> SeamResult<Vec<u8>> {
            // SAFETY: strerror returns a static string; copy it out immediately.
            let bytes = unsafe {
                let p = libc::strerror(code);
                if p.is_null() {
                    return Ok(format!("Unknown error {code}").into_bytes());
                }
                CStr::from_ptr(p).to_bytes().to_vec()
            };
            Ok(bytes)
        }

        fn getuid() -> SeamResult<i64> {
            Ok(unsafe { libc::getuid() } as i64)
        }

        fn geteuid() -> SeamResult<i64> {
            Ok(unsafe { libc::geteuid() } as i64)
        }

        fn getgid() -> SeamResult<i64> {
            Ok(unsafe { libc::getgid() } as i64)
        }

        fn getegid() -> SeamResult<i64> {
            Ok(unsafe { libc::getegid() } as i64)
        }

        fn unlink(path: &[u8]) -> SeamResult<()> {
            let c = cstr(path)?;
            blocking(|| {
                if unsafe { libc::unlink(c.as_ptr()) } < 0 {
                    Err(last_os_error())
                } else {
                    Ok(())
                }
            })
        }

        fn mkdir(path: &[u8], mode: u32) -> SeamResult<()> {
            let c = cstr(path)?;
            blocking(|| {
                if unsafe { libc::mkdir(c.as_ptr(), mode as libc::mode_t) } < 0 {
                    Err(last_os_error())
                } else {
                    Ok(())
                }
            })
        }

        fn time() -> SeamResult<f64> {
            Ok(std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0))
        }

        fn clock() -> SeamResult<f64> {
            // Process CPU time (user + system), the ll_time_clock analog.
            // SAFETY: getrusage into a zeroed, owned `libc::rusage`.
            let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
            if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) } != 0 {
                return Err(last_os_error());
            }
            let secs = |t: libc::timeval| t.tv_sec as f64 + t.tv_usec as f64 * 1e-6;
            Ok(secs(usage.ru_utime) + secs(usage.ru_stime))
        }

        fn sleep(seconds: f64) -> SeamResult<()> {
            if seconds > 0.0 {
                blocking(|| std::thread::sleep(std::time::Duration::from_secs_f64(seconds)));
            }
            Ok(())
        }
    }
}
