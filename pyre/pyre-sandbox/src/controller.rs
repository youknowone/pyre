//! The trusted parent process — port of `pypy/sandbox/pypy_interact.py`'s
//! `PyPySandboxedProc` plus the `SandboxedProc` spawn/interact/timeout machinery
//! from `rpython/translator/sandbox/sandlib.py`.
//!
//! `PyPySandboxedProc::interact` spawns the untrusted child with a cleared
//! environment and piped stdio, then runs [`SandboxPolicy::handle_until_return`]
//! against the real console, servicing every `ll_os.*`/`ll_time.*` request over a
//! virtual filesystem.

use std::io;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use indexmap::IndexMap;

use crate::sandlib::{Console, SandboxPolicy, TimeoutControl};
use crate::vfs::{Dir, FsNode, RealDir, RealFile};

// pypy_interact.py:39 `argv0 = '/bin/pypy3-c'`.
const ARGV0: &str = "/bin/pypy3-c";
// pypy_interact.py:40 `virtual_cwd = '/tmp'`.
const VIRTUAL_CWD: &str = "/tmp";

/// A monotonic "last activity" timestamp shared with the timeout watchdog. The
/// loop pings it after each serviced message; the watchdog kills the child if
/// too long elapses between pings (the `signal.alarm` reset in `sandlib.py`).
#[derive(Clone)]
struct ActivityClock(Arc<Mutex<Instant>>);

impl ActivityClock {
    fn new() -> Self {
        ActivityClock(Arc::new(Mutex::new(Instant::now())))
    }

    fn ping(&self) {
        *self.0.lock().expect("activity clock poisoned") = Instant::now();
    }

    fn since(&self) -> Duration {
        self.0.lock().expect("activity clock poisoned").elapsed()
    }
}

/// A watchdog thread that `SIGKILL`s the child if it goes quiet for `timeout`.
struct Watchdog {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl Watchdog {
    fn spawn(pid: u32, clock: ActivityClock, timeout: Duration, control: TimeoutControl) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let stop_thread = stop.clone();
        let handle = thread::spawn(move || {
            while !stop_thread.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_millis(200));
                if stop_thread.load(Ordering::Relaxed) {
                    break;
                }
                // While the child is blocked at the interactive prompt, keep the
                // activity clock fresh so idle time is not charged against the
                // timeout (sandlib.py enter_idle/leave_idle).
                if control.idle.load(Ordering::Relaxed) {
                    clock.ping();
                    continue;
                }
                if clock.since() >= timeout {
                    // SAFETY: a plain kill(2) on the child's pid.
                    unsafe {
                        libc::kill(pid as libc::pid_t, libc::SIGKILL);
                    }
                    // Unblock a long sleep being serviced so the controller
                    // stops promptly instead of parking for the full duration.
                    control.cancelled.store(true, Ordering::Relaxed);
                    break;
                }
            }
        });
        Watchdog {
            stop,
            handle: Some(handle),
        }
    }

    fn stop(mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// pypy_interact.py:43 `build_virtual_root`.
///
/// `lib_root`, when present, is mounted read-only at `/bin/lib` so the child can
/// import the standard library; an import-free script needs only the executable
/// and `/tmp`. PyPy hardcodes two mounts (`/bin/lib-python` + `/bin/lib_pypy`)
/// matching its own source tree; pyre instead mounts the single `--lib`
/// directory at `/bin/lib` and seeds `PYRE_STDLIB=/bin/lib` (see
/// `PyPySandboxedProc::new`), so the child's importer locates the stdlib through
/// the same env convention it uses untranslated rather than PyPy's fixed layout.
fn build_virtual_root(executable: &Path, tmpdir: Option<&Path>, lib_root: Option<&Path>) -> FsNode {
    // pypy_interact.py:47 `exclude = ['.pyc', '.pyo']`.
    let exclude = vec![".pyc".to_owned(), ".pyo".to_owned()];

    let tmpnode: FsNode = match tmpdir {
        Some(dir) => Rc::new(RealDir::new(dir, false, false, exclude.clone())),
        None => Rc::new(Dir::new(IndexMap::new())),
    };

    let mut bin: IndexMap<String, FsNode> = IndexMap::new();
    // pypy_interact.py:56 `RealFile(self.executable, mode=0o111)`.
    bin.insert(
        "pypy3-c".to_owned(),
        Rc::new(RealFile::new(executable, 0o111)),
    );
    if let Some(lib) = lib_root {
        bin.insert(
            "lib".to_owned(),
            Rc::new(RealDir::new(lib, false, false, exclude)),
        );
    }

    let mut root: IndexMap<String, FsNode> = IndexMap::new();
    root.insert("bin".to_owned(), Rc::new(Dir::new(bin)));
    root.insert("tmp".to_owned(), tmpnode);
    Rc::new(Dir::new(root))
}

/// The trusted controller around an untrusted pyre sandbox child.
pub struct PyPySandboxedProc {
    policy: SandboxPolicy,
    child: Child,
    timeout: Option<Duration>,
}

impl PyPySandboxedProc {
    /// pypy_interact.py:66 `__init__` + sandlib.py:`SandboxedProc.__init__`.
    ///
    /// Spawns `executable` (the real sandbox binary) with argv[0] forced to
    /// `/bin/pypy3-c`, a cleared environment, and piped stdin/stdout.
    pub fn new(
        executable: impl AsRef<Path>,
        arguments: &[String],
        tmpdir: Option<PathBuf>,
        lib_root: Option<PathBuf>,
        timeout: Option<Duration>,
        allow_net: bool,
        log_file: Option<PathBuf>,
    ) -> io::Result<Self> {
        let executable = std::fs::canonicalize(executable.as_ref())
            .unwrap_or_else(|_| executable.as_ref().to_path_buf());
        let virtual_root = build_virtual_root(&executable, tmpdir.as_deref(), lib_root.as_deref());
        // Expose the mounted stdlib to the child's importer: `--lib` mounts the
        // host directory read-only at `/bin/lib` (build_virtual_root), and the
        // child resolves PYRE_STDLIB through the env seam — but `env_clear()`
        // below wipes its real environment, so seed the value into the virtual
        // environment the controller answers `getenv` from.
        let virtual_env = match lib_root {
            Some(_) => vec![(b"PYRE_STDLIB".to_vec(), b"/bin/lib".to_vec())],
            None => Vec::new(),
        };
        // pypy_interact.py:41 `virtual_console_isatty = True`.
        let mut policy = SandboxPolicy::new(virtual_root, VIRTUAL_CWD, virtual_env, true);
        // VirtualizedSocketProc (sandlib.py:546): the operator opts into
        // `tcp://` mediation with `--allow-net`; the default policy is
        // network-closed.
        policy.set_allow_net(allow_net);
        // setlogfile (sandlib.py:334): `--log FILE` appends the guest's stdin
        // to FILE. Open eagerly so a bad path fails before the child starts.
        if let Some(path) = log_file {
            let file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)?;
            policy.set_input_log(file);
        }

        let mut command = Command::new(&executable);
        command
            .arg0(ARGV0)
            .args(arguments)
            .env_clear()
            .stdin(Stdio::piped())
            .stdout(Stdio::piped());
        // Close any inherited fds beyond stdio before exec, so a fd leaked from
        // the trusted controller cannot cross into the untrusted child (the
        // analog of subprocess `close_fds=True`). Runs post-fork/pre-exec and is
        // async-signal-safe (only raw close(2)/close_range(2)).
        // SAFETY: the hook calls only async-signal-safe syscalls and allocates
        // nothing.
        unsafe {
            command.pre_exec(close_inherited_fds);
        }
        let child = command.spawn()?;

        Ok(PyPySandboxedProc {
            policy,
            child,
            timeout,
        })
    }

    /// sandlib.py:`interact` — drive the request/reply loop against the real
    /// console until the child exits, returning its exit code.
    pub fn interact(&mut self) -> io::Result<i32> {
        let child_stdout = self
            .child
            .stdout
            .take()
            .expect("child stdout is piped at spawn");
        let mut child_stdin = self
            .child
            .stdin
            .take()
            .expect("child stdin is piped at spawn");

        let stdin = io::stdin();
        let stdout = io::stdout();
        let stderr = io::stderr();
        let mut input = stdin.lock();
        let mut output = stdout.lock();
        let mut error = stderr.lock();
        // SAFETY: isatty(2) on fd 0 — read-only query, no aliasing.
        let input_isatty = unsafe { libc::isatty(0) == 1 };
        let mut console = Console {
            input: &mut input,
            output: &mut output,
            error: &mut error,
            input_isatty,
        };

        let clock = ActivityClock::new();
        // Shared with the request handlers so an interactive read pauses the
        // timeout and a watchdog kill unblocks a long sleep.
        let control = TimeoutControl::default();
        self.policy.set_timeout_control(control.clone());
        let watchdog = self
            .timeout
            .map(|t| Watchdog::spawn(self.child.id(), clock.clone(), t, control.clone()));

        let result = {
            let mut ping = {
                let clock = clock.clone();
                move || clock.ping()
            };
            self.policy.handle_until_return_ticked(
                child_stdout,
                &mut child_stdin,
                &mut console,
                &mut ping,
            )
        };

        // Close the child's stdin so it observes EOF, then reap it. The watchdog
        // is stopped before the reap (inside reap_with_deadline) to close the
        // pid-reuse race; the post-EOF exit deadline is enforced there instead.
        drop(child_stdin);
        let status = self.reap_with_deadline(watchdog);
        result?;
        let status = status?;
        Ok(status.code().unwrap_or(-1))
    }

    /// Reap the child after stdin EOF, stopping the timeout watchdog first.
    ///
    /// Stopping the watchdog before the reap closes the pid-reuse race: once
    /// `wait()`/`try_wait()` reaps the child the OS may recycle its pid, and the
    /// watchdog holds only the bare pid, so a late `SIGKILL` could hit an
    /// unrelated process. A well-behaved child exits promptly once stdin closes;
    /// one that closed stdout but will not exit is bounded by `self.timeout`
    /// (the same budget the watchdog enforced per message) and then SIGKILLed
    /// via the owned `Child`, so it cannot hang the controller.
    fn reap_with_deadline(&mut self, watchdog: Option<Watchdog>) -> io::Result<ExitStatus> {
        if let Some(w) = watchdog {
            w.stop();
        }
        let Some(timeout) = self.timeout else {
            return self.child.wait();
        };
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(status) = self.child.try_wait()? {
                return Ok(status);
            }
            if Instant::now() >= deadline {
                let _ = self.child.kill();
                return self.child.wait();
            }
            thread::sleep(Duration::from_millis(20));
        }
    }
}

/// Post-fork/pre-exec hook: close every fd >= 3 so only stdio (0/1/2) crosses
/// into the untrusted child. Async-signal-safe — only raw close syscalls, no
/// allocation. A fd already closed is ignored.
fn close_inherited_fds() -> io::Result<()> {
    // Bounded brute-force close(2) loop; closing an unopened fd is a harmless
    // error. SAFETY: only raw sysconf/close syscalls, no allocation.
    unsafe fn close_from_brute_force() {
        // SAFETY: sysconf(_SC_OPEN_MAX) and close(2) over a bounded fd range;
        // closing an unopened fd is a harmless error.
        unsafe {
            let max = libc::sysconf(libc::_SC_OPEN_MAX);
            let max = if max < 0 { 1024 } else { max as i32 };
            for fd in 3..max {
                libc::close(fd);
            }
        }
    }
    #[cfg(target_os = "linux")]
    // SAFETY: close_range(2) over a fd range; harmless on already-closed fds.
    unsafe {
        let r = libc::syscall(
            libc::SYS_close_range,
            3 as libc::c_long,
            libc::c_uint::MAX as libc::c_long,
            0 as libc::c_long,
        );
        // Pre-5.9 kernels lack close_range and return ENOSYS; never leave an
        // inherited fd open — fall back to the brute-force loop, as
        // _posixsubprocess reverts to brute force whenever the primary
        // mechanism is unavailable.
        if r != 0 {
            close_from_brute_force();
        }
    }
    #[cfg(not(target_os = "linux"))]
    // SAFETY: bounded close(2) loop.
    unsafe {
        close_from_brute_force();
    }
    Ok(())
}

impl Drop for PyPySandboxedProc {
    fn drop(&mut self) {
        // Never leave an orphaned child if interact() bailed early.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    #[test]
    fn virtual_root_exposes_bin_and_tmp() {
        let exe = std::env::current_exe().unwrap();
        let root = build_virtual_root(&exe, None, None);
        let mut keys = root.keys().unwrap();
        keys.sort();
        assert_eq!(keys, vec!["bin".to_owned(), "tmp".to_owned()]);

        let bin = root.join("bin").unwrap();
        let bin_keys = bin.keys().unwrap();
        assert_eq!(bin_keys, vec!["pypy3-c".to_owned()]);

        // the exe is mounted read-only and openable
        let exe_node = bin.join("pypy3-c").unwrap();
        let mut data = Vec::new();
        exe_node.open().unwrap().read_to_end(&mut data).unwrap();
        assert!(!data.is_empty());
    }

    #[test]
    fn virtual_root_with_lib_mounts_lib() {
        let exe = std::env::current_exe().unwrap();
        let libdir = exe.parent().unwrap();
        let root = build_virtual_root(&exe, None, Some(libdir));
        let bin = root.join("bin").unwrap();
        let mut bin_keys = bin.keys().unwrap();
        bin_keys.sort();
        assert_eq!(bin_keys, vec!["lib".to_owned(), "pypy3-c".to_owned()]);
    }

    // The protocol loop itself is covered by sandlib.rs's in-memory test; this
    // exercises the spawn + reap path: a child that closes its stdout at once
    // produces a clean EOF, ending the loop with the child's exit code.
    #[test]
    fn interact_reaps_a_child_that_closes_immediately() {
        let candidates = ["/usr/bin/true", "/bin/true"];
        let exe = candidates.iter().find(|p| Path::new(p).exists());
        let Some(exe) = exe else {
            return; // no `true` binary on this platform: skip
        };
        let mut proc = PyPySandboxedProc::new(
            exe,
            &[],
            None,
            None,
            Some(Duration::from_secs(5)),
            false,
            None,
        )
        .unwrap();
        let code = proc.interact().unwrap();
        assert_eq!(code, 0);
    }
}
