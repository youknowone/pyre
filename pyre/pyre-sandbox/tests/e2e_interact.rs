//! End-to-end sandbox test (port of `pypy/sandbox/test/test_pypy_interact.py`).
//!
//! Builds `pyre` with the `sandbox` feature — the variant whose mediated OS
//! calls are compiled into marshalling trampolines — and drives it through the
//! Rust controller (`pyre interact`) over a virtual filesystem.  The same
//! binary serves as both the trusted controller (the `interact` subcommand
//! never touches `host_seam`) and the untrusted child.
//!
//! Marked `#[ignore]` because it shells out to a release `cargo build`; run it
//! explicitly with:
//!
//! ```text
//! cargo test -p pyre-sandbox --test e2e_interact -- --ignored --nocapture
//! ```
//!
//! The probes use only builtin modules (`posix`/`time`/`sys`/`_locale`): the
//! controller mounts no stdlib, so `import os` (a stdlib `.py`) is unavailable.
//!
//! Unix-only: the `controller` it drives compiles only on unix, so the whole
//! test is gated out elsewhere to keep `cargo test --all` green on non-unix.
#![cfg(unix)]

use std::path::PathBuf;
use std::process::Command;

/// Workspace root is two directories above this crate (`pyre/pyre-sandbox`).
fn workspace_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop(); // pyre/
    p.pop(); // workspace root
    p
}

/// Build the sandbox `pyre` binary once and return its path.  Cargo skips the
/// work when the binary is already current, so repeated runs are cheap.
fn build_sandbox_pyre() -> PathBuf {
    let root = workspace_root();
    let status = Command::new(env!("CARGO"))
        .current_dir(&root)
        .args([
            "build",
            "--release",
            "-p",
            "pyrex",
            "--bin",
            "pyre",
            "--features",
            "sandbox",
        ])
        .status()
        .expect("spawn cargo build");
    assert!(status.success(), "building sandbox pyre failed");
    let bin = root.join("target/release/pyre");
    assert!(bin.is_file(), "sandbox pyre missing at {}", bin.display());
    bin
}

/// Run `pyre interact --tmp <root> <pyre> -c <script>` and return
/// `(stdout, exit_code)`.  The controller relays the child's mediated stdout to
/// its own, so `stdout` is the program output seen through the protocol.
fn interact(pyre: &PathBuf, tmp: &std::path::Path, script: &str) -> (String, i32) {
    let out = Command::new(pyre)
        .args([
            "interact",
            "--tmp",
            tmp.to_str().unwrap(),
            "--timeout",
            "20",
            pyre.to_str().unwrap(),
            "-c",
            script,
        ])
        .output()
        .expect("spawn pyre interact");
    let code = out.status.code().unwrap_or(-1);
    // A signal-killed guest makes the controller return -1 (→255) with the
    // cause otherwise hidden. On any non-clean exit, surface the terminating
    // signal and the guest's stderr (which carries the seccomp handler's
    // "blocked syscall N" line, if it ran) so a CI failure is diagnosable.
    if code != 0 {
        #[cfg(unix)]
        let signal = std::os::unix::process::ExitStatusExt::signal(&out.status);
        #[cfg(not(unix))]
        let signal: Option<i32> = None;
        eprintln!(
            "[e2e] interact exited non-zero: code={code} signal={signal:?} status={:?}\n[e2e] guest stderr:\n{}",
            out.status,
            String::from_utf8_lossy(&out.stderr),
        );
    }
    (String::from_utf8_lossy(&out.stdout).into_owned(), code)
}

#[test]
#[ignore = "builds a release sandbox binary; run with --ignored"]
fn sandbox_reads_virtual_file_and_blocks_escapes() {
    let pyre = build_sandbox_pyre();
    let tmp = tempfile::tempdir().expect("tempdir");
    std::fs::write(tmp.path().join("hello.txt"), b"sandbox-ok").unwrap();

    // 1. A file inside the virtual root reads back through the controller.
    let (out, code) = interact(&pyre, tmp.path(), r#"print(open("/tmp/hello.txt").read())"#);
    assert_eq!(code, 0, "virtual read exited non-zero");
    assert!(out.contains("sandbox-ok"), "virtual read produced: {out:?}");

    // 2. The virtual cwd is `/tmp`, not the controller's real directory.
    let (out, code) = interact(
        &pyre,
        tmp.path(),
        "import posix\nprint('cwd:', posix.getcwd())",
    );
    assert_eq!(code, 0);
    assert!(out.contains("cwd: /tmp"), "getcwd produced: {out:?}");

    // 3. Reading outside the virtual root is denied (the node is absent).
    let (out, _) = interact(
        &pyre,
        tmp.path(),
        "try:\n    open('/etc/passwd').read()\n    print('LEAK')\nexcept OSError:\n    print('BLOCKED')",
    );
    assert!(out.contains("BLOCKED"), "escape read produced: {out:?}");
    assert!(!out.contains("LEAK"), "escape read leaked: {out:?}");
    assert!(!out.contains("root:"), "escape read leaked passwd: {out:?}");

    // 4. Write attempts are rejected — the controller is read-only. Use a
    //    per-pid name so a regression that escaped to the real host temp dir
    //    cannot collide with a concurrent run or a pre-planted symlink.
    let sentinel = format!("evil-{}.txt", std::process::id());
    let host_escape = std::env::temp_dir().join(&sentinel);
    let _ = std::fs::remove_file(&host_escape);
    let script = format!(
        "try:\n    open('/tmp/{sentinel}', 'w').write('x')\n    print('WROTE')\nexcept OSError:\n    print('BLOCKED')"
    );
    let (out, _) = interact(&pyre, tmp.path(), &script);
    assert!(out.contains("BLOCKED"), "write attempt produced: {out:?}");
    assert!(!out.contains("WROTE"), "write attempt succeeded: {out:?}");
    // The controller created the file nowhere: neither in the virtual root's
    // real backing dir nor — on a seam-bypass regression — the real host temp.
    assert!(
        !tmp.path().join(&sentinel).exists(),
        "write escaped to the virtual root's backing dir"
    );
    assert!(
        !host_escape.exists(),
        "write escaped to the real host temp dir at {host_escape:?}"
    );
    let _ = std::fs::remove_file(&host_escape);
}

/// The audited escape surface stays closed: every host-access builtin that the
/// 2026-06-27 escape audit found reachable must raise rather than touch the
/// host, the pure-computation survivors must still work, and the host-access
/// modules must be absent.  Regression guard for the per-module sandbox stubs,
/// the seam-backed import provider, and the module omissions.
#[test]
#[ignore = "builds a release sandbox binary; run with --ignored"]
fn sandbox_blocks_known_escapes() {
    let pyre = build_sandbox_pyre();
    let tmp = tempfile::tempdir().expect("tempdir");

    // Each `blocked` entry must raise (the function is stubbed/unavailable); a
    // value return means it executed against the host == a regression.
    let script = r#"
import posix, time, sys, _locale
fails = []
def must_raise(name, fn):
    try:
        fn()
        fails.append(name + ":ran")
    except Exception:
        pass

# posix host-access surface (critical/high/info-leak)
must_raise("sendfile",   lambda: posix.sendfile(1, 0, None, 16))
must_raise("readlink",   lambda: posix.readlink("/etc/localtime"))
must_raise("scandir",    lambda: list(posix.scandir("/etc")))
must_raise("statvfs",    lambda: posix.statvfs("/"))
must_raise("getpid",     lambda: posix.getpid())
must_raise("getppid",    lambda: posix.getppid())
must_raise("uname",      lambda: posix.uname())
must_raise("umask",      lambda: posix.umask(0))
must_raise("getlogin",   lambda: posix.getlogin())
must_raise("system",     lambda: posix.system("echo hi"))
must_raise("fork",       lambda: posix.fork())
# time/locale host-state readers
must_raise("localtime",  lambda: time.localtime())
must_raise("mktime",     lambda: time.mktime((2020, 1, 1, 0, 0, 0, 0, 0, -1)))
must_raise("ctime",      lambda: time.ctime(0))
must_raise("strftime",   lambda: time.strftime("%Y"))
must_raise("setlocale",  lambda: _locale.setlocale(6))
must_raise("localeconv", lambda: _locale.localeconv())
must_raise("nl_langinfo", lambda: _locale.nl_langinfo(0))

# pure-computation survivors must still work
if time.gmtime(0)[:6] != (1970, 1, 1, 0, 0, 0):
    fails.append("gmtime:broken")
if not isinstance(time.time(), float):
    fails.append("time:broken")

# os.urandom and _random are mediated through the trusted controller (entropy
# served by the controller), not stubbed: they must work, not raise. _random is
# a builtin, so this needs no mounted stdlib. _random.Random() with no argument
# default-seeds via os.urandom(8) — the exact path being mediated.
if len(posix.urandom(8)) != 8:
    fails.append("urandom:badlen")
import _random
_r = _random.Random().random()
if not (isinstance(_r, float) and 0.0 <= _r < 1.0):
    fails.append("_random:" + repr(_r))

# no host path/username leak via sys.executable
if sys.executable != "/bin/pyre":
    fails.append("executable:" + sys.executable)

# host-access modules are compiled out / not registered
for m in ("_socket", "_ctypes", "_posixsubprocess", "_multiprocessing",
          "_signal", "signal", "fcntl", "termios", "select", "mmap",
          "resource", "syslog", "pwd", "grp"):
    try:
        __import__(m)
        fails.append("import:" + m)
    except ImportError:
        pass

print("FAILS:" + ",".join(fails) if fails else "ALL-BLOCKED")
"#;

    let (out, code) = interact(&pyre, tmp.path(), script);
    assert_eq!(code, 0, "escape-probe exited non-zero: {out:?}");
    assert!(
        out.contains("ALL-BLOCKED"),
        "escape regression detected: {out:?}"
    );
}
