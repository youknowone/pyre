//! The two lazily-installed app-level handles report a failure to the call that
//! forced them, instead of aborting the process.
//!
//! `aiter`/`anext` and the protocol-2 reduce are app-level functions in PyPy's
//! MixedModule. Pyre bundles their sources and executes each one into a fresh
//! namespace on the first call that needs it -- the builtins module dict is
//! populated inside `ExecutionContext::new`, before an execution context
//! exists, so neither can be installed at builtins-setup time.
//!
//! Executing a source is running Python: it builds a frame, answers the
//! recursion limit the program set, and reads the `builtins` the program may
//! have emptied. All three belong to the call that forced the install, and
//! before this each was a `panic!` on the installer's error arm.
//!
//! Only an end-to-end run sees it: the failure was a process abort, which no
//! in-process test can observe, and each arm has to be the FIRST such call of
//! the process or the cached handle answers instead.
//!
//! What is asserted is that the process survives and the program is told. The
//! exception's identity is not: pyre reaches these bodies through app-level
//! code where CPython uses C, so a `builtins` the program emptied reaches
//! pyre's `object()` and never reaches CPython's `aiter`. That difference is
//! the app-level port, not this hand-off.
//!
//! Do not invoke Cargo from this test: the parent `cargo test` holds the target
//! directory lock. `CARGO_BIN_EXE_pyre-dynasm` supplies the binary Cargo already
//! built for this integration test.

// `pyre-dynasm` is `required-features = ["dynasm"]`, so under
// `--no-default-features` the binary does not exist and `CARGO_BIN_EXE_…` is
// unset — which would be a compile error rather than a skip. Compile this file
// to nothing in that configuration instead.
#![cfg(feature = "dynasm")]

use std::path::PathBuf;
use std::process::Command;

/// The dynasm-backend `pyre` binary cargo built for this test.
const PYRE: &str = env!("CARGO_BIN_EXE_pyre-dynasm");

/// `aiter`'s source binds a module-level sentinel with `object()`, so an
/// emptied `builtins` is visible to the install rather than to the call.
const NO_OBJECT: &str = "\
import builtins
del builtins.object
try:
    aiter(3)
except BaseException as exc:
    print('raised', type(exc).__name__)
else:
    print('returned')
";

/// The install runs a frame of its own, so a program already at its limit is
/// what refuses it. `anext` is untouched by the `aiter` call above, but each
/// arm is its own process, so both are first.
const AT_THE_LIMIT: &str = "\
import sys
sys.setrecursionlimit(60)


def deep(n):
    return deep(n - 1) if n else FORCE


try:
    deep(58)
except BaseException as exc:
    print('raised', type(exc).__name__)
else:
    print('returned')
";

/// A handler that reaches the reduce protocol while the stack is still deep is
/// where upstream lets a `RecursionError` through and carries on.
const REDUCE_IN_A_HANDLER: &str = "\
def deep():
    try:
        deep()
    except RecursionError:
        try:
            [].__reduce_ex__(2)
        except RecursionError:
            pass


deep()
print('done')
";

fn run(name: &str, source: &str) -> (std::process::Output, String) {
    let mut path = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    path.push(format!("lazy_app_handles_{name}.py"));
    std::fs::write(&path, source).expect("write the program under test");
    let out = Command::new(PYRE)
        .arg(&path)
        .output()
        .expect("spawn pyre-dynasm");
    let report = format!(
        "{name} exited {:?}\n--- stdout ---\n{}\n--- stderr ---\n{}",
        out.status,
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
            .lines()
            .rev()
            .take(20)
            .collect::<Vec<_>>()
            .join("\n"),
    );
    (out, report)
}

/// The whole invariant: a Rust panic and a signal are both the process ending
/// where a Python program should have been told.
fn assert_survived(name: &str, source: &str) -> String {
    let (out, report) = run(name, source);
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(!stderr.contains("panicked at"), "{report}");
    // A signal leaves `code()` as `None`; 101 is the abort a `panic!` takes.
    assert_eq!(out.status.code(), Some(0), "{report}");
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

#[test]
fn aiter_reports_an_emptied_builtins_to_its_caller() {
    let stdout = assert_survived("no_object", NO_OBJECT);
    assert!(stdout.starts_with("raised "), "{stdout}");
}

#[test]
fn aiter_reports_the_recursion_limit_to_its_caller() {
    let stdout = assert_survived(
        "aiter_at_the_limit",
        &AT_THE_LIMIT.replace("FORCE", "aiter(3)"),
    );
    assert!(stdout.starts_with("raised "), "{stdout}");
}

#[test]
fn anext_reports_the_recursion_limit_to_its_caller() {
    let stdout = assert_survived(
        "anext_at_the_limit",
        &AT_THE_LIMIT.replace("FORCE", "anext(3)"),
    );
    assert!(stdout.starts_with("raised "), "{stdout}");
}

#[test]
fn the_reduce_protocol_reports_the_recursion_limit_to_its_caller() {
    assert_eq!(
        assert_survived("reduce_in_a_handler", REDUCE_IN_A_HANDLER),
        "done"
    );
}
