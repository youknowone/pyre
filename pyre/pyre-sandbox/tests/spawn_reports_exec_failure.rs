//! `PyPySandboxedProc::new` reports an executable that will not exec.
//!
//! The controller closes every inherited fd before the child execs, and the
//! one fd it must not close is the one `std::process::Command` keeps for the
//! child to report a failed `execvp` on.  With that fd gone the report is
//! written nowhere, the parent reads EOF, and a spawn that never happened is
//! indistinguishable from one that did -- so `pyre interact /etc/hosts` gets a
//! controller talking to a corpse instead of the one-line error it owes.
//!
//! Unix-only, like the controller it drives.
#![cfg(unix)]

use pyre_sandbox::controller::PyPySandboxedProc;

/// A path that exists and is readable but carries no execute bit, so `execvp`
/// fails with `EACCES` rather than `ENOENT`.  Both reach the same report; this
/// one also proves the failure is the kernel's and not a missing-file check the
/// controller could have made itself.
const NOT_EXECUTABLE: &str = "/etc/hosts";

#[test]
fn a_non_executable_target_is_an_error_not_a_child() {
    let spawned = PyPySandboxedProc::new(NOT_EXECUTABLE, &[], None, None, None, false, None);
    let Err(error) = spawned else {
        panic!("spawning {NOT_EXECUTABLE} reported success");
    };
    assert_eq!(
        error.raw_os_error(),
        Some(libc::EACCES),
        "{error} ({error:?})"
    );
}

#[test]
fn a_missing_target_is_an_error_not_a_child() {
    let spawned = PyPySandboxedProc::new(
        "/nonexistent/pyre-sandbox-child",
        &[],
        None,
        None,
        None,
        false,
        None,
    );
    let Err(error) = spawned else {
        panic!("spawning a missing path reported success");
    };
    assert_eq!(
        error.raw_os_error(),
        Some(libc::ENOENT),
        "{error} ({error:?})"
    );
}
