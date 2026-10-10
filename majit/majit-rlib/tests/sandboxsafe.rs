//! `llexternal!` sandboxsafe substitution (`rsandbox.py` `get_sandbox_stub`).
//!
//! `sandboxsafe = true` is always the real expansion. Non-sandboxsafe
//! declarations stub under `any(not(host_env), sandbox)`.

#[cfg(any(not(feature = "host_env"), feature = "sandbox"))]
use majit_rlib::rffi::SandboxStub;
use majit_rlib::rffi::llexternal;
#[cfg(feature = "sandbox")]
use majit_rlib::rffi::register_sandbox_stub_publisher;
#[cfg(feature = "sandbox")]
use std::sync::atomic::{AtomicBool, Ordering};

#[unsafe(no_mangle)]
unsafe extern "C" fn majit_rffi_sandboxsafe_probe() -> i32 {
    42
}

llexternal!(
    pub probe_safe = "majit_rffi_sandboxsafe_probe",
    [],
    i32,
    sandboxsafe = true,
    releasegil = false,
);

llexternal!(
    pub probe_unsafe = "majit_rffi_sandboxsafe_probe",
    [],
    i32,
    releasegil = false,
);

#[cfg(all(feature = "host_env", not(feature = "sandbox")))]
#[test]
fn default_calls_the_extern() {
    assert_eq!(unsafe { probe_safe() }, 42);
    assert_eq!(unsafe { probe_unsafe() }, 42);
}

#[cfg(all(feature = "host_env", feature = "sandbox"))]
#[test]
fn sandbox_stubs_non_safe_and_calls_safe() {
    assert_eq!(unsafe { probe_safe() }, 42);

    let panicked = std::panic::catch_unwind(|| unsafe { probe_unsafe() });
    let payload = panicked.expect_err("non-sandboxsafe external panics");
    let stub = payload
        .downcast_ref::<SandboxStub>()
        .expect("SandboxStub payload");
    assert_eq!(stub.fnname, "majit_rffi_sandboxsafe_probe");

    static PUBLISHED: AtomicBool = AtomicBool::new(false);
    fn publisher(fnname: &'static str) {
        assert_eq!(fnname, "majit_rffi_sandboxsafe_probe");
        PUBLISHED.store(true, Ordering::SeqCst);
    }
    register_sandbox_stub_publisher(publisher);
    let zero = unsafe { ccall_probe_unsafe() };
    assert_eq!(zero, 0);
    assert!(PUBLISHED.load(Ordering::SeqCst));
}

#[cfg(not(feature = "host_env"))]
#[test]
fn no_host_env_keeps_sandboxsafe_real_and_stubs_the_rest() {
    assert_eq!(unsafe { probe_safe() }, 42);

    let panicked = std::panic::catch_unwind(|| unsafe { probe_unsafe() });
    let payload = panicked.expect_err("non-sandboxsafe external is a stub without host_env");
    let stub = payload
        .downcast_ref::<SandboxStub>()
        .expect("SandboxStub payload");
    assert_eq!(stub.fnname, "majit_rffi_sandboxsafe_probe");
}
