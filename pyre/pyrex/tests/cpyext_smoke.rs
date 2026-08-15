//! End-to-end check for the first PyPy-cpyext vertical slice.
//!
//! The fixture is compiled against pyre's own 3.14 ABI header, discovered by
//! the normal import path, loaded through `PyInit_*`, and created by the C call
//! back into the executable's exported `PyModule_Create2`.

#![cfg(all(
    feature = "cpyext",
    not(feature = "sandbox"),
    any(target_os = "macos", target_os = "linux")
))]

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("pyrex manifest sits two levels below the repo root")
        .to_path_buf()
}

#[cfg(feature = "dynasm")]
fn pyre_binary() -> &'static str {
    env!("CARGO_BIN_EXE_pyre-dynasm")
}

#[cfg(all(not(feature = "dynasm"), feature = "cranelift"))]
fn pyre_binary() -> &'static str {
    env!("CARGO_BIN_EXE_pyre-cranelift")
}

#[cfg(not(any(feature = "dynasm", feature = "cranelift")))]
fn pyre_binary() -> &'static str {
    env!("CARGO_BIN_EXE_pyre")
}

#[test]
fn imports_single_phase_module_created_through_c_api() {
    let root = repo_root();
    let unique = format!(
        "pyre-cpyext-smoke-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let out_dir = std::env::temp_dir().join(unique);
    std::fs::create_dir(&out_dir).expect("create cpyext test directory");

    let suffix = if cfg!(target_os = "macos") {
        ".pyre314-darwin.so"
    } else if cfg!(target_arch = "x86_64") {
        ".pyre314-x86_64-linux-gnu.so"
    } else if cfg!(target_arch = "aarch64") {
        ".pyre314-aarch64-linux-gnu.so"
    } else {
        ".pyre314-linux-gnu.so"
    };
    let extension = out_dir.join(format!("cpyext_smoke{suffix}"));
    let init_marker = out_dir.join("initialized");
    let unload_marker = out_dir.join("unloaded");
    let source = root.join("pyre/pyrex/tests/fixtures/cpyext_smoke.c");
    let include = root.join("include/pyre3.14t");
    let mut cc = Command::new(std::env::var_os("CC").unwrap_or_else(|| "cc".into()));
    cc.arg(&source)
        .arg("-I")
        .arg(&include)
        .arg("-o")
        .arg(&extension);
    if cfg!(target_os = "macos") {
        cc.args(["-bundle", "-undefined", "dynamic_lookup"]);
    } else {
        cc.args(["-fPIC", "-shared"]);
    }
    let output = cc.output().expect("run C compiler");
    assert!(
        output.status.success(),
        "C compiler failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let code = r#"
import _imp
import _sysconfig
import sys
expected_suffix = %SUFFIX%
assert _imp.extension_suffixes() == [expected_suffix]
config = _sysconfig.config_vars()
assert config['EXT_SUFFIX'] == expected_suffix
assert expected_suffix == '.' + config['SOABI'] + '.so'
old_dlopenflags = sys.getdlopenflags()
sys.setdlopenflags(0)
try:
    import cpyext_smoke
finally:
    sys.setdlopenflags(old_dlopenflags)
again = __import__('cpyext_smoke')
assert cpyext_smoke is again
assert cpyext_smoke.__name__ == 'cpyext_smoke'
assert cpyext_smoke.__doc__ == 'pyre cpyext smoke module'
assert cpyext_smoke.__file__.endswith(_imp.extension_suffixes()[0])
first = cpyext_smoke
first.__doc__ = 'mutated'
first.runtime_only = 1
del sys.modules['cpyext_smoke']
del cpyext_smoke
reloaded = __import__('cpyext_smoke')
assert reloaded is not first
assert reloaded.__name__ == 'cpyext_smoke'
assert reloaded.__doc__ == 'pyre cpyext smoke module'
assert not hasattr(reloaded, 'runtime_only')
try:
    with open(%UNLOAD_MARKER%, 'rb'):
        pass
except FileNotFoundError:
    pass
else:
    raise AssertionError('extension unloaded during cache restore')
print('cpyext-smoke-ok')
"#
    .replace("%SUFFIX%", &format!("{suffix:?}"))
    .replace(
        "%UNLOAD_MARKER%",
        &format!("{:?}", unload_marker.to_string_lossy()),
    );
    let output = Command::new(pyre_binary())
        .arg("-S")
        .arg("-c")
        .arg(&code)
        .env("PYTHONPATH", &out_dir)
        .env("PYRE_CPYEXT_INIT_MARKER", &init_marker)
        .env("PYRE_CPYEXT_UNLOAD_MARKER", &unload_marker)
        .output()
        .expect("run pyre cpyext fixture");
    assert!(
        output.status.success(),
        "pyre failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&output.stdout).trim(),
        "cpyext-smoke-ok"
    );

    let _ = std::fs::remove_dir_all(out_dir);
}
