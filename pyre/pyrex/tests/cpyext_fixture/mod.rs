//! Shared plumbing for the cpyext end-to-end tests: build a fixture against
//! pyre's own 3.14 ABI header and run a script that imports it.

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

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

/// The extension suffix this build's `_imp.extension_suffixes()` reports.
pub fn extension_suffix() -> &'static str {
    if cfg!(target_os = "macos") {
        ".pyre314-darwin.so"
    } else if cfg!(target_arch = "x86_64") {
        ".pyre314-x86_64-linux-gnu.so"
    } else if cfg!(target_arch = "aarch64") {
        ".pyre314-aarch64-linux-gnu.so"
    } else {
        ".pyre314-linux-gnu.so"
    }
}

/// A scratch directory on `PYTHONPATH` holding the compiled fixtures.
pub struct Fixtures {
    directory: PathBuf,
}

impl Fixtures {
    pub fn new(label: &str) -> Self {
        let unique = format!(
            "pyre-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let directory = std::env::temp_dir().join(unique);
        std::fs::create_dir(&directory).expect("create cpyext test directory");
        Self { directory }
    }

    pub fn path(&self) -> &Path {
        &self.directory
    }

    pub fn join(&self, name: &str) -> PathBuf {
        self.directory.join(name)
    }

    /// Compile `tests/fixtures/<name>.c` into an importable extension.
    pub fn compile(&self, name: &str) -> PathBuf {
        let root = repo_root();
        let extension = self.join(&format!("{name}{}", extension_suffix()));
        let source = root.join(format!("pyre/pyrex/tests/fixtures/{name}.c"));
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
        extension
    }

    /// Run `pyre -S -c <code>` with this directory on `PYTHONPATH`.
    pub fn run(&self, code: &str, env: &[(&str, &Path)]) -> Output {
        let mut command = Command::new(pyre_binary());
        command
            .arg("-S")
            .arg("-c")
            .arg(code)
            .env("PYTHONPATH", &self.directory);
        for (name, value) in env {
            command.env(name, value);
        }
        command.output().expect("run pyre cpyext fixture")
    }

    /// Run the script and require it to end with `marker` on stdout.
    pub fn expect_ok(&self, code: &str, env: &[(&str, &Path)], marker: &str) {
        let output = self.run(code, env);
        assert!(
            output.status.success(),
            "pyre failed\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), marker);
    }
}

impl Drop for Fixtures {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.directory);
    }
}
