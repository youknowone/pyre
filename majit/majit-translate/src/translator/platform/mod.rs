//! Minimal port of `rpython/translator/platform/__init__.py`.
//!
//! This file owns the process execution substrate used by later C-backend
//! leaves.  It ports the observable shape of `ExecutionResult` (`:25-33`) and
//! `Platform.execute` (`:82-106`): copy/replace the environment, inject the
//! platform library path when compilation info is provided, run the program,
//! and normalize CRLF in captured output.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::translator::tool::taskengine::TaskError;

/// RPython `Platform.execute` receives `env` as a Python dict and immediately
/// does `env.copy()` (`__init__.py:83-86`), so the Rust port uses the direct
/// dict-shaped equivalent here.
pub type EnvMapping = HashMap<String, String>;

/// Port of upstream `ExecutionResult` (`__init__.py:25-33`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionResult {
    pub returncode: i32,
    pub out: String,
    pub err: String,
}

impl ExecutionResult {
    pub fn new(returncode: i32, out: Vec<u8>, err: Vec<u8>) -> Self {
        Self {
            returncode,
            out: normalize_newlines(String::from_utf8_lossy(&out).into_owned()),
            err: normalize_newlines(String::from_utf8_lossy(&err).into_owned()),
        }
    }
}

/// Minimal slice of upstream ExternalCompilationInfo used by
/// `Platform.execute(..., compilation_info=...)` (`__init__.py:96-101`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CompilationInfo {
    pub library_dirs: Vec<PathBuf>,
}

/// Minimal concrete platform object.  Upstream's base class is abstract
/// (`__init__.py:43-46`); this Rust type only ports the process-execution hook.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Platform {
    pub name: String,
}

impl Platform {
    pub fn host() -> Self {
        Self {
            name: std::env::consts::OS.to_string(),
        }
    }

    /// Port of upstream `Platform.execute` (`__init__.py:82-106`).
    pub fn execute(
        &self,
        executable: impl AsRef<Path>,
        args: Option<&[String]>,
        env: Option<&EnvMapping>,
        compilation_info: Option<&CompilationInfo>,
    ) -> Result<ExecutionResult, TaskError> {
        execute(executable, args, env, compilation_info)
    }
}

/// Free-function helper for call sites that do not need a platform value.
pub fn execute(
    executable: impl AsRef<Path>,
    args: Option<&[String]>,
    env: Option<&EnvMapping>,
    compilation_info: Option<&CompilationInfo>,
) -> Result<ExecutionResult, TaskError> {
    let mut command = Command::new(executable.as_ref());
    if let Some(args) = args {
        command.args(args);
    }

    // Upstream `:83-86`: `env is None` means copy process environment; a
    // provided env replaces it.  `Command` inherits by default, so only the
    // replacement branch needs `env_clear`.
    if let Some(env) = env {
        command.env_clear();
        command.envs(env);
        // Upstream `:88-92`: Windows programs often require SystemRoot.
        #[cfg(windows)]
        if !env.contains_key("SystemRoot") {
            if let Ok(system_root) = std::env::var("SystemRoot") {
                command.env("SystemRoot", system_root);
            }
        }
    }

    // Upstream `:94-101`: inject loader search path for compiled artifacts.
    if cfg!(unix) {
        if let Some(info) = compilation_info {
            let joined = info
                .library_dirs
                .iter()
                .map(|path| path.to_string_lossy())
                .collect::<Vec<_>>()
                .join(":");
            if cfg!(target_os = "macos") {
                command.env("DYLD_LIBRARY_PATH", joined);
            } else {
                command.env("LD_LIBRARY_PATH", joined);
            }
        }
    }

    let output = command.output().map_err(|e| TaskError {
        message: format!(
            "platform/__init__.py:103 Platform.execute failed to run {}: {e}",
            executable.as_ref().display()
        ),
    })?;
    Ok(ExecutionResult::new(
        output.status.code().unwrap_or(-1),
        output.stdout,
        output.stderr,
    ))
}

fn normalize_newlines(s: String) -> String {
    s.replace("\r\n", "\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn execution_result_normalizes_crlf() {
        let result = ExecutionResult::new(7, b"a\r\nb\r\n".to_vec(), b"e\r\n".to_vec());

        assert_eq!(result.returncode, 7);
        assert_eq!(result.out, "a\nb\n");
        assert_eq!(result.err, "e\n");
    }

    #[cfg(unix)]
    #[test]
    fn execute_runs_program_and_captures_output() {
        let args = vec!["-c".to_string(), "printf 'ok\\n'".to_string()];
        let result = execute("/bin/sh", Some(&args), None, None).expect("execute");

        assert_eq!(result.returncode, 0);
        assert_eq!(result.out, "ok\n");
        assert_eq!(result.err, "");
    }

    #[cfg(unix)]
    #[test]
    fn execute_replaces_environment_when_env_is_provided() {
        let args = vec![
            "-c".to_string(),
            "printf '%s' \"$RPY_PLATFORM_TEST_VALUE\"".to_string(),
        ];
        let mut env = EnvMapping::new();
        env.insert("RPY_PLATFORM_TEST_VALUE".to_string(), "visible".to_string());

        let result = execute("/bin/sh", Some(&args), Some(&env), None).expect("execute");

        assert_eq!(result.returncode, 0);
        assert_eq!(result.out, "visible");
    }
}
