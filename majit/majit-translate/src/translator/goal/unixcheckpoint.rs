//! Port of `rpython/translator/goal/unixcheckpoint.py`.
//!
//! Upstream is 85 LOC of three top-level helpers
//! (`restart_process`, `restartable_point_fork`,
//! `restartable_point_nofork`) plus a `restartable_point` re-export
//! that selects fork-vs-nofork by `sys.platform`. Driver
//! `task_jittest_lltype` (`driver.py:371-372`) is the only consumer:
//! it calls `restartable_point(auto='run')` once, before importing
//! `jittest` and invoking `jittest.jittest(self)`.
//!
//! The fork / `os.execv` / `raw_input` body cannot run inside a Rust
//! library without committing the host process to forking and
//! interactive prompts — both are observable side effects that the
//! upstream user opts into. The local port keeps the upstream call
//! signature, validates the `auto` argument matches the `'run'` /
//! `'cont'` / `'quit'` / `'pdb'` / `'restart-it-all'` vocabulary, and
//! returns [`TaskError`] documenting the still-unported leaf.

use crate::translator::tool::taskengine::TaskError;

/// Port of upstream `restartable_point(auto=None)` (the platform-
/// selected re-export at `:74-77`).
///
/// `auto` follows the upstream prompt vocabulary at `:23-35`:
/// * `Some("run") | Some("cont")` — break out of the inner loop and
///   proceed to the fork point. Upstream `:23-24`.
/// * `Some("quit")` — `raise SystemExit`. Upstream `:25-26`.
/// * `Some("pdb")` — open `pdb`. Upstream `:27-33`.
/// * `Some("restart-it-all")` — call `restart_process()`
///   (`os.execv(sys.executable, [sys.executable] + sys.argv)`).
///   Upstream `:34-35` + `:3-5`.
/// * `None` — read a line from stdin. Upstream `:18-22`.
///
/// Until the fork-then-wait body lands, the local port surfaces a
/// `TaskError` citing the upstream line so callers see a structural
/// stub rather than a panic.
pub fn restartable_point(auto: Option<&str>) -> Result<(), TaskError> {
    // Upstream `:23-35`: validate the `auto` token against the prompt
    // vocabulary so a typo at the call site is caught early.
    if let Some(tok) = auto {
        match tok {
            "run" | "cont" | "quit" | "pdb" | "restart-it-all" => {}
            _ => {
                return Err(TaskError {
                    message: format!(
                        "unixcheckpoint.py:23 restartable_point: unknown auto token {tok:?}; expected one of run/cont/quit/pdb/restart-it-all"
                    ),
                });
            }
        }
    }
    // Upstream `:7-77`: `restartable_point_fork` / `restartable_point_nofork`.
    // Both bodies require `os.fork`, `os.execv`, `os.waitpid` and an
    // interactive-prompt loop. None are ported here.
    Err(TaskError {
        message: "unixcheckpoint.py:7 restartable_point — fork / execv / waitpid / raw_input loop not yet ported".to_string(),
    })
}

/// Port of upstream `restart_process()` at `:3-5`.
///
/// Upstream calls `os.execv(sys.executable, [sys.executable] + sys.argv)`
/// which replaces the running process. The local port surfaces a
/// `TaskError` so callers see a structural stub rather than the host
/// being unconditionally re-execed.
pub fn restart_process() -> Result<(), TaskError> {
    Err(TaskError {
        message: "unixcheckpoint.py:3 restart_process — os.execv replacement not yet ported"
            .to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restartable_point_validates_auto_vocabulary() {
        // Upstream `:23-35` accepts: 'run', 'cont', 'quit', 'pdb',
        // 'restart-it-all'.
        for tok in [
            Some("run"),
            Some("cont"),
            Some("quit"),
            Some("pdb"),
            Some("restart-it-all"),
            None,
        ] {
            let err = restartable_point(tok).expect_err("must surface DEFERRED");
            assert!(
                err.message.contains("unixcheckpoint.py:7"),
                "expected DEFERRED message for tok {tok:?}, got: {}",
                err.message
            );
        }
    }

    #[test]
    fn restartable_point_rejects_unknown_token() {
        // Upstream prompt loop at `:23-35` doesn't have a fall-through
        // case for unknown tokens — they print and re-prompt. The local
        // port catches the typo at call time.
        let err = restartable_point(Some("yolo")).expect_err("must reject typo");
        assert!(
            err.message.contains("unknown auto token"),
            "{}",
            err.message
        );
    }

    #[test]
    fn restart_process_surfaces_task_error() {
        let err = restart_process().expect_err("must be DEFERRED");
        assert!(
            err.message.contains("unixcheckpoint.py:3"),
            "{}",
            err.message
        );
    }
}
