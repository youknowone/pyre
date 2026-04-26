//! Port of `rpython/tool/udir.py`.
//!
//! Upstream creates a process-wide "usession" directory named
//! `$PYPY_USESSION_DIR/usession-$PYPY_USESSION_BASENAME-N` and exposes it
//! as module global `udir` (`udir.py:28-51`).  This Rust port mirrors
//! upstream's environment-variable contract, the basename defaulting that
//! consults `rpython.tool.version.get_repo_version_info` (`udir.py:31-40`),
//! and the numbered-directory + cleanup machinery (`udir.py:46`,
//! `py.path.local.make_numbered_dir(..., keep=PYPY_KEEP)`).
//!
//! The upstream `py.path.local.make_numbered_dir` cleanup keeps the most
//! recent N (default 3) numbered directories, removing older ones when a
//! new one is created.  The Rust port replays the same eviction loop in
//! place rather than depending on `py.path.local`.
//!
//! The optional `usession-<basename>-$USER` symlink upstream creates on
//! Unix (`udir.py:9-13` comment) is replicated when `$USER` is set and
//! the platform supports symlinks.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

/// Upstream module global `PYPY_KEEP = int(os.environ.get(
/// 'PYPY_USESSION_KEEP', '3'))` (`udir.py:26`).
fn pypy_keep() -> usize {
    env::var("PYPY_USESSION_KEEP")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(3)
}

/// Process-wide equivalent of upstream module global `udir` (`udir.py:50-51`).
pub fn udir() -> &'static Path {
    static UDIR: OnceLock<PathBuf> = OnceLock::new();
    UDIR.get_or_init(|| {
        make_udir(
            env::var_os("PYPY_USESSION_DIR").map(PathBuf::from),
            env::var("PYPY_USESSION_BASENAME").ok(),
        )
        .expect("failed to create RPython usession directory")
    })
    .as_path()
}

/// Port of upstream `make_udir(dir=None, basename=None)` (`udir.py:28-48`).
pub fn make_udir(dir: Option<PathBuf>, basename: Option<String>) -> std::io::Result<PathBuf> {
    let root = dir.unwrap_or_else(env::temp_dir);
    // Upstream `udir.py:31-40`: if `basename is None`, look up the repo's
    // `(hgtag, hgid)` via `get_repo_version_info`.  The first element
    // (tag/branch) is used as basename; '?' is rewritten to 'unknown'.
    let basename = match basename {
        Some(b) => b,
        None => match get_repo_version_info() {
            Some((hgtag, _hgid)) => {
                if hgtag == "?" {
                    "unknown".to_string()
                } else {
                    hgtag
                }
            }
            None => String::new(),
        },
    };
    let mut basename = basename.replace('/', "--");
    // Upstream `udir.py:42-45`: force leading and trailing '-'.
    if !basename.starts_with('-') {
        basename.insert(0, '-');
    }
    if !basename.ends_with('-') {
        basename.push('-');
    }
    let prefix = format!("usession{basename}");
    make_numbered_dir(&root, &prefix, pypy_keep())
}

/// Port of `py.path.local.make_numbered_dir(rootdir, prefix, keep)` as
/// invoked at upstream `udir.py:46-48`.  Creates a fresh numbered
/// directory under `root`, prunes older numbered directories (keeping
/// `keep` most-recent ones) and refreshes the optional `$USER` symlink.
fn make_numbered_dir(root: &Path, prefix: &str, keep: usize) -> std::io::Result<PathBuf> {
    fs::create_dir_all(root)?;
    let candidate = pick_next_numbered_path(root, prefix)?;
    fs::create_dir(&candidate)?;
    cleanup_older_numbered_dirs(root, prefix, keep, &candidate)?;
    refresh_user_symlink(root, prefix, &candidate);
    Ok(candidate)
}

/// Atomic-ish number selection: walk existing siblings, pick `max + 1`.
fn pick_next_numbered_path(root: &Path, prefix: &str) -> std::io::Result<PathBuf> {
    let mut highest: Option<u32> = None;
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let name = match entry.file_name().into_string() {
            Ok(n) => n,
            Err(_) => continue,
        };
        if let Some(suffix) = name.strip_prefix(prefix) {
            if let Ok(n) = suffix.parse::<u32>() {
                highest = Some(highest.map_or(n, |cur| cur.max(n)));
            }
        }
    }
    let next = highest.map_or(0, |n| n + 1);
    Ok(root.join(format!("{prefix}{next}")))
}

/// Replays `py.path.local.make_numbered_dir`'s `keep` semantics: leave
/// the most recent `keep` numbered children intact, remove older ones.
fn cleanup_older_numbered_dirs(
    root: &Path,
    prefix: &str,
    keep: usize,
    fresh: &Path,
) -> std::io::Result<()> {
    let mut entries: Vec<(u32, PathBuf)> = Vec::new();
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let name = match entry.file_name().into_string() {
            Ok(n) => n,
            Err(_) => continue,
        };
        if let Some(suffix) = name.strip_prefix(prefix) {
            if let Ok(n) = suffix.parse::<u32>() {
                entries.push((n, path));
            }
        }
    }
    entries.sort_by_key(|(n, _)| *n);
    let drop = entries.len().saturating_sub(keep);
    for (_, path) in entries.into_iter().take(drop) {
        if path == fresh {
            continue;
        }
        let _ = fs::remove_dir_all(&path);
    }
    Ok(())
}

/// Port of the `usession-<basename>-$USER` convenience symlink mentioned
/// at `udir.py:9-13`.  Best-effort: ignore failures.
fn refresh_user_symlink(root: &Path, prefix: &str, target: &Path) {
    #[cfg(unix)]
    {
        if let Ok(user) = env::var("USER") {
            if user.is_empty() {
                return;
            }
            let link = root.join(format!("{prefix}{user}"));
            let _ = fs::remove_file(&link);
            let _ = std::os::unix::fs::symlink(target, &link);
        }
    }
    #[cfg(not(unix))]
    {
        let _ = (root, prefix, target);
    }
}

/// Port of `rpython.tool.version.get_repo_version_info` (`version.py:22-31`)
/// for the git path only.  Returns `(branch_or_tag, revision_id)` when the
/// surrounding source tree is a git checkout, `None` otherwise.  Mercurial
/// is upstream's other branch (`version.py:48-94`); this port targets the
/// majit/pypy git mirror layout that pyre-stdlib actually ships in.
fn get_repo_version_info() -> Option<(String, String)> {
    let git_root = locate_git_root()?;
    let revision = run_git(&git_root, &["rev-parse", "HEAD"])
        .map(|s| s.chars().take(12).collect::<String>())?;

    if let Some(tag) = run_git(&git_root, &["describe", "--tags", "--exact-match"]) {
        return Some((tag, revision));
    }

    if let Some(branch_text) = run_git(&git_root, &["branch"]) {
        for line in branch_text.lines() {
            if let Some(rest) = line.strip_prefix("* ") {
                let rest = rest.trim();
                let branch = if rest == "(no branch)" {
                    "pypy-HEAD"
                } else if rest.starts_with("(HEAD detached") {
                    "pypy-HEAD"
                } else {
                    rest
                };
                return Some((branch.to_string(), revision));
            }
        }
        return Some(("pypy-HEAD".to_string(), revision));
    }
    Some(("pypy-HEAD".to_string(), revision))
}

fn locate_git_root() -> Option<PathBuf> {
    let mut cur: PathBuf = env::current_dir().ok()?;
    loop {
        if cur.join(".git").exists() {
            return Some(cur);
        }
        if !cur.pop() {
            return None;
        }
    }
}

fn run_git(cwd: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(cwd)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let s = String::from_utf8(output.stdout).ok()?.trim().to_string();
    if s.is_empty() { None } else { Some(s) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn make_udir_matches_upstream_prefix_rules() {
        let root = tempfile::tempdir().expect("tempdir");
        let first = make_udir(Some(root.path().to_path_buf()), Some("fun/bar".to_string()))
            .expect("first udir");
        let second = make_udir(Some(root.path().to_path_buf()), Some("fun/bar".to_string()))
            .expect("second udir");

        assert_eq!(first.file_name().unwrap(), "usession-fun--bar-0");
        assert_eq!(second.file_name().unwrap(), "usession-fun--bar-1");
    }

    #[test]
    fn make_udir_normalizes_empty_basename() {
        let root = tempfile::tempdir().expect("tempdir");
        let path = make_udir(Some(root.path().to_path_buf()), Some(String::new())).expect("udir");

        assert_eq!(path.file_name().unwrap(), "usession-0");
    }

    #[test]
    fn cleanup_keeps_only_n_most_recent() {
        // Upstream `py.path.local.make_numbered_dir` keeps `PYPY_KEEP`
        // most-recent numbered children. With keep=2 + 5 fresh dirs, the
        // 3 oldest must vanish.
        let root = tempfile::tempdir().expect("tempdir");
        for _ in 0..5 {
            make_numbered_dir(root.path(), "usession-x-", 2).expect("mkdir");
        }
        let mut numbered: Vec<u32> = Vec::new();
        for entry in fs::read_dir(root.path()).unwrap() {
            let entry = entry.unwrap();
            let name = entry.file_name().into_string().unwrap();
            if let Some(s) = name.strip_prefix("usession-x-") {
                if let Ok(n) = s.parse::<u32>() {
                    numbered.push(n);
                }
            }
        }
        numbered.sort();
        assert_eq!(numbered, vec![3, 4]);
    }

    #[test]
    fn make_udir_with_unknown_repo_uses_repo_default_or_empty() {
        // Upstream falls back to `''` when get_repo_version_info returns
        // None, then prepends '-' (`udir.py:42-45`). The Rust port must
        // produce a directory whose name still starts with `usession-`
        // and ends in a number.
        let root = tempfile::tempdir().expect("tempdir");
        let path = make_udir(Some(root.path().to_path_buf()), None).expect("udir");
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        assert!(name.starts_with("usession-"));
        assert!(
            name.chars().last().is_some_and(|c| c.is_ascii_digit()),
            "expected trailing digit, got {name}"
        );
    }
}
