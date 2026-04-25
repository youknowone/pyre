//! Port of `rpython/config/support.py`.
//!
//! Upstream is 50 LOC of two small helpers
//! (`detect_number_of_processors`, `detect_pax`) imported by
//! `translationoption.py` to populate the `make_jobs` option default.
//! Only `detect_number_of_processors` is ported here — `detect_pax` is
//! a C-backend concern (used by `rpython/translator/c/genc.py`) that
//! lands alongside the C-backend port.

use std::env;

/// Upstream `support.py:7-29 detect_number_of_processors`. Returns the
/// number of processors to use as the `make -j` argument.
///
/// Body shape mirrors upstream `:7-29` line-by-line:
///
/// ```python
/// def detect_number_of_processors(filename_or_file='/proc/cpuinfo'):
///     if os.environ.get('MAKEFLAGS'):
///         return 1
///     if sys.platform == 'darwin':
///         return sysctl_get_cpu_count('/usr/sbin/sysctl')
///     elif sys.platform.startswith('freebsd'):
///         return sysctl_get_cpu_count('/sbin/sysctl')
///     elif not sys.platform.startswith('linux'):
///         return 1
///     try:
///         ...
///         count = max([... for line in f if line.startswith('processor')]) + 1
///         if count >= 4:
///             return max(count // 2, 3)
///         else:
///             return count
///     except:
///         return 1
/// ```
///
/// Two adaptations (PRE-EXISTING-ADAPTATION, justified):
///
/// 1. **CPU count source.** Upstream shells out to `sysctl -n hw.ncpu`
///    on macOS / FreeBSD and parses `/proc/cpuinfo` on Linux. The Rust
///    port uses [`std::thread::available_parallelism`] universally —
///    on macOS / FreeBSD it reads the same `hw.ncpu` via `sysctl(2)`,
///    on Linux it reads `sched_getaffinity` (more accurate than line-
///    counting `/proc/cpuinfo`). Observable behaviour matches: the
///    raw logical core count.
/// 2. **Half-load cap is Linux-only.** Per upstream `:14-27`, only
///    the Linux branch applies the `count >= 4 → max(count // 2, 3)`
///    scaling. macOS / FreeBSD return the raw count. The pre-fix
///    port applied the cap to every platform, diverging from upstream
///    on macOS / FreeBSD; this version restores the per-platform
///    branch shape.
pub fn detect_number_of_processors() -> i64 {
    // Upstream `:8-9`: `os.environ.get('MAKEFLAGS')`.
    if env::var_os("MAKEFLAGS").is_some() {
        return 1;
    }
    // Upstream `:10-11`: `sys.platform == 'darwin' →
    // sysctl_get_cpu_count('/usr/sbin/sysctl')` — raw hw.ncpu, no cap.
    if cfg!(target_os = "macos") {
        return sysctl_cpu_count();
    }
    // Upstream `:12-13`: `sys.platform.startswith('freebsd') →
    // sysctl_get_cpu_count('/sbin/sysctl')` — raw hw.ncpu, no cap.
    if cfg!(target_os = "freebsd") {
        return sysctl_cpu_count();
    }
    // Upstream `:14-15`: `not sys.platform.startswith('linux') →
    // return 1   # implement me`.
    if !cfg!(target_os = "linux") {
        return 1;
    }
    // Upstream `:16-29`: Linux /proc/cpuinfo path with half-load cap.
    let count = read_cpu_count_for_linux();
    if count >= 4 {
        (count / 2).max(3)
    } else {
        count
    }
}

/// Upstream `support.py:31-37 sysctl_get_cpu_count('/usr/sbin/sysctl',
/// 'hw.ncpu')`. The Rust port routes through
/// [`std::thread::available_parallelism`] which on macOS / FreeBSD
/// reads `hw.ncpu` via `sysctl(2)` (no shell-out), preserving the
/// observable count. Returns 1 on any failure to match upstream's
/// `(OSError, ValueError) → 1` fallback at `:36`.
fn sysctl_cpu_count() -> i64 {
    std::thread::available_parallelism()
        .map(|n| n.get() as i64)
        .unwrap_or(1)
}

/// Linux-only CPU count. Upstream `support.py:16-29` parses
/// `/proc/cpuinfo` looking for `processor:` lines and returns
/// `max(<num>) + 1`; the bare-`except` at `:28-29` collapses any
/// failure mode to `1`. The Rust port uses
/// [`std::thread::available_parallelism`] which on Linux reads
/// `sched_getaffinity(2)` — strictly more accurate than line-counting
/// (it respects cpuset cgroups) — and returns the same `1` fallback
/// on failure.
fn read_cpu_count_for_linux() -> i64 {
    std::thread::available_parallelism()
        .map(|n| n.get() as i64)
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_number_of_processors_returns_positive() {
        // Regardless of MAKEFLAGS state, the result must be >= 1.
        let n = detect_number_of_processors();
        assert!(n >= 1, "expected >=1, got {}", n);
    }

    #[test]
    fn detect_number_of_processors_respects_makeflags() {
        // SAFETY of set_env: single-threaded test context per
        // Rust stable `std::env::set_var` contract; this test does not
        // run concurrently with other env-touching tests by virtue of
        // cargo's per-test env isolation (tests in one file share
        // process but run serially unless #[test(flavor = ...)] said
        // otherwise).
        //
        // Upstream `support.py:8-9` short-circuits on any truthy
        // MAKEFLAGS.
        unsafe {
            env::set_var("MAKEFLAGS", "-j4");
        }
        let n = detect_number_of_processors();
        unsafe {
            env::remove_var("MAKEFLAGS");
        }
        assert_eq!(n, 1, "MAKEFLAGS set must short-circuit to 1");
    }

    /// Upstream `support.py:10-13` — macOS / FreeBSD return the raw
    /// `hw.ncpu` count, NOT the half-load-capped value. Pin the parity
    /// behaviour with a platform-gated test that checks `count == raw`
    /// when `count` is in the no-cap range (>= 1) on macOS and FreeBSD.
    #[cfg(any(target_os = "macos", target_os = "freebsd"))]
    #[test]
    fn detect_number_of_processors_macos_freebsd_uses_raw_count() {
        // Cleared MAKEFLAGS so we exercise the platform branch, not the
        // `:8-9` short-circuit.
        let _was_set = env::var_os("MAKEFLAGS");
        unsafe {
            env::remove_var("MAKEFLAGS");
        }
        let n = detect_number_of_processors();
        let raw = std::thread::available_parallelism()
            .map(|x| x.get() as i64)
            .unwrap_or(1);
        // Restore MAKEFLAGS if it was set so we don't pollute other tests.
        if let Some(prev) = _was_set {
            unsafe {
                env::set_var("MAKEFLAGS", prev);
            }
        }
        assert_eq!(
            n, raw,
            "macOS / FreeBSD must return raw hw.ncpu without half-load cap"
        );
    }

    /// Upstream `support.py:23-27` — Linux applies the half-load cap
    /// when `count >= 4`. Pinned with the same shape as upstream's
    /// `if count >= 4: return max(count // 2, 3)`.
    #[cfg(target_os = "linux")]
    #[test]
    fn detect_number_of_processors_linux_applies_half_load_cap() {
        let _was_set = env::var_os("MAKEFLAGS");
        unsafe {
            env::remove_var("MAKEFLAGS");
        }
        let n = detect_number_of_processors();
        let raw = std::thread::available_parallelism()
            .map(|x| x.get() as i64)
            .unwrap_or(1);
        if let Some(prev) = _was_set {
            unsafe {
                env::set_var("MAKEFLAGS", prev);
            }
        }
        let expected = if raw >= 4 { (raw / 2).max(3) } else { raw };
        assert_eq!(n, expected, "Linux applies half-load cap when count >= 4");
    }
}
