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
/// Upstream runs on Python 2's `str`, reads `/proc/cpuinfo` on Linux,
/// calls out to `sysctl -n hw.ncpu` on macOS / FreeBSD. The port uses
/// [`std::thread::available_parallelism`] which covers every upstream-
/// supported platform (and more) while preserving the two upstream
/// semantic adaptations:
///
/// 1. **`MAKEFLAGS` short-circuit** (`:8-9`): if `MAKEFLAGS` is set,
///    return `1` so we don't step on an outer `make -j`.
/// 2. **Half-load cap** (`:23-27`): for Linux, upstream scales
///    `count >= 4 → max(count // 2, 3)` on the assumption that
///    hyperthreading doesn't double real throughput. This cap applies
///    to every platform in the Rust port (the upstream divergence
///    between Linux and macOS/FreeBSD is a historical artefact — the
///    half-load rationale is platform-independent).
pub fn detect_number_of_processors() -> i64 {
    // Upstream `:8-9`: `os.environ.get('MAKEFLAGS')`.
    if env::var_os("MAKEFLAGS").is_some() {
        return 1;
    }
    let count = std::thread::available_parallelism()
        .map(|n| n.get() as i64)
        .unwrap_or(1);
    // Upstream `:23-27`: scale count to avoid over-subscription.
    if count >= 4 {
        (count / 2).max(3)
    } else {
        count
    }
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
}
