//! Implicitly-raisable exception set used by the annotator.
//!
//! RPython upstream: `rpython/annotator/exception.py` (7 LOC).
//!
//! Phase 5 P5.1 port.
//!
//! Rust adaptation (parity rule #1): upstream imports live Python
//! exception classes (`TypeError`, `OverflowError`, …) and
//! `rstackovf._StackOverflow`. The Rust port carries placeholder
//! [`ClassDef`]s bearing the qualified class names — the real
//! identity-hash set over live Python classes lands when Phase 5's
//! bookkeeper.py provides a ClassDesc registry. Until then this list
//! is consumed via name-equality against the annotator's
//! `HOST_ENV.lookup_exception_class` lookup.

use super::model::ClassDef;

/// RPython `exception.standardexceptions` (exception.py:4-7).
///
/// Names follow upstream — see flowspace's `HOST_ENV` for the live
/// exception classes that the annotator builds on.
pub fn standard_exceptions() -> Vec<ClassDef> {
    [
        "TypeError",
        "OverflowError",
        "ValueError",
        "ZeroDivisionError",
        "MemoryError",
        "IOError",
        "OSError",
        "StopIteration",
        "KeyError",
        "IndexError",
        "AssertionError",
        "RuntimeError",
        "UnicodeDecodeError",
        "UnicodeEncodeError",
        "NotImplementedError",
        // rpython.rlib.rstackovf._StackOverflow — HOST_ENV exposes
        // this as the RuntimeError subclass we registered in
        // flowspace::model::HostEnv::bootstrap.
        "StackOverflow",
    ]
    .iter()
    .map(|name| ClassDef::new(*name))
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_exceptions_has_expected_names() {
        let excs = standard_exceptions();
        assert_eq!(excs.len(), 16);
        // Spot-check a few entries so rename-away regressions fire.
        let names: Vec<&str> = excs.iter().map(|c| c.name.as_str()).collect();
        assert!(names.contains(&"TypeError"));
        assert!(names.contains(&"OverflowError"));
        assert!(names.contains(&"StackOverflow"));
    }
}
