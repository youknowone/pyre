//! Implicitly-raisable exception set used by the annotator.
//!
//! RPython upstream: `rpython/annotator/exception.py` (7 LOC).
//!
//! ClassDefs surface as `Rc<RefCell<classdesc::ClassDef>>` so identity
//! equality (`Rc::ptr_eq`) matches upstream's Python class identity.
//!
//! Rust adaptation (parity rule #1): upstream imports live Python
//! exception classes (`TypeError`, `OverflowError`, …) and
//! `rstackovf._StackOverflow`. The Rust port uses
//! [`classdesc::ClassDef::new_standalone`] to build fresh classdef
//! shells under the upstream qualnames; once a real ClassDesc registry
//! is wired into the builtin-module port these callers should route
//! through `bookkeeper.getdesc(cls).getuniqueclassdef()` instead.

use std::cell::RefCell;
use std::rc::Rc;

use super::classdesc::ClassDef;

/// RPython `exception.standardexceptions` (exception.py:4-7).
///
/// Names follow upstream — see flowspace's `HOST_ENV` for the live
/// exception classes that the annotator builds on.
pub fn standard_exceptions() -> Vec<Rc<RefCell<ClassDef>>> {
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
        // rpython.rlib.rstackovf._StackOverflow — underscore prefix
        // matches upstream exception.py:7 exactly. HOST_ENV exposes
        // this as the RuntimeError subclass registered in
        // flowspace::model::HostEnv::bootstrap.
        "_StackOverflow",
    ]
    .iter()
    .map(|name| ClassDef::new_standalone(*name, None))
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
        let names: Vec<String> = excs.iter().map(|c| c.borrow().name.clone()).collect();
        assert!(names.iter().any(|n| n == "TypeError"));
        assert!(names.iter().any(|n| n == "OverflowError"));
        assert!(names.iter().any(|n| n == "_StackOverflow"));
    }
}
