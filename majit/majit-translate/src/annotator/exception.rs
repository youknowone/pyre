//! Implicitly-raisable exception set used by the annotator.
//!
//! RPython upstream: `rpython/annotator/exception.py` (7 LOC).
//!
//! Upstream stores a set of live exception classes; callers that need
//! `ClassDef`s resolve them through `bookkeeper.getuniqueclassdef(cls)`.
//! The Rust port keeps that shape: [`standard_exception_classes`]
//! returns the class objects, and [`standard_exception_classdefs`]
//! threads them through a caller-provided [`Bookkeeper`] so each typer
//! owns the classdef identity it reasons about (no cross-typer stale
//! `ClassDef.repr` slot).

use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

use super::bookkeeper::Bookkeeper;
use super::classdesc::ClassDef;
use super::model::AnnotatorError;
use crate::flowspace::model::{HOST_ENV, HostObject};

const STANDARD_EXCEPTION_NAMES: [&str; 16] = [
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
    "_StackOverflow",
];

#[derive(Clone)]
struct StandardExceptionObjects {
    memory_error: HostObject,
    os_error: HostObject,
    unicode_error: HostObject,
    unicode_decode_error: HostObject,
    unicode_encode_error: HostObject,
}

thread_local! {
    static STANDARD_EXCEPTION_OBJECTS: StandardExceptionObjects =
        StandardExceptionObjects::new();
}

impl StandardExceptionObjects {
    fn new() -> Self {
        let exception = HOST_ENV
            .lookup_exception_class("Exception")
            .expect("HOST_ENV missing builtin Exception");
        let value_error = HOST_ENV
            .lookup_exception_class("ValueError")
            .expect("HOST_ENV missing builtin ValueError");
        let os_error = HOST_ENV
            .lookup_exception_class("OSError")
            .unwrap_or_else(|| HostObject::new_class("OSError", vec![exception.clone()]));
        let unicode_error = HOST_ENV
            .lookup_exception_class("UnicodeError")
            .unwrap_or_else(|| HostObject::new_class("UnicodeError", vec![value_error]));

        StandardExceptionObjects {
            memory_error: HOST_ENV
                .lookup_exception_class("MemoryError")
                .unwrap_or_else(|| HostObject::new_class("MemoryError", vec![exception.clone()])),
            os_error,
            unicode_decode_error: HOST_ENV
                .lookup_exception_class("UnicodeDecodeError")
                .unwrap_or_else(|| {
                    HostObject::new_class("UnicodeDecodeError", vec![unicode_error.clone()])
                }),
            unicode_encode_error: HOST_ENV
                .lookup_exception_class("UnicodeEncodeError")
                .unwrap_or_else(|| {
                    HostObject::new_class("UnicodeEncodeError", vec![unicode_error.clone()])
                }),
            unicode_error,
        }
    }

    fn resolve(&self, name: &'static str) -> HostObject {
        if let Some(cls) = HOST_ENV.lookup_exception_class(name) {
            return cls;
        }
        match name {
            "MemoryError" => self.memory_error.clone(),
            // Python 3 / modern PyPy expose IOError as an OSError alias.
            "IOError" => HOST_ENV
                .lookup_exception_class("IOError")
                .unwrap_or_else(|| self.os_error.clone()),
            "OSError" => self.os_error.clone(),
            "UnicodeDecodeError" => self.unicode_decode_error.clone(),
            "UnicodeEncodeError" => self.unicode_encode_error.clone(),
            "UnicodeError" => self.unicode_error.clone(),
            _ => panic!("HOST_ENV missing standard exception class {:?}", name),
        }
    }
}

/// RPython `exception.standardexceptions` data shape (exception.py:4-7).
///
/// Upstream stores live exception class objects in a set. The Rust port
/// keeps the same "classes first" surface and resolves them through
/// `HOST_ENV` bootstrap.
pub(crate) fn standard_exception_classes() -> Vec<HostObject> {
    STANDARD_EXCEPTION_OBJECTS.with(|objects| {
        let mut seen = HashSet::new();
        let mut classes = Vec::new();
        for name in STANDARD_EXCEPTION_NAMES {
            let cls = objects.resolve(name);
            if seen.insert(cls.identity_id()) {
                classes.push(cls);
            }
        }
        classes
    })
}

/// Upstream caller pattern:
///
/// ```python
/// for cls in self.standardexceptions:
///     classdef = bk.getuniqueclassdef(cls)
/// ```
pub(crate) fn standard_exception_classdefs(
    bookkeeper: &Rc<Bookkeeper>,
) -> Result<Vec<Rc<RefCell<ClassDef>>>, AnnotatorError> {
    let mut classdefs = Vec::with_capacity(STANDARD_EXCEPTION_NAMES.len());
    for cls in standard_exception_classes() {
        classdefs.push(bookkeeper.getuniqueclassdef(&cls)?);
    }
    Ok(classdefs)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole resolved sequence, in order.
    ///
    /// These names are **measured**, not derived from
    /// [`STANDARD_EXCEPTION_NAMES`]. `standard_exception_classes` transforms
    /// that constant twice — `resolve` maps a name to whatever `HOST_ENV`
    /// bootstraps, and the `identity_id()` dedup then drops repeats — so the
    /// output is not the input, and two entries differ:
    ///
    /// - **16 names in, 15 classes out.** `IOError` has no distinct host class;
    ///   it resolves to the `OSError` object and is deduped. It is the `IOError`
    ///   slot that survives (it comes first), so the list has one `OSError`
    ///   entry, not two, and no `IOError` entry at all.
    /// - **`_StackOverflow` reads back as `StackOverflow`.** The bootstrap
    ///   lookup key keeps the underscore; the qualname does not.
    ///
    /// The order is a property of the producer, not an accident: `classes` is a
    /// `Vec` pushed in `STANDARD_EXCEPTION_NAMES` order, and the `HashSet` is
    /// consulted for membership only. Pinning it ordered therefore also pins
    /// that a reorder of the constant is a visible change.
    const RESOLVED_EXCEPTION_NAMES: [&str; 15] = [
        "TypeError",
        "OverflowError",
        "ValueError",
        "ZeroDivisionError",
        "MemoryError",
        "OSError",
        "StopIteration",
        "KeyError",
        "IndexError",
        "AssertionError",
        "RuntimeError",
        "UnicodeDecodeError",
        "UnicodeEncodeError",
        "NotImplementedError",
        "StackOverflow",
    ];

    #[test]
    fn standard_exception_classes_have_expected_names() {
        let excs = standard_exception_classes();
        let names: Vec<String> = excs.iter().map(|c| c.qualname().to_string()).collect();
        assert_eq!(names, RESOLVED_EXCEPTION_NAMES);
    }

    /// The dedup is the reason the set is 15 and not 16, so pin it at its cause
    /// rather than only at the length: `IOError` and `OSError` must still name
    /// one object.
    ///
    /// Without this, a bootstrap that grew a distinct `IOError` class would fail
    /// `standard_exception_classes_have_expected_names` with a diff that looks
    /// like a naming change, and the obvious repair — add `"IOError"` to the
    /// expected list — would restore green while silently changing what the
    /// annotator treats as one exception.
    #[test]
    fn io_error_and_os_error_resolve_to_one_class() {
        STANDARD_EXCEPTION_OBJECTS.with(|objects| {
            assert_eq!(
                objects.resolve("IOError").identity_id(),
                objects.resolve("OSError").identity_id(),
                "IOError is no longer an OSError alias, so the standard set is \
                 16 classes and the dedup no longer fires"
            );
        });
    }

    /// `_StackOverflow` is bootstrapped under a name the qualname does not
    /// carry, so a name-keyed check alone cannot show the two spellings are one
    /// object.
    #[test]
    fn stack_overflow_entry_is_the_bootstrapped_builtin() {
        let excs = standard_exception_classes();
        let stack_overflow = HOST_ENV
            .lookup_builtin("_StackOverflow")
            .expect("HOST_ENV _StackOverflow bootstrap");
        assert_eq!(stack_overflow.qualname(), "StackOverflow");
        assert!(
            excs.iter()
                .any(|cls| cls.identity_id() == stack_overflow.identity_id())
        );
    }

    #[test]
    fn standard_exception_classdefs_use_caller_bookkeeper_cache() {
        let bk = Rc::new(Bookkeeper::new());
        let first = standard_exception_classdefs(&bk).expect("standard exception classdefs");
        let second = standard_exception_classdefs(&bk).expect("cached classdefs");
        assert_eq!(first.len(), second.len());
        for (lhs, rhs) in first.iter().zip(second.iter()) {
            assert!(Rc::ptr_eq(lhs, rhs));
        }
    }

    #[test]
    fn standard_exception_classdefs_reset_per_bookkeeper() {
        let bk1 = Rc::new(Bookkeeper::new());
        let bk2 = Rc::new(Bookkeeper::new());
        let first = standard_exception_classdefs(&bk1).expect("bk1 classdefs");
        let second = standard_exception_classdefs(&bk2).expect("bk2 classdefs");
        assert_eq!(first.len(), second.len());
        // Classdefs must be distinct per bookkeeper so that `ClassDef.repr`
        // slots cannot leak across typers.
        for (lhs, rhs) in first.iter().zip(second.iter()) {
            assert!(!Rc::ptr_eq(lhs, rhs));
        }
    }
}
