//! RPython `rpython/rtyper/error.py` — TyperError hierarchy.
//!
//! Upstream defines:
//!
//! ```python
//! class TyperError(Exception):
//!     def __str__(self):
//!         result = Exception.__str__(self)
//!         if hasattr(self, 'where'):
//!             result += '\n.. %s\n.. %r\n.. %r' % self.where
//!         return result
//!
//! class MissingRTypeOperation(TyperError):
//!     pass
//! ```
//!
//! Plus `rpython/rtyper/rmodel.py:397 BrokenReprTyperError(TyperError)`
//! for the Repr.setup() BROKEN state. Rust collapses the three-class
//! hierarchy into an enum so call sites can pattern-match on kind
//! without downcasts; the `where` annotation (`rtyper.py:252
//! gottypererror`) is carried on every variant as an optional trio
//! matching upstream's (stage, block, op) tuple.

use std::fmt;

/// RPython `TyperError(Exception)` hierarchy (error.py:1 + rmodel.py:397).
///
/// The three upstream subclasses correspond to enum variants. `where`
/// is a structured annotation upstream stores on the exception via
/// `__dict__`-style attach after construction (see `rtyper.py:252
/// gottypererror`); mirror it as an `Option<TyperWhere>` field on
/// every variant.
#[derive(Debug, Clone)]
pub enum TyperError {
    /// RPython `TyperError("...")` — generic diagnostic. The plain
    /// case covers ~95% of upstream raises (e.g. `rmodel.py:117 /
    /// :123 / :188 / :193 / :204`).
    Message {
        text: String,
        where_info: Option<TyperWhere>,
    },
    /// RPython `class MissingRTypeOperation(TyperError)` (error.py:8).
    /// Raised by `Repr` methods that have no implementation for the
    /// operation name (e.g. `rmodel.py:174 rtype_bltn_list`,
    /// `rmodel.py:178 rtype_unichr`, `rmodel.py:234 make_iterator_repr`).
    /// `Repr.rtype_bool` catches this specifically at rmodel.py:201-205.
    MissingRTypeOperation {
        text: String,
        where_info: Option<TyperWhere>,
    },
    /// RPython `class BrokenReprTyperError(TyperError)` (rmodel.py:397).
    /// Raised by `Repr.setup()` when the target Repr is already in
    /// `setupstate.BROKEN` state (rmodel.py:42-44).
    BrokenRepr {
        text: String,
        where_info: Option<TyperWhere>,
    },
}

/// RPython `self.where = (stage, block, op)` (rtyper.py:252
/// gottypererror). Structured trio for post-mortem diagnostic
/// formatting.
#[derive(Debug, Clone)]
pub struct TyperWhere {
    pub stage: String,
    /// Block descriptor — upstream passes the `Block` object; pyre
    /// stores the human-readable `Debug` rendering to avoid holding a
    /// reference that crosses the unwind boundary.
    pub block: String,
    /// Operation descriptor. Same serialization rationale.
    pub op: String,
}

impl TyperError {
    /// RPython `TyperError(msg)` constructor.
    pub fn message<S: Into<String>>(text: S) -> Self {
        TyperError::Message {
            text: text.into(),
            where_info: None,
        }
    }

    /// RPython `MissingRTypeOperation(msg)` constructor.
    pub fn missing_rtype_operation<S: Into<String>>(text: S) -> Self {
        TyperError::MissingRTypeOperation {
            text: text.into(),
            where_info: None,
        }
    }

    /// RPython `BrokenReprTyperError(msg)` constructor.
    pub fn broken_repr<S: Into<String>>(text: S) -> Self {
        TyperError::BrokenRepr {
            text: text.into(),
            where_info: None,
        }
    }

    /// Returns `true` for `MissingRTypeOperation` instances. Mirrors
    /// `isinstance(e, MissingRTypeOperation)` in `rmodel.py:202` (the
    /// rtype_bool fallback).
    pub fn is_missing_rtype_operation(&self) -> bool {
        matches!(self, TyperError::MissingRTypeOperation { .. })
    }

    /// Returns `true` for `BrokenReprTyperError` instances.
    pub fn is_broken_repr(&self) -> bool {
        matches!(self, TyperError::BrokenRepr { .. })
    }

    /// Attach (or overwrite) the `where` annotation. Mirrors upstream
    /// `e.where = (stage, block, op)` assignment.
    pub fn with_where(mut self, where_info: TyperWhere) -> Self {
        match &mut self {
            TyperError::Message { where_info: w, .. }
            | TyperError::MissingRTypeOperation { where_info: w, .. }
            | TyperError::BrokenRepr { where_info: w, .. } => {
                *w = Some(where_info);
            }
        }
        self
    }

    fn parts(&self) -> (&str, &str, Option<&TyperWhere>) {
        match self {
            TyperError::Message { text, where_info } => ("TyperError", text, where_info.as_ref()),
            TyperError::MissingRTypeOperation { text, where_info } => {
                ("MissingRTypeOperation", text, where_info.as_ref())
            }
            TyperError::BrokenRepr { text, where_info } => {
                ("BrokenReprTyperError", text, where_info.as_ref())
            }
        }
    }
}

impl fmt::Display for TyperError {
    /// RPython `TyperError.__str__`:
    ///
    /// ```python
    /// result = Exception.__str__(self)
    /// if hasattr(self, 'where'):
    ///     result += '\n.. %s\n.. %r\n.. %r' % self.where
    /// return result
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (class_name, text, where_info) = self.parts();
        write!(f, "{class_name}: {text}")?;
        if let Some(w) = where_info {
            write!(f, "\n.. {}\n.. {}\n.. {}", w.stage, w.block, w.op)?;
        }
        Ok(())
    }
}

impl std::error::Error for TyperError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_variant_uses_class_name_prefix() {
        // upstream `TyperError.__str__` returns the Exception base's
        // formatted message; pyre prefixes the class name so mixed
        // logs disambiguate kinds at a glance.
        let e = TyperError::message("bad cast");
        assert_eq!(e.to_string(), "TyperError: bad cast");
        assert!(!e.is_missing_rtype_operation());
        assert!(!e.is_broken_repr());
    }

    #[test]
    fn missing_rtype_operation_variant_is_detected_by_helper() {
        // rmodel.py:202 catches MissingRTypeOperation by class — pyre
        // callers check `.is_missing_rtype_operation()`.
        let e = TyperError::missing_rtype_operation("no iter() for Void");
        assert!(e.is_missing_rtype_operation());
        assert_eq!(e.to_string(), "MissingRTypeOperation: no iter() for Void");
    }

    #[test]
    fn broken_repr_variant_is_detected_by_helper() {
        // rmodel.py:42-44 raises BrokenReprTyperError from setup();
        // callers distinguish via isinstance. Pyre mirrors the
        // detection via `.is_broken_repr()`.
        let e = TyperError::broken_repr("cannot setup already failed Repr: <FooRepr>");
        assert!(e.is_broken_repr());
        assert!(e.to_string().starts_with("BrokenReprTyperError: "));
    }

    #[test]
    fn where_suffix_appended_on_display() {
        // rtyper.py:252 gottypererror assigns `e.where = (stage,
        // block, op)` before re-raising. Lock in the formatter so
        // post-mortem readers see the upstream 3-line suffix.
        let e = TyperError::message("bad cast").with_where(TyperWhere {
            stage: "block-entry".to_string(),
            block: "<block @123>".to_string(),
            op: "v0 = simple_call(v1)".to_string(),
        });
        let s = e.to_string();
        assert!(s.contains("\n.. block-entry"));
        assert!(s.contains("\n.. <block @123>"));
        assert!(s.contains("\n.. v0 = simple_call(v1)"));
    }
}
