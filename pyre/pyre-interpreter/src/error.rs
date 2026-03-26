use pyre_object::PyObjectRef;
use pyre_object::excobject::{ExcKind, exc_kind_name, w_exception_new};

/// Result type for Python operations.
pub type PyResult = Result<PyObjectRef, PyError>;

/// Python exception.
#[derive(Debug, Clone)]
pub struct PyError {
    pub kind: PyErrorKind,
    pub message: String,
    /// Cached W_ExceptionObject pointer — reused by to_exc_object()
    /// to avoid re-allocating an exception object that already exists.
    pub exc_object: PyObjectRef,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PyErrorKind {
    TypeError,
    ValueError,
    ZeroDivisionError,
    NameError,
    IndexError,
    KeyError,
    AttributeError,
    RuntimeError,
    StopIteration,
    OverflowError,
    ArithmeticError,
    ImportError,
    NotImplementedError,
    AssertionError,
    /// Internal: RETURN_GENERATOR unwind signal (not a real exception).
    /// Carries the generator PyObjectRef as message.
    GeneratorReturn,
}

impl PyError {
    pub fn new(kind: PyErrorKind, message: impl Into<String>) -> Self {
        PyError {
            kind,
            message: message.into(),
            exc_object: std::ptr::null_mut(),
        }
    }

    pub fn type_error(msg: impl Into<String>) -> Self {
        PyError {
            kind: PyErrorKind::TypeError,
            message: msg.into(),
            exc_object: std::ptr::null_mut(),
        }
    }

    pub fn value_error(msg: impl Into<String>) -> Self {
        PyError {
            kind: PyErrorKind::ValueError,
            message: msg.into(),
            exc_object: std::ptr::null_mut(),
        }
    }

    pub fn zero_division(msg: impl Into<String>) -> Self {
        PyError {
            kind: PyErrorKind::ZeroDivisionError,
            message: msg.into(),
            exc_object: std::ptr::null_mut(),
        }
    }

    pub fn runtime_error(msg: impl Into<String>) -> Self {
        PyError {
            kind: PyErrorKind::RuntimeError,
            message: msg.into(),
            exc_object: std::ptr::null_mut(),
        }
    }

    pub fn stop_iteration() -> Self {
        PyError {
            kind: PyErrorKind::StopIteration,
            message: String::new(),
            exc_object: std::ptr::null_mut(),
        }
    }

    /// Convert to a W_ExceptionObject for pushing onto the value stack.
    /// Reuses the cached object from from_exc_object() if available.
    pub fn to_exc_object(&self) -> PyObjectRef {
        if !self.exc_object.is_null() {
            return self.exc_object;
        }
        w_exception_new(self.to_exc_kind(), &self.message)
    }

    fn to_exc_kind(&self) -> ExcKind {
        match self.kind {
            PyErrorKind::TypeError => ExcKind::TypeError,
            PyErrorKind::ValueError => ExcKind::ValueError,
            PyErrorKind::ZeroDivisionError => ExcKind::ZeroDivisionError,
            PyErrorKind::NameError => ExcKind::NameError,
            PyErrorKind::IndexError => ExcKind::IndexError,
            PyErrorKind::KeyError => ExcKind::KeyError,
            PyErrorKind::AttributeError => ExcKind::AttributeError,
            PyErrorKind::RuntimeError => ExcKind::RuntimeError,
            PyErrorKind::StopIteration => ExcKind::StopIteration,
            PyErrorKind::OverflowError => ExcKind::OverflowError,
            PyErrorKind::ArithmeticError => ExcKind::ArithmeticError,
            PyErrorKind::ImportError => ExcKind::ImportError,
            PyErrorKind::NotImplementedError => ExcKind::NotImplementedError,
            PyErrorKind::AssertionError => ExcKind::AssertionError,
            PyErrorKind::GeneratorReturn => ExcKind::RuntimeError,
        }
    }

    /// Create a PyError from a W_ExceptionObject.
    ///
    /// # Safety
    /// `obj` must point to a valid `W_ExceptionObject`.
    pub unsafe fn from_exc_object(obj: PyObjectRef) -> Self {
        let kind = pyre_object::excobject::w_exception_get_kind(obj);
        let message = pyre_object::excobject::w_exception_get_message(obj).to_string();
        PyError {
            kind: Self::kind_from_exc(kind),
            message,
            exc_object: obj,
        }
    }

    fn kind_from_exc(kind: ExcKind) -> PyErrorKind {
        match kind {
            ExcKind::TypeError => PyErrorKind::TypeError,
            ExcKind::ValueError => PyErrorKind::ValueError,
            ExcKind::ZeroDivisionError => PyErrorKind::ZeroDivisionError,
            ExcKind::NameError => PyErrorKind::NameError,
            ExcKind::IndexError => PyErrorKind::IndexError,
            ExcKind::KeyError => PyErrorKind::KeyError,
            ExcKind::AttributeError => PyErrorKind::AttributeError,
            ExcKind::RuntimeError => PyErrorKind::RuntimeError,
            ExcKind::StopIteration => PyErrorKind::StopIteration,
            ExcKind::OverflowError => PyErrorKind::OverflowError,
            ExcKind::ArithmeticError => PyErrorKind::ArithmeticError,
            ExcKind::ImportError => PyErrorKind::ImportError,
            ExcKind::NotImplementedError => PyErrorKind::NotImplementedError,
            ExcKind::AssertionError => PyErrorKind::AssertionError,
            ExcKind::BaseException | ExcKind::Exception => PyErrorKind::RuntimeError,
        }
    }
}

impl std::fmt::Display for PyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = exc_kind_name(self.to_exc_kind());
        write!(f, "{}: {}", name, self.message)
    }
}
