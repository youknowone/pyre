use pyre_object::PyObjectRef;
use pyre_object::excobject::{ExcKind, exc_kind_name, w_exception_new};
use std::io::Write;

#[derive(Debug, Clone)]
pub struct OperationError {
    pub w_type: PyObjectRef,
    pub w_value: PyObjectRef,
    pub _application_traceback: Option<PyObjectRef>,
}

impl OperationError {
    pub fn new(w_type: PyObjectRef, w_value: PyObjectRef) -> Self {
        Self {
            w_type,
            w_value,
            _application_traceback: None,
        }
    }

    pub fn get_w_value(&self, _space: PyObjectRef) -> PyObjectRef {
        let _ = _space;
        self.w_value
    }

    pub fn match_(&self, _space: PyObjectRef, _check: PyObjectRef) -> bool {
        false
    }
}

impl From<OperationError> for PyError {
    fn from(value: OperationError) -> Self {
        let message = if value.w_value.is_null() {
            String::new()
        } else {
            "operation error".to_string()
        };
        PyError {
            kind: PyErrorKind::RuntimeError,
            message,
            exc_object: value.w_value,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ClearedOpErr;

#[derive(Debug, Clone)]
pub struct OpErrFmtNoArgs;

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

    pub fn overflow_error(msg: impl Into<String>) -> Self {
        PyError {
            kind: PyErrorKind::OverflowError,
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

pub fn get_cleared_operation_error(_space: PyObjectRef) -> OperationError {
    let _ = _space;
    OperationError::new(std::ptr::null_mut(), std::ptr::null_mut())
}

pub fn get_converted_unexpected_exception(
    _space: PyObjectRef,
    _error: &dyn std::error::Error,
) -> OperationError {
    let _ = (_space, _error);
    OperationError::new(std::ptr::null_mut(), std::ptr::null_mut())
}

pub fn decompose_valuefmt(valuefmt: &str) -> (Vec<String>, Vec<String>) {
    let mut strings = Vec::new();
    let mut formats = Vec::new();
    let mut current = String::new();

    let mut iter = valuefmt.chars().peekable();
    while let Some(ch) = iter.next() {
        if ch == '%' {
            if let Some('%') = iter.peek() {
                let _ = iter.next();
                current.push('%');
                continue;
            }
            strings.push(std::mem::take(&mut current));
            if let Some(spec) = iter.next() {
                formats.push(spec.to_string());
            }
        } else {
            current.push(ch);
        }
    }

    if !current.is_empty() {
        strings.push(current);
    }

    (strings, formats)
}

pub fn get_operrcls2(valuefmt: &str) -> (PyObjectRef, Vec<String>) {
    let (strings, _formats) = decompose_valuefmt(valuefmt);
    (std::ptr::null_mut(), strings)
}

pub fn get_operr_class(valuefmt: &str) -> (PyObjectRef, Vec<String>) {
    get_operrcls2(valuefmt)
}

pub fn oefmt(w_type: PyObjectRef, valuefmt: &str, _args: impl std::fmt::Display) -> OperationError {
    let _ = valuefmt;
    let _ = format!("{}", _args);
    OperationError::new(w_type, std::ptr::null_mut())
}

pub fn debug_print(text: &str, file: Option<&mut dyn Write>, _newline: bool) {
    if let Some(file) = file {
        let _ = file.write_all(text.as_bytes());
    }
}

pub fn exception_from_errno(
    _space: PyObjectRef,
    w_type: PyObjectRef,
    _errno: i32,
) -> OperationError {
    let _ = _space;
    OperationError::new(w_type, std::ptr::null_mut())
}

pub fn exception_from_saved_errno(_space: PyObjectRef, w_type: PyObjectRef) -> OperationError {
    let _ = _space;
    OperationError::new(w_type, std::ptr::null_mut())
}

pub fn new_exception_class(
    _space: PyObjectRef,
    _name: &str,
    _bases: Option<PyObjectRef>,
    _dict: Option<PyObjectRef>,
) -> PyObjectRef {
    let _ = (_space, _name, _bases, _dict);
    std::ptr::null_mut()
}

pub fn wrap_oserror2(
    _space: PyObjectRef,
    _error: &dyn std::error::Error,
    _filename: Option<PyObjectRef>,
    _exception_class: Option<PyObjectRef>,
) -> OperationError {
    let _ = (_filename, _exception_class, _error);
    let _ = _space;
    OperationError::new(std::ptr::null_mut(), std::ptr::null_mut())
}

pub fn wrap_oserror(
    space: PyObjectRef,
    error: &dyn std::error::Error,
    _filename: Option<&str>,
    w_exception_class: Option<PyObjectRef>,
) -> OperationError {
    let _ = _filename;
    wrap_oserror2(space, error, None, w_exception_class)
}
