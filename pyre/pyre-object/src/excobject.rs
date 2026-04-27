//! W_ExceptionObject — Python exception instance.
//!
//! Each exception carries a `kind` tag (mapping to PyErrorKind) and
//! a message string. The `ob_type` pointer is `EXCEPTION_TYPE` for all
//! exception instances; the `kind` field distinguishes the actual type.

use crate::pyobject::*;

pub static EXCEPTION_TYPE: PyType = crate::pyobject::new_pytype("exception");

/// Numeric tags for exception kinds — must stay in sync with PyErrorKind.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExcKind {
    BaseException = 0,
    Exception = 1,
    TypeError = 2,
    ValueError = 3,
    ZeroDivisionError = 4,
    NameError = 5,
    IndexError = 6,
    KeyError = 7,
    AttributeError = 8,
    RuntimeError = 9,
    StopIteration = 10,
    OverflowError = 11,
    ArithmeticError = 12,
    ImportError = 13,
    NotImplementedError = 14,
    AssertionError = 15,
    /// Raised by `_weakref` when a proxy is dereferenced after the
    /// referent has been collected — pypy/module/_weakref/interp__weakref.py:347
    /// `oefmt(space.w_ReferenceError, "weakly referenced object no longer exists")`.
    ReferenceError = 16,
    GeneratorExit = 17,
    RecursionError = 18,
    /// Base class for all operating-system errors
    /// (formerly IOError / WindowsError / EnvironmentError in Python 2).
    /// pypy/module/exceptions/interp_exceptions.py W_OSError.
    OSError = 19,
    /// Subclass of OSError raised when a file or directory is not found.
    FileNotFoundError = 20,
    /// Subclass of ValueError raised by codecs on invalid input.
    UnicodeDecodeError = 21,
    /// Subclass of ValueError raised by codecs on invalid input.
    UnicodeEncodeError = 22,
    /// Raised by sys.exit(). Subclass of BaseException, not Exception.
    SystemExit = 23,
    /// rpython/jit/metainterp/compile.py:1090 `memory_error = MemoryError()`
    /// — module-level singleton instance the JIT raises through
    /// `PropagateExceptionDescr.handle_fail` when a malloc helper
    /// returns NULL.  Subclass of Exception per
    /// pypy/module/exceptions/interp_exceptions.py.
    MemoryError = 24,
}

/// Layout: `[ob_type: *const PyType | kind: ExcKind | message: *mut String]`
#[repr(C)]
pub struct W_ExceptionObject {
    pub ob_header: PyObject,
    pub kind: ExcKind,
    pub message: *mut String,
}

pub const EXC_KIND_OFFSET: usize = std::mem::offset_of!(W_ExceptionObject, kind);
pub const EXC_MESSAGE_OFFSET: usize = std::mem::offset_of!(W_ExceptionObject, message);

/// GC type id assigned to `W_ExceptionObject` at JitDriver init time.
pub const W_EXCEPTION_GC_TYPE_ID: u32 = 31;

/// Fixed payload size (`framework.py:811`).
pub const W_EXCEPTION_OBJECT_SIZE: usize = std::mem::size_of::<W_ExceptionObject>();

impl crate::lltype::GcType for W_ExceptionObject {
    const TYPE_ID: u32 = W_EXCEPTION_GC_TYPE_ID;
    const SIZE: usize = W_EXCEPTION_OBJECT_SIZE;
}

/// Allocate a new exception object on the heap.
pub fn w_exception_new(kind: ExcKind, message: &str) -> PyObjectRef {
    let message = crate::lltype::malloc_raw(message.to_string());
    crate::lltype::malloc_typed(W_ExceptionObject {
        ob_header: PyObject {
            ob_type: &EXCEPTION_TYPE as *const PyType,
            w_class: get_instantiate(&EXCEPTION_TYPE),
        },
        kind,
        message,
    }) as PyObjectRef
}

/// `compile.py:1090` `memory_error = MemoryError()` parity — module-level
/// singleton instance the JIT raises through
/// `PropagateExceptionDescr.handle_fail` when a malloc helper returns
/// NULL.  RPython allocates the singleton at translation time; pyre
/// allocates lazily on first OOM (most workloads never trigger it).
///
/// Stored as `usize` because `PyObjectRef` is `*mut PyObject`, which is
/// neither `Send` nor `Sync` — `OnceLock<usize>` is the standard escape
/// hatch.  The `W_ExceptionObject` lives forever (`malloc_typed` is
/// `Box::into_raw` today; future GC integration must root it).
pub fn memory_error_singleton() -> PyObjectRef {
    static MEMORY_ERROR_SINGLETON: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MEMORY_ERROR_SINGLETON.get_or_init(|| w_exception_new(ExcKind::MemoryError, "") as usize)
        as PyObjectRef
}

/// Check if an object is an exception instance.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_exception(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &EXCEPTION_TYPE) }
}

/// Get the exception kind tag.
///
/// # Safety
/// `obj` must point to a valid `W_ExceptionObject`.
#[inline]
pub unsafe fn w_exception_get_kind(obj: PyObjectRef) -> ExcKind {
    unsafe { (*(obj as *const W_ExceptionObject)).kind }
}

/// Get the exception message.
///
/// # Safety
/// `obj` must point to a valid `W_ExceptionObject`.
#[inline]
pub unsafe fn w_exception_get_message(obj: PyObjectRef) -> &'static str {
    unsafe { &*(*(obj as *const W_ExceptionObject)).message }
}

/// Get the Python type name string for an ExcKind.
pub fn exc_kind_name(kind: ExcKind) -> &'static str {
    match kind {
        ExcKind::BaseException => "BaseException",
        ExcKind::Exception => "Exception",
        ExcKind::TypeError => "TypeError",
        ExcKind::ValueError => "ValueError",
        ExcKind::ZeroDivisionError => "ZeroDivisionError",
        ExcKind::NameError => "NameError",
        ExcKind::IndexError => "IndexError",
        ExcKind::KeyError => "KeyError",
        ExcKind::AttributeError => "AttributeError",
        ExcKind::RuntimeError => "RuntimeError",
        ExcKind::StopIteration => "StopIteration",
        ExcKind::OverflowError => "OverflowError",
        ExcKind::ArithmeticError => "ArithmeticError",
        ExcKind::ImportError => "ImportError",
        ExcKind::NotImplementedError => "NotImplementedError",
        ExcKind::AssertionError => "AssertionError",
        ExcKind::ReferenceError => "ReferenceError",
        ExcKind::GeneratorExit => "GeneratorExit",
        ExcKind::RecursionError => "RecursionError",
        ExcKind::OSError => "OSError",
        ExcKind::FileNotFoundError => "FileNotFoundError",
        ExcKind::UnicodeDecodeError => "UnicodeDecodeError",
        ExcKind::UnicodeEncodeError => "UnicodeEncodeError",
        ExcKind::SystemExit => "SystemExit",
        ExcKind::MemoryError => "MemoryError",
    }
}

/// Check if `exc_kind` matches `type_name`, considering Python's
/// exception hierarchy (e.g. ZeroDivisionError is-a ArithmeticError
/// is-a Exception is-a BaseException).
pub fn exc_kind_matches(kind: ExcKind, type_name: &str) -> bool {
    if type_name == "BaseException" {
        return true;
    }
    if type_name == "Exception" {
        return !matches!(
            kind,
            ExcKind::BaseException | ExcKind::GeneratorExit | ExcKind::SystemExit
        );
    }
    if type_name == "ArithmeticError" {
        return matches!(
            kind,
            ExcKind::ArithmeticError | ExcKind::ZeroDivisionError | ExcKind::OverflowError
        );
    }
    if type_name == "RuntimeError" {
        return matches!(kind, ExcKind::RuntimeError | ExcKind::RecursionError);
    }
    // OSError hierarchy — FileNotFoundError is-a OSError is-a Exception.
    // IOError / EnvironmentError are aliases for OSError in Python 3.
    if type_name == "OSError" || type_name == "IOError" || type_name == "EnvironmentError" {
        return matches!(kind, ExcKind::OSError | ExcKind::FileNotFoundError);
    }
    // Unicode errors are subclasses of ValueError.
    if type_name == "ValueError" {
        return matches!(
            kind,
            ExcKind::ValueError | ExcKind::UnicodeDecodeError | ExcKind::UnicodeEncodeError
        );
    }
    exc_kind_name(kind) == type_name
}

/// Convert a Python exception type name to an ExcKind.
pub fn exc_kind_from_name(name: &str) -> Option<ExcKind> {
    match name {
        "BaseException" => Some(ExcKind::BaseException),
        "Exception" => Some(ExcKind::Exception),
        "TypeError" => Some(ExcKind::TypeError),
        "ValueError" => Some(ExcKind::ValueError),
        "ZeroDivisionError" => Some(ExcKind::ZeroDivisionError),
        "NameError" => Some(ExcKind::NameError),
        "IndexError" => Some(ExcKind::IndexError),
        "KeyError" => Some(ExcKind::KeyError),
        "AttributeError" => Some(ExcKind::AttributeError),
        "RuntimeError" => Some(ExcKind::RuntimeError),
        "StopIteration" => Some(ExcKind::StopIteration),
        "OverflowError" => Some(ExcKind::OverflowError),
        "ArithmeticError" => Some(ExcKind::ArithmeticError),
        "ImportError" => Some(ExcKind::ImportError),
        "NotImplementedError" => Some(ExcKind::NotImplementedError),
        "AssertionError" => Some(ExcKind::AssertionError),
        "ReferenceError" => Some(ExcKind::ReferenceError),
        "GeneratorExit" => Some(ExcKind::GeneratorExit),
        "RecursionError" => Some(ExcKind::RecursionError),
        "OSError" | "IOError" | "EnvironmentError" => Some(ExcKind::OSError),
        "FileNotFoundError" => Some(ExcKind::FileNotFoundError),
        "UnicodeDecodeError" => Some(ExcKind::UnicodeDecodeError),
        "UnicodeEncodeError" => Some(ExcKind::UnicodeEncodeError),
        "SystemExit" => Some(ExcKind::SystemExit),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exception_create_and_read() {
        let obj = w_exception_new(ExcKind::ValueError, "bad value");
        unsafe {
            assert!(is_exception(obj));
            assert_eq!(w_exception_get_kind(obj), ExcKind::ValueError);
            assert_eq!(w_exception_get_message(obj), "bad value");
        }
    }

    #[test]
    fn test_exc_kind_matches_hierarchy() {
        assert!(exc_kind_matches(
            ExcKind::ZeroDivisionError,
            "ZeroDivisionError"
        ));
        assert!(exc_kind_matches(
            ExcKind::ZeroDivisionError,
            "ArithmeticError"
        ));
        assert!(exc_kind_matches(ExcKind::ZeroDivisionError, "Exception"));
        assert!(exc_kind_matches(
            ExcKind::ZeroDivisionError,
            "BaseException"
        ));
        assert!(!exc_kind_matches(ExcKind::ZeroDivisionError, "ValueError"));
    }

    #[test]
    fn test_exc_kind_from_name_roundtrip() {
        for kind in [
            ExcKind::TypeError,
            ExcKind::ValueError,
            ExcKind::ZeroDivisionError,
        ] {
            let name = exc_kind_name(kind);
            assert_eq!(exc_kind_from_name(name), Some(kind));
        }
    }

    #[test]
    fn memory_error_singleton_is_idempotent_and_typed() {
        let a = memory_error_singleton();
        let b = memory_error_singleton();
        assert_eq!(a as usize, b as usize, "singleton must be stable");
        unsafe {
            assert!(is_exception(a));
            assert_eq!(w_exception_get_kind(a), ExcKind::MemoryError);
            assert_eq!(w_exception_get_message(a), "");
        }
    }

    #[test]
    fn w_exception_gc_type_id_matches_descr() {
        assert_eq!(W_EXCEPTION_GC_TYPE_ID, 31);
        assert_eq!(
            <W_ExceptionObject as crate::lltype::GcType>::TYPE_ID,
            W_EXCEPTION_GC_TYPE_ID
        );
        assert_eq!(
            <W_ExceptionObject as crate::lltype::GcType>::SIZE,
            W_EXCEPTION_OBJECT_SIZE
        );
    }
}
