//! W_BaseException — Python exception instance.
//!
//! Each exception carries a `kind` tag (mapping to PyErrorKind) and a
//! message string. `ob_type` is the per-subclass `PyType` static
//! (`EXC_VALUE_ERROR_TYPE`, `EXC_TYPE_ERROR_TYPE`, …) registered with
//! the appropriate parent in `all_foreign_pytypes`, so backend
//! `GuardClass` at `OB_TYPE_OFFSET` discriminates exception
//! subclasses without any IR/backend change — matching RPython
//! `OBJECT.typeptr = specific class` (`rclass.py:167-174`).
//! `EXCEPTION_TYPE` is the BaseException root that every per-kind
//! `PyType` chains up to; `is_exception` is an `ll_isinstance` against
//! it via the assigned `subclassrange_{min,max}`.

use crate::pyobject::*;
use rustpython_wtf8::Wtf8;

pub static EXCEPTION_TYPE: PyType = crate::pyobject::new_pytype("BaseException");
pub static EXC_EXCEPTION_TYPE: PyType = crate::pyobject::new_pytype("Exception");
pub static EXC_ARITHMETIC_ERROR_TYPE: PyType = crate::pyobject::new_pytype("ArithmeticError");
pub static EXC_OVERFLOW_ERROR_TYPE: PyType = crate::pyobject::new_pytype("OverflowError");
pub static EXC_ZERO_DIVISION_ERROR_TYPE: PyType = crate::pyobject::new_pytype("ZeroDivisionError");
pub static EXC_TYPE_ERROR_TYPE: PyType = crate::pyobject::new_pytype("TypeError");
pub static EXC_VALUE_ERROR_TYPE: PyType = crate::pyobject::new_pytype("ValueError");
pub static EXC_NAME_ERROR_TYPE: PyType = crate::pyobject::new_pytype("NameError");
pub static EXC_UNBOUND_LOCAL_ERROR_TYPE: PyType = crate::pyobject::new_pytype("UnboundLocalError");
pub static EXC_INDEX_ERROR_TYPE: PyType = crate::pyobject::new_pytype("IndexError");
pub static EXC_KEY_ERROR_TYPE: PyType = crate::pyobject::new_pytype("KeyError");
pub static EXC_ATTRIBUTE_ERROR_TYPE: PyType = crate::pyobject::new_pytype("AttributeError");
pub static EXC_RUNTIME_ERROR_TYPE: PyType = crate::pyobject::new_pytype("RuntimeError");
pub static EXC_STOP_ITERATION_TYPE: PyType = crate::pyobject::new_pytype("StopIteration");
pub static EXC_STOP_ASYNC_ITERATION_TYPE: PyType =
    crate::pyobject::new_pytype("StopAsyncIteration");
pub static EXC_IMPORT_ERROR_TYPE: PyType = crate::pyobject::new_pytype("ImportError");
pub static EXC_MODULE_NOT_FOUND_ERROR_TYPE: PyType =
    crate::pyobject::new_pytype("ModuleNotFoundError");
pub static EXC_NOT_IMPLEMENTED_ERROR_TYPE: PyType =
    crate::pyobject::new_pytype("NotImplementedError");
pub static EXC_ASSERTION_ERROR_TYPE: PyType = crate::pyobject::new_pytype("AssertionError");
pub static EXC_REFERENCE_ERROR_TYPE: PyType = crate::pyobject::new_pytype("ReferenceError");
pub static EXC_GENERATOR_EXIT_TYPE: PyType = crate::pyobject::new_pytype("GeneratorExit");
pub static EXC_RECURSION_ERROR_TYPE: PyType = crate::pyobject::new_pytype("RecursionError");
pub static EXC_OS_ERROR_TYPE: PyType = crate::pyobject::new_pytype("OSError");
pub static EXC_FILE_NOT_FOUND_ERROR_TYPE: PyType = crate::pyobject::new_pytype("FileNotFoundError");
pub static EXC_UNICODE_DECODE_ERROR_TYPE: PyType =
    crate::pyobject::new_pytype("UnicodeDecodeError");
pub static EXC_UNICODE_ENCODE_ERROR_TYPE: PyType =
    crate::pyobject::new_pytype("UnicodeEncodeError");
/// PyPy `pypy/module/exceptions/interp_exceptions.py:426
/// W_UnicodeTranslateError = _new_exception('UnicodeTranslateError',
/// W_UnicodeError, ...)` — subclass of UnicodeError.  Identity-only
/// port (dedicated PyType + ExcKind for isinstance / `ob_type`
/// discrimination); the 4-arg `(object, start, end, reason)` init
/// signature and custom `__str__` formatting on the
/// W_UnicodeTranslateError class itself are not yet ported.  See the
/// `ExcKind::UnicodeTranslateError` doc for the broader identity-only
/// pattern across pyre's exception subclasses.
pub static EXC_UNICODE_TRANSLATE_ERROR_TYPE: PyType =
    crate::pyobject::new_pytype("UnicodeTranslateError");
pub static EXC_SYSTEM_EXIT_TYPE: PyType = crate::pyobject::new_pytype("SystemExit");
pub static EXC_MEMORY_ERROR_TYPE: PyType = crate::pyobject::new_pytype("MemoryError");
pub static EXC_SYSTEM_ERROR_TYPE: PyType = crate::pyobject::new_pytype("SystemError");
/// PyPy `W_EOFError`, a direct `Exception` subclass used by stream readers
/// such as pickle and marshal.
pub static EXC_EOF_ERROR_TYPE: PyType = crate::pyobject::new_pytype("EOFError");
/// `BufferError` — raised when an operation cannot proceed because a
/// buffer is exported (e.g. resizing a bytearray that backs a live
/// memoryview).  Direct subclass of Exception.
pub static EXC_BUFFER_ERROR_TYPE: PyType = crate::pyobject::new_pytype("BufferError");
/// PyPy `pypy/module/exceptions/interp_exceptions.py:474
/// W_LookupError = _new_exception('LookupError', W_Exception, ...)`
/// — intermediate parent for IndexError and KeyError.
pub static EXC_LOOKUP_ERROR_TYPE: PyType = crate::pyobject::new_pytype("LookupError");
/// PyPy `pypy/module/exceptions/interp_exceptions.py:418
/// W_UnicodeError = _new_exception('UnicodeError', W_ValueError, ...)`
/// — intermediate parent for UnicodeDecodeError and UnicodeEncodeError.
pub static EXC_UNICODE_ERROR_TYPE: PyType = crate::pyobject::new_pytype("UnicodeError");
/// `pypy/module/exceptions/interp_exceptions.py W_SyntaxError` — subclass
/// of Exception raised by `compile`/`exec`/`eval`/`ast.parse`.
pub static EXC_SYNTAX_ERROR_TYPE: PyType = crate::pyobject::new_pytype("SyntaxError");

/// Per-`ExcKind` `ob_type` resolver. `w_exception_new` writes the
/// returned pointer into the allocated `W_BaseException` so the
/// backend's `GuardClass` at `OB_TYPE_OFFSET` matches the actual
/// subclass.
#[inline]
pub fn exc_kind_to_pytype(kind: ExcKind) -> &'static PyType {
    match kind {
        ExcKind::BaseException => &EXCEPTION_TYPE,
        ExcKind::Exception => &EXC_EXCEPTION_TYPE,
        ExcKind::ArithmeticError => &EXC_ARITHMETIC_ERROR_TYPE,
        ExcKind::OverflowError => &EXC_OVERFLOW_ERROR_TYPE,
        ExcKind::ZeroDivisionError => &EXC_ZERO_DIVISION_ERROR_TYPE,
        ExcKind::TypeError => &EXC_TYPE_ERROR_TYPE,
        ExcKind::ValueError => &EXC_VALUE_ERROR_TYPE,
        ExcKind::NameError => &EXC_NAME_ERROR_TYPE,
        ExcKind::UnboundLocalError => &EXC_UNBOUND_LOCAL_ERROR_TYPE,
        ExcKind::IndexError => &EXC_INDEX_ERROR_TYPE,
        ExcKind::KeyError => &EXC_KEY_ERROR_TYPE,
        ExcKind::AttributeError => &EXC_ATTRIBUTE_ERROR_TYPE,
        ExcKind::RuntimeError => &EXC_RUNTIME_ERROR_TYPE,
        ExcKind::StopIteration => &EXC_STOP_ITERATION_TYPE,
        ExcKind::StopAsyncIteration => &EXC_STOP_ASYNC_ITERATION_TYPE,
        ExcKind::ImportError => &EXC_IMPORT_ERROR_TYPE,
        ExcKind::ModuleNotFoundError => &EXC_MODULE_NOT_FOUND_ERROR_TYPE,
        ExcKind::NotImplementedError => &EXC_NOT_IMPLEMENTED_ERROR_TYPE,
        ExcKind::AssertionError => &EXC_ASSERTION_ERROR_TYPE,
        ExcKind::ReferenceError => &EXC_REFERENCE_ERROR_TYPE,
        ExcKind::GeneratorExit => &EXC_GENERATOR_EXIT_TYPE,
        ExcKind::RecursionError => &EXC_RECURSION_ERROR_TYPE,
        ExcKind::OSError => &EXC_OS_ERROR_TYPE,
        ExcKind::FileNotFoundError => &EXC_FILE_NOT_FOUND_ERROR_TYPE,
        ExcKind::UnicodeDecodeError => &EXC_UNICODE_DECODE_ERROR_TYPE,
        ExcKind::UnicodeEncodeError => &EXC_UNICODE_ENCODE_ERROR_TYPE,
        ExcKind::SystemExit => &EXC_SYSTEM_EXIT_TYPE,
        ExcKind::MemoryError => &EXC_MEMORY_ERROR_TYPE,
        ExcKind::SystemError => &EXC_SYSTEM_ERROR_TYPE,
        ExcKind::EOFError => &EXC_EOF_ERROR_TYPE,
        ExcKind::BufferError => &EXC_BUFFER_ERROR_TYPE,
        ExcKind::LookupError => &EXC_LOOKUP_ERROR_TYPE,
        ExcKind::UnicodeError => &EXC_UNICODE_ERROR_TYPE,
        ExcKind::UnicodeTranslateError => &EXC_UNICODE_TRANSLATE_ERROR_TYPE,
        ExcKind::SyntaxError => &EXC_SYNTAX_ERROR_TYPE,
    }
}

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
    // The jd1 drain-match fusion bakes this discriminant as a literal
    // `ConstInt(10)` kind test (majit-translate front/result_exc.rs); keep the
    // two in sync — renumbering silently miscompiles the fused break test.
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
    /// `pypy/module/exceptions/interp_exceptions.py W_SystemError` —
    /// raised by interpreter-internal invariants (e.g.
    /// `chain_exceptions` rejecting non-BaseException context).
    SystemError = 25,
    /// `pypy/module/exceptions/interp_exceptions.py:474
    /// W_LookupError = _new_exception('LookupError', W_Exception, ...)`
    /// — intermediate parent for IndexError and KeyError.
    LookupError = 26,
    /// `pypy/module/exceptions/interp_exceptions.py:418
    /// W_UnicodeError = _new_exception('UnicodeError', W_ValueError, ...)`
    /// — intermediate parent for UnicodeDecodeError and
    /// UnicodeEncodeError.
    UnicodeError = 27,
    /// `pypy/module/exceptions/interp_exceptions.py:426
    /// W_UnicodeTranslateError = _new_exception('UnicodeTranslateError',
    /// W_UnicodeError, ...)`.  Identity-only port: a dedicated kind so
    /// `ob_type` and `isinstance` discriminate it correctly; the 4-arg
    /// `(object, start, end, reason)` `__init__` and custom `__str__`
    /// remain TODO.
    ///
    /// Pyre takes the "union of all per-class fields" route: a single
    /// GC type id for `W_BaseException`, with every per-subclass slot
    /// flattened onto it.  W_UnicodeDecodeError / W_UnicodeEncodeError /
    /// W_UnicodeTranslateError carry `w_object`/`w_start`/`w_end`/
    /// `w_reason`/`w_encoding`; W_OSError carries `w_errno`/`w_strerror`/
    /// `w_filename`/`w_filename2`.  Still identity-only (per-class
    /// fields not yet flattened): W_StopIteration (`w_value`),
    /// W_ImportError (`w_name`/`w_path`/`w_msg`), W_AttributeError
    /// (`w_name`/`w_obj`).  The alternative — per-subclass
    /// `W_<Kind>Object` structs, one GC type id per kind with isolated
    /// layouts — would be more PyPy-orthodox but is not implemented.
    UnicodeTranslateError = 28,
    /// Subclass of ImportError raised when a module cannot be found.
    /// Identity-only like ImportError (no flattened per-class fields).
    ModuleNotFoundError = 29,
    /// `pypy/module/exceptions/interp_exceptions.py W_SyntaxError` —
    /// raised by `compile` / `exec` / `eval` / `ast.parse` on malformed
    /// source.  Identity-only port: a dedicated kind so `ob_type` and
    /// `isinstance(e, SyntaxError)` discriminate it; the
    /// `(msg, (filename, lineno, offset, text))` `__init__` and the
    /// flattened `msg`/`filename`/`lineno`/`offset`/`text` slots remain
    /// TODO (the generic `args_w` constructor is used for now).
    SyntaxError = 30,
    /// Raised when a buffer-related operation cannot proceed — e.g.
    /// resizing a `bytearray` whose storage backs a live `memoryview`
    /// (`PyByteArray_Resize`: "Existing exports of data: object cannot
    /// be re-sized").  Direct subclass of Exception.
    BufferError = 31,
    /// Subclass of NameError raised when a fast local is read while unbound.
    UnboundLocalError = 32,
    /// Signals exhaustion of an asynchronous iterator.  Appended so the
    /// existing discriminants embedded by the JIT remain stable.
    StopAsyncIteration = 33,
    /// Direct Exception subclass raised when a stream ends before an object.
    /// Appended so existing JIT-baked discriminants remain stable.
    EOFError = 34,
}

impl ExcKind {
    /// The largest valid discriminant.  `PyError::kind_from_exc` matches
    /// every variant with no wildcard arm, so the compiler is free to lower
    /// it to a bounds-check-free jump table over exactly `0..=` this value;
    /// a byte outside the range reaching that match is an indirect branch to
    /// an address computed from garbage.  Anything reading the tag out of a
    /// value whose provenance is not proven must go through
    /// `w_exception_kind_checked`, which range-checks against this.
    pub const MAX_DISCRIMINANT: u8 = ExcKind::EOFError as u8;

    /// True when this kind's constructor is the trivial
    /// `W_BaseException.descr_init` (`self.args_w = args_w`) — i.e. it
    /// stores nothing beyond `args_w`.
    ///
    /// False for the kinds whose `descr_init` parses arguments and stores
    /// extra flattened fields (and, for `OSError`, rewrites `args_w`):
    /// `OSError` / `FileNotFoundError` set `errno` / `strerror` /
    /// `filename` / `filename2` (`builtins.rs::os_error_init`,
    /// interp_exceptions.py:552/629); `UnicodeDecodeError` /
    /// `UnicodeEncodeError` / `UnicodeTranslateError` set `w_object` /
    /// `start` / `end` / `reason` (and `encoding` for the codec errors)
    /// (`builtins.rs::exc_unicode_*_error_init`,
    /// interp_exceptions.py:433/1041/1159); `SyntaxError` sets `msg` /
    /// `filename` / `lineno` / `offset` / `text` / `end_lineno` /
    /// `end_offset` (interp_exceptions.py:836); `StopIteration` sets
    /// `value` (:496); `AttributeError` sets `name` / `obj` (:1134);
    /// `NameError` sets `name` (:810); `SystemExit` sets `code` (:993);
    /// `ImportError` / `ModuleNotFoundError` set `name` / `path` /
    /// `name_from` (:363).
    ///
    /// The subclasses that inherit one of these initializers share their
    /// parent's kind — `UnboundLocalError` is a `NameError`,
    /// `IndentationError` and `TabError` are `SyntaxError`s — so naming the
    /// parent covers them.
    ///
    /// A caller that reconstructs an exception from only
    /// `kind` / `w_class` / `args_w` (e.g. the traced inline
    /// constructor) must reject the non-trivial kinds and defer to the
    /// full runtime constructor, which initializes those fields.
    pub fn has_trivial_args_constructor(self) -> bool {
        !matches!(
            self,
            ExcKind::OSError
                | ExcKind::FileNotFoundError
                | ExcKind::UnicodeDecodeError
                | ExcKind::UnicodeEncodeError
                | ExcKind::UnicodeTranslateError
                | ExcKind::SyntaxError
                | ExcKind::StopIteration
                | ExcKind::AttributeError
                | ExcKind::NameError
                | ExcKind::SystemExit
                | ExcKind::ImportError
                | ExcKind::ModuleNotFoundError
        )
    }
}

/// Layout: `[ob_header | kind: ExcKind | args_w: PyObjectRef | …]`
///
/// `args_w` mirrors `pypy/module/exceptions/interp_exceptions.py:121-124`
/// `W_BaseException.descr_init`:
///
/// ```python
/// def descr_init(self, space, args_w):
///     self.args_w = args_w
/// ```
///
/// PyPy keeps `args_w` as an RPython list and rebuilds the tuple on
/// every read (`descr_getargs: return space.newtuple(self.args_w)`,
/// line 153).  Pyre matches that shape line-by-line — the slot points
/// at a `W_ListObject` (RPython list ↔ pyre `W_ListObject` parity);
/// `w_exception_get_args` builds a fresh `W_TupleObject` from the
/// list on every call, and `w_exception_set_args` coerces the
/// incoming iterable via `fixedview` semantics into a brand-new list
/// (line 156 `self.args_w = space.fixedview(w_newargs)`).
///
/// `PY_NULL` means "not yet set" — the `args` getattr arm surfaces an
/// empty tuple in that case, matching the path where the constructor
/// is bypassed (e.g. internal `w_exception_new` callers in
/// `gateway.rs`).
#[repr(C)]
pub struct W_BaseException {
    pub ob_header: PyObject,
    pub kind: ExcKind,
    pub args_w: PyObjectRef,
    /// `interp_exceptions.py:114 W_BaseException.w_cause = None` —
    /// `raise X from Y` cause set by `descr_setcause` (line 167-174).
    /// `PY_NULL` mirrors PyPy's "internal None" (raises AttributeError
    /// on read in CPython; PyPy returns `space.w_None`).
    pub w_cause: PyObjectRef,
    /// `interp_exceptions.py:115 W_BaseException.w_context = None` —
    /// chained exception context set by `descr_setcontext`
    /// (line 183-190).
    pub w_context: PyObjectRef,
    /// `interp_exceptions.py:116 W_BaseException.w_traceback = None` —
    /// traceback object stamped by `descr_settraceback` (line 200-205)
    /// and the `raise` machinery via `OperationError.normalize_exception`.
    pub w_traceback: PyObjectRef,
    /// `interp_exceptions.py:117 W_BaseException.suppress_context =
    /// False` — `raise X from Y` flips this to True via
    /// `descr_setcause` (line 172).
    pub suppress_context: bool,
    /// `interp_exceptions.py:428 W_UnicodeTranslateError.w_object` /
    /// `:1036 W_UnicodeDecodeError.w_object` /
    /// `:1154 W_UnicodeEncodeError.w_object`.  The offending string /
    /// bytes object passed to `__init__`.  Populated by
    /// `descr_init`; `PY_NULL` for non-Unicode-error kinds and for
    /// Unicode errors constructed without going through the public
    /// `descr_init` path (matches PyPy's class-default `w_object = None`
    /// — `descr_str` checks `if self.object is None: return ""`).
    ///
    /// TODO: PyPy uses three distinct
    /// `W_UnicodeTranslateError` / `W_UnicodeDecodeError` /
    /// `W_UnicodeEncodeError` classes each with their own field set.
    /// Pyre flattens them onto `W_BaseException` to keep a single
    /// GC type id; per-kind structural split is tracked separately.
    pub w_object: PyObjectRef,
    /// `interp_exceptions.py:429 W_UnicodeTranslateError.w_start`
    /// (and `:1037` / `:1155` for Decode / Encode).
    pub w_start: PyObjectRef,
    /// `interp_exceptions.py:430 W_UnicodeTranslateError.w_end`
    /// (and `:1038` / `:1156` for Decode / Encode).
    pub w_end: PyObjectRef,
    /// `interp_exceptions.py:431 W_UnicodeTranslateError.w_reason`
    /// (and `:1039` / `:1157` for Decode / Encode).
    pub w_reason: PyObjectRef,
    /// `interp_exceptions.py:1035 W_UnicodeDecodeError.w_encoding` /
    /// `:1153 W_UnicodeEncodeError.w_encoding`.  `W_UnicodeTranslateError`
    /// has no `w_encoding` field per PyPy — left `PY_NULL` for Translate.
    pub w_encoding: PyObjectRef,
    /// `interp_exceptions.py:523 W_OSError.w_errno` — writable
    /// `readwrite_attrproperty_w('w_errno', W_OSError)` slot (`:739`).
    /// `PY_NULL` is the class default `None`; the `errno` getattr arm
    /// falls back to deriving the value from `args_w` when the slot is
    /// unset (the internal-constructor path that bypasses the public
    /// setter), so a later `e.errno = x` write persists here.
    pub w_errno: PyObjectRef,
    /// `interp_exceptions.py:524 W_OSError.w_winerror` — the Windows error
    /// code, exposed as the writable `winerror` attribute only on the platform
    /// that has one (`:723-728` gates the attrproperty on `rwin32.WIN32`).
    /// PyPy declares the slot everywhere and reads it only under that gate;
    /// keeping it unconditional here leaves one exception layout for every
    /// target instead of a Windows-only field ordering.
    pub w_winerror: PyObjectRef,
    /// `interp_exceptions.py:525 W_OSError.w_strerror` /
    /// `:740 readwrite_attrproperty_w('w_strerror', W_OSError)`.
    pub w_strerror: PyObjectRef,
    /// `interp_exceptions.py:526 W_OSError.w_filename` /
    /// `:741 readwrite_attrproperty_w('w_filename', W_OSError)`.
    pub w_filename: PyObjectRef,
    /// `interp_exceptions.py:527 W_OSError.w_filename2` /
    /// `:742 readwrite_attrproperty_w('w_filename2', W_OSError)`.
    pub w_filename2: PyObjectRef,
    /// `interp_exceptions.py:990 W_SystemExit.w_code` /
    /// `:1006 readwrite_attrproperty_w('w_code', W_SystemExit)`.
    /// `PY_NULL` is the class default `None`; the `code` getattr arm
    /// derives the value from `args_w` (descr_init: `args_w[0]` for one
    /// argument, the args tuple for several) when the slot is unset, and
    /// a later `e.code = x` write persists here ahead of that fallback.
    pub w_code: PyObjectRef,
    /// `interp_exceptions.py:494 W_StopIteration.w_value` — initialized to
    /// None, replaced with the first argument by `descr_init`, and exposed as
    /// the writable `value` attribute.
    pub w_value: PyObjectRef,
    /// Shared `w_name` slot for the exception kinds that expose a
    /// `name` attribute: `W_ImportError.w_name`
    /// (`interp_exceptions.py:643`, `:680
    /// readwrite_attrproperty_w('w_name', W_ImportError)`),
    /// `W_NameError.w_name` and `W_AttributeError.w_name` (Python
    /// 3.10+).  An exception is exactly one kind, so a single slot
    /// serves all three.  `PY_NULL` is the class default `None`; set
    /// from the `name=` keyword by `descr_init` and writable via
    /// `e.name = ...`.
    pub w_exc_name: PyObjectRef,
    /// `W_AttributeError.w_obj` — the object whose attribute lookup
    /// failed, set from the `obj=` keyword (Python 3.10+), default
    /// `None`.
    pub w_attr_obj: PyObjectRef,
    /// `interp_exceptions.py:644 W_ImportError.w_path` /
    /// `:681 readwrite_attrproperty_w('w_path', W_ImportError)`, set
    /// from the `path=` keyword.
    pub w_import_path: PyObjectRef,
    /// `W_ImportError.w_name_from` — set from the `name_from=` keyword,
    /// exposed as the `name_from` attribute (default `None`).
    pub w_import_name_from: PyObjectRef,
    /// `interp_exceptions.py:409-411 W_ImportError.w_msg` /
    /// `readwrite_attrproperty_w('w_msg', W_ImportError)`.  Set to the
    /// single positional argument by `descr_init`; read back by the `msg`
    /// attrproperty; class default `None`.
    pub w_import_msg: PyObjectRef,
    /// `interp_exceptions.py:827-834 W_SyntaxError` per-instance fields.
    /// PyPy keeps these on `W_SyntaxError`; pyre's shared exception GC layout
    /// flattens subclass payloads onto `W_BaseException`, as it does for the
    /// Unicode and OSError families above.
    pub w_syntax_msg: PyObjectRef,
    pub w_syntax_filename: PyObjectRef,
    pub w_syntax_lineno: PyObjectRef,
    pub w_syntax_offset: PyObjectRef,
    pub w_syntax_text: PyObjectRef,
    pub w_syntax_end_lineno: PyObjectRef,
    pub w_syntax_end_offset: PyObjectRef,
    pub w_syntax_print_file_and_line: PyObjectRef,
    /// CPython 3.14's private `SyntaxError._metadata` member.  This is the
    /// 3.14-specific extension to PyPy's `W_SyntaxError` field set.
    pub w_syntax_metadata: PyObjectRef,
    /// `interp_group.py:19 W_BaseExceptionGroup.descr_new` `exc.w_message`,
    /// exposed as the read-only `message` attrproperty (`:71`).
    pub w_group_message: PyObjectRef,
    /// `interp_group.py:20` `exc.w_exceptions`, exposed as the read-only
    /// `exceptions` attrproperty (`:72`).  This is the immutable tuple built at
    /// construction time, independent of the `args` the caller passed.
    pub w_group_exceptions: PyObjectRef,
    /// The `repr` of the sequence `descr_new` received, rendered before it was
    /// flattened into `w_group_exceptions`.  `BaseExceptionGroup.__repr__`
    /// reproduces the constructor-time spelling, which a later mutation of
    /// `args` must not change; `PY_NULL` selects the derive-from-args path.
    pub w_group_exceptions_repr: PyObjectRef,
    /// `interp_exceptions.py:113 W_BaseException.w_dict = None` — the
    /// per-instance attribute dict, lazily allocated by `getdict`
    /// (`:222-225`) and replaced wholesale by `setdict` (`:227-231`).
    /// Extra attributes (`e.note = ...`, PEP 678 `__notes__`) live
    /// here.
    pub w_dict: PyObjectRef,
    /// Per-object weakref lifeline.  PyPy's app-level
    /// `W_ExceptionGroup(W_BaseExceptionGroup, W_Exception)` acquires the
    /// ordinary heap-type weakref slot even though `W_BaseExceptionGroup`
    /// itself is not weakrefable.  Pyre flattens all exception payloads into
    /// this struct, so the storage lives here and the `ExceptionGroup` type
    /// flag controls whether it is observable.
    pub w_weakreflifeline: PyObjectRef,
}

pub const EXC_KIND_OFFSET: usize = std::mem::offset_of!(W_BaseException, kind);
pub const EXC_ARGS_W_OFFSET: usize = std::mem::offset_of!(W_BaseException, args_w);
pub const EXC_W_CAUSE_OFFSET: usize = std::mem::offset_of!(W_BaseException, w_cause);
pub const EXC_W_CONTEXT_OFFSET: usize = std::mem::offset_of!(W_BaseException, w_context);
pub const EXC_W_TRACEBACK_OFFSET: usize = std::mem::offset_of!(W_BaseException, w_traceback);
pub const EXC_W_OBJECT_OFFSET: usize = std::mem::offset_of!(W_BaseException, w_object);
pub const EXC_W_START_OFFSET: usize = std::mem::offset_of!(W_BaseException, w_start);
pub const EXC_W_END_OFFSET: usize = std::mem::offset_of!(W_BaseException, w_end);
pub const EXC_W_REASON_OFFSET: usize = std::mem::offset_of!(W_BaseException, w_reason);
pub const EXC_W_ENCODING_OFFSET: usize = std::mem::offset_of!(W_BaseException, w_encoding);
pub const EXC_W_ERRNO_OFFSET: usize = std::mem::offset_of!(W_BaseException, w_errno);
pub const EXC_W_WINERROR_OFFSET: usize = std::mem::offset_of!(W_BaseException, w_winerror);
pub const EXC_W_STRERROR_OFFSET: usize = std::mem::offset_of!(W_BaseException, w_strerror);
pub const EXC_W_FILENAME_OFFSET: usize = std::mem::offset_of!(W_BaseException, w_filename);
pub const EXC_W_FILENAME2_OFFSET: usize = std::mem::offset_of!(W_BaseException, w_filename2);
pub const EXC_W_CODE_OFFSET: usize = std::mem::offset_of!(W_BaseException, w_code);
pub const EXC_W_VALUE_OFFSET: usize = std::mem::offset_of!(W_BaseException, w_value);
pub const EXC_W_NAME_OFFSET: usize = std::mem::offset_of!(W_BaseException, w_exc_name);
pub const EXC_W_ATTR_OBJ_OFFSET: usize = std::mem::offset_of!(W_BaseException, w_attr_obj);
pub const EXC_W_IMPORT_PATH_OFFSET: usize = std::mem::offset_of!(W_BaseException, w_import_path);
pub const EXC_W_IMPORT_NAME_FROM_OFFSET: usize =
    std::mem::offset_of!(W_BaseException, w_import_name_from);
pub const EXC_W_IMPORT_MSG_OFFSET: usize = std::mem::offset_of!(W_BaseException, w_import_msg);
pub const EXC_W_SYNTAX_MSG_OFFSET: usize = std::mem::offset_of!(W_BaseException, w_syntax_msg);
pub const EXC_W_SYNTAX_FILENAME_OFFSET: usize =
    std::mem::offset_of!(W_BaseException, w_syntax_filename);
pub const EXC_W_SYNTAX_LINENO_OFFSET: usize =
    std::mem::offset_of!(W_BaseException, w_syntax_lineno);
pub const EXC_W_SYNTAX_OFFSET_OFFSET: usize =
    std::mem::offset_of!(W_BaseException, w_syntax_offset);
pub const EXC_W_SYNTAX_TEXT_OFFSET: usize = std::mem::offset_of!(W_BaseException, w_syntax_text);
pub const EXC_W_SYNTAX_END_LINENO_OFFSET: usize =
    std::mem::offset_of!(W_BaseException, w_syntax_end_lineno);
pub const EXC_W_SYNTAX_END_OFFSET_OFFSET: usize =
    std::mem::offset_of!(W_BaseException, w_syntax_end_offset);
pub const EXC_W_SYNTAX_PRINT_FILE_AND_LINE_OFFSET: usize =
    std::mem::offset_of!(W_BaseException, w_syntax_print_file_and_line);
pub const EXC_W_SYNTAX_METADATA_OFFSET: usize =
    std::mem::offset_of!(W_BaseException, w_syntax_metadata);
pub const EXC_W_GROUP_MESSAGE_OFFSET: usize =
    std::mem::offset_of!(W_BaseException, w_group_message);
pub const EXC_W_GROUP_EXCEPTIONS_OFFSET: usize =
    std::mem::offset_of!(W_BaseException, w_group_exceptions);
pub const EXC_W_GROUP_EXCEPTIONS_REPR_OFFSET: usize =
    std::mem::offset_of!(W_BaseException, w_group_exceptions_repr);
pub const EXC_W_DICT_OFFSET: usize = std::mem::offset_of!(W_BaseException, w_dict);
pub const EXC_W_WEAKREF_OFFSET: usize = std::mem::offset_of!(W_BaseException, w_weakreflifeline);

/// GC trace offsets for `W_BaseException` — `args_w` plus the three
/// `PyObjectRef`-shaped chained-exception slots per
/// `interp_exceptions.py:113-117 W_BaseException` class defaults,
/// plus the five Unicode*Error per-class slots (w_object / w_start /
/// w_end / w_reason / w_encoding) that PyPy distributes across the
/// W_UnicodeTranslateError / W_UnicodeDecodeError / W_UnicodeEncodeError
/// subclasses, plus the five W_OSError per-class slots (w_errno /
/// w_winerror / w_strerror / w_filename / w_filename2), plus the W_SystemExit
/// `w_code` slot, the W_StopIteration `w_value` slot, plus the shared
/// `w_exc_name` slot (ImportError /
/// NameError / AttributeError) and the W_AttributeError `w_attr_obj`
/// slot, plus the three remaining W_ImportError per-class slots
/// (w_import_path / w_import_name_from / w_import_msg), plus the eight
/// W_SyntaxError fields and CPython 3.14 `_metadata`, plus the three
/// `W_BaseExceptionGroup` slots (w_group_message / w_group_exceptions and the
/// constructor-time sequence repr), plus the lazily-allocated
/// `w_dict`, plus the heap-type weakref lifeline used by
/// `ExceptionGroup`
/// (interp_exceptions.py:113/222-231).  `kind` is a `u8` tag, `message`
/// is a `*mut String` (raw heap), and `suppress_context` is a bool —
/// none of those are GC-traced.
pub const W_BASE_EXCEPTION_GC_PTR_OFFSETS: [usize; 35] = [
    EXC_ARGS_W_OFFSET,
    EXC_W_CAUSE_OFFSET,
    EXC_W_CONTEXT_OFFSET,
    EXC_W_TRACEBACK_OFFSET,
    EXC_W_OBJECT_OFFSET,
    EXC_W_START_OFFSET,
    EXC_W_END_OFFSET,
    EXC_W_REASON_OFFSET,
    EXC_W_ENCODING_OFFSET,
    EXC_W_ERRNO_OFFSET,
    EXC_W_WINERROR_OFFSET,
    EXC_W_STRERROR_OFFSET,
    EXC_W_FILENAME_OFFSET,
    EXC_W_FILENAME2_OFFSET,
    EXC_W_CODE_OFFSET,
    EXC_W_VALUE_OFFSET,
    EXC_W_NAME_OFFSET,
    EXC_W_ATTR_OBJ_OFFSET,
    EXC_W_IMPORT_PATH_OFFSET,
    EXC_W_IMPORT_NAME_FROM_OFFSET,
    EXC_W_IMPORT_MSG_OFFSET,
    EXC_W_SYNTAX_MSG_OFFSET,
    EXC_W_SYNTAX_FILENAME_OFFSET,
    EXC_W_SYNTAX_LINENO_OFFSET,
    EXC_W_SYNTAX_OFFSET_OFFSET,
    EXC_W_SYNTAX_TEXT_OFFSET,
    EXC_W_SYNTAX_END_LINENO_OFFSET,
    EXC_W_SYNTAX_END_OFFSET_OFFSET,
    EXC_W_SYNTAX_PRINT_FILE_AND_LINE_OFFSET,
    EXC_W_SYNTAX_METADATA_OFFSET,
    EXC_W_GROUP_MESSAGE_OFFSET,
    EXC_W_GROUP_EXCEPTIONS_OFFSET,
    EXC_W_GROUP_EXCEPTIONS_REPR_OFFSET,
    EXC_W_DICT_OFFSET,
    EXC_W_WEAKREF_OFFSET,
];

/// GC type id assigned to `W_BaseException` at JitDriver init time.
pub const W_BASE_EXCEPTION_GC_TYPE_ID: u32 = 31;

/// Record an old→young edge when a `W_BaseException` slot
/// (`W_BASE_EXCEPTION_GC_PTR_OFFSETS`) is overwritten after allocation.
/// No-op while the exception still lives in the nursery; once it is
/// promoted, the minor collector relies on the remembered set to find
/// the young pointers reachable only through it. The slot writers below
/// call this for the same reason `function.rs` barriers its setters.
#[inline]
fn exception_write_barrier(obj: PyObjectRef) {
    crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
}

/// Fixed payload size (`framework.py:811`).
pub const W_BASE_EXCEPTION_SIZE: usize = std::mem::size_of::<W_BaseException>();

impl crate::lltype::GcType for W_BaseException {
    fn type_id() -> u32 {
        W_BASE_EXCEPTION_GC_TYPE_ID
    }
    const SIZE: usize = W_BASE_EXCEPTION_SIZE;
}

/// Allocate a new exception object on the heap.
///
/// `ob_header.w_class` is populated from the per-`ExcKind` class
/// registry (`register_exc_class_for_kind`) when the interpreter has
/// finished installing builtin exception types; otherwise it falls
/// back to the generic `EXCEPTION_TYPE` instantiate slot. Callers
/// that rely on `space.type(w_exc)` returning the specific class
/// (e.g. `cmp_exc_match` at `pyopcode.py:1040`) get the registered
/// class once init has run; pre-init callers see the generic
/// placeholder, matching the legacy "internal `w_exception_new`"
/// path.
pub fn w_exception_new(kind: ExcKind, message: &str) -> PyObjectRef {
    let exc = w_exception_new_empty(kind);
    // `oefmt(space.w_ValueError, "...")` parity — an internal raise with
    // a message stores it as the single constructor arg
    // (`args_w = [space.newtext(msg)]`); `descr_str` then derives the
    // string lazily.  Empty message → no args (the `args_w` stays
    // `PY_NULL` so `args` reads as `()`), matching the prebuilt
    // singletons (`MemoryError`, `StopIteration`).
    if !message.is_empty() {
        // Root the fresh managed exception across the arg-list build: `exc`
        // lives only in this Rust local while `w_list_new` allocates, so a
        // collection there could sweep the unrooted (non-moving oldgen)
        // exception before `w_exception_set_args` writes through it.
        let _roots = crate::gc_roots::push_roots();
        crate::gc_roots::pin_root(exc);
        let arg = crate::unicodeobject::w_str_new(message);
        unsafe { w_exception_set_args(exc, w_exception_args_new(vec![arg])) };
    }
    exc
}

/// Like `w_exception_new` but stores an arbitrary WTF-8 message,
/// preserving lone surrogates that a `&str` message cannot carry.
pub fn w_exception_new_wtf8(kind: ExcKind, message: &Wtf8) -> PyObjectRef {
    let exc = w_exception_new_empty(kind);
    if !message.is_empty() {
        // See `w_exception_new`: pin `exc` across the allocating arg build.
        let _roots = crate::gc_roots::push_roots();
        crate::gc_roots::pin_root(exc);
        let arg = crate::unicodeobject::w_str_from_wtf8(message.to_wtf8_buf());
        unsafe { w_exception_set_args(exc, w_exception_args_new(vec![arg])) };
    }
    exc
}

/// Allocate a `W_BaseException` of `kind` with no constructor args
/// (`args_w = PY_NULL`).  The public Python `__new__` path
/// (`exc_constructor`) and the message helpers above attach `args_w`
/// afterwards via `w_exception_set_args`.
pub fn w_exception_new_empty(kind: ExcKind) -> PyObjectRef {
    w_exception_new_empty_impl(kind, false)
}

/// Immortal variant for the prebuilt singletons (`memory_error_singleton` /
/// `standard_exc_instance`): they are cached in `OnceLock<usize>` (GC-invisible)
/// and baked into JIT constant pools as immediate pointers, so they must never
/// be swept — keep them `malloc_typed`-immortal (stable, never reclaimed).
pub fn w_exception_new_empty_immortal(kind: ExcKind) -> PyObjectRef {
    w_exception_new_empty_impl(kind, true)
}

/// `#[dont_look_inside]` (`@jit.dont_look_inside`, `rlib/jit.py:139`), the
/// `w_dict_new` / `w_dict_view_iterator_new_direction` twin: the body builds a
/// `W_BaseException` and boxes it through the non-numeric `malloc_typed`
/// (`fuse_boxing_alloc` fuses only the numeric boxes), so tracing into it
/// carries the unported `malloc->new` lowering into the caller. Residualise
/// the whole constructor — the JIT models it by signature as a plain
/// `PyObjectRef` GCREF and emits a residual call.
#[majit_macros::dont_look_inside]
fn w_exception_new_empty_impl(kind: ExcKind, immortal: bool) -> PyObjectRef {
    let w_class = lookup_exc_class_for_kind(kind);
    let w_class = if w_class != PY_NULL {
        w_class
    } else {
        get_instantiate(&EXCEPTION_TYPE)
    };
    let value = W_BaseException {
        ob_header: PyObject {
            ob_type: exc_kind_to_pytype(kind) as *const PyType,
            w_class,
        },
        kind,
        args_w: PY_NULL,
        w_cause: PY_NULL,
        w_context: PY_NULL,
        w_traceback: PY_NULL,
        suppress_context: false,
        // `interp_exceptions.py:428-431` W_UnicodeTranslateError class
        // defaults `w_object = w_start = w_end = w_reason = None`
        // (and `:1035-1039` Decode / `:1153-1157` Encode add
        // `w_encoding = None`).  PyPy reads `None` as "unset" via
        // `if self.object is None: return ""`; pyre uses `PY_NULL`
        // (the args getattr / descr_str arms surface `space.w_None`
        // when an instance was allocated outside `descr_init`).
        w_object: PY_NULL,
        w_start: PY_NULL,
        w_end: PY_NULL,
        w_reason: PY_NULL,
        w_encoding: PY_NULL,
        // `interp_exceptions.py:523-527` W_OSError class defaults
        // `w_errno = w_winerror = w_strerror = w_filename = w_filename2 = None`.
        w_errno: PY_NULL,
        w_winerror: PY_NULL,
        w_strerror: PY_NULL,
        w_filename: PY_NULL,
        w_filename2: PY_NULL,
        // `interp_exceptions.py:990` W_SystemExit class default
        // `w_code = None`.
        w_code: PY_NULL,
        // `interp_exceptions.py:494` W_StopIteration class default
        // `w_value = None`.
        w_value: PY_NULL,
        // Shared `name` slot (ImportError / NameError / AttributeError)
        // + W_AttributeError `obj`; class default `None`.
        w_exc_name: PY_NULL,
        w_attr_obj: PY_NULL,
        // `interp_exceptions.py:642-644` W_ImportError class defaults
        // `w_msg = w_path = None` (plus `w_name_from`).
        w_import_path: PY_NULL,
        w_import_name_from: PY_NULL,
        w_import_msg: PY_NULL,
        // `interp_exceptions.py:827-834` W_SyntaxError defaults, plus
        // CPython 3.14's private `_metadata` member.
        w_syntax_msg: PY_NULL,
        w_syntax_filename: PY_NULL,
        w_syntax_lineno: PY_NULL,
        w_syntax_offset: PY_NULL,
        w_syntax_text: PY_NULL,
        w_syntax_end_lineno: PY_NULL,
        w_syntax_end_offset: PY_NULL,
        w_syntax_print_file_and_line: PY_NULL,
        w_syntax_metadata: PY_NULL,
        // `interp_group.py:19-20` W_BaseExceptionGroup defaults, stamped by
        // `descr_new` on the group kinds only.
        w_group_message: PY_NULL,
        w_group_exceptions: PY_NULL,
        w_group_exceptions_repr: PY_NULL,
        // `interp_exceptions.py:113 w_dict = None` — allocated on the
        // first `getdict` (`:222-225`).
        w_dict: PY_NULL,
        // Only ExceptionGroup exposes this slot; the shared flattened
        // exception layout keeps it null for every other exception kind.
        w_weakreflifeline: PY_NULL,
    };
    if !immortal {
        // GC-manage the exception object: allocate it in the non-moving
        // oldgen so accessors can deref a bare `*W_BaseException` and the
        // JIT can carry it as a raw i64 across allocating opcodes without
        // it moving. Oldgen is mark-sweep, so all carriers must root it
        // (`walk_in_flight_exception`, the value-stack walker, and the
        // raw-i64 JIT carriers). Mirrors `w_generator_new`.
        let raw = crate::gc_hook::try_gc_alloc_stable_raw(
            W_BASE_EXCEPTION_GC_TYPE_ID,
            W_BASE_EXCEPTION_SIZE,
        );
        if !raw.is_null() {
            unsafe {
                std::ptr::write(raw as *mut W_BaseException, value);
            }
            crate::gc_hook::try_gc_write_barrier(raw);
            return raw as PyObjectRef;
        }
        return crate::lltype::malloc_typed(value) as PyObjectRef;
    }
    crate::lltype::malloc_typed(value) as PyObjectRef
}

/// Per-`ExcKind` class-pointer registry. Populated by
/// `pyre-interpreter::builtins::register_exc_class` during
/// `install_default_builtins`; consumed by `w_exception_new` so each
/// builtin-raised exception's `ob_header.w_class` points at the
/// specific class object (rather than the generic `EXCEPTION_TYPE`).
/// PyPy's equivalent is the `space.w_TypeError` / `space.w_ValueError`
/// / ... attributes on `ObjSpace`.
///
/// The builtin `W_TypeObject` identities and this registry are process-global.
/// A class installed by one execution-context thread must therefore be the
/// same class used to stamp and match exceptions on every other thread.
/// Registration is first-writer-wins so rebuilding a builtins dictionary
/// cannot replace a canonical class. The pointer is stored as `usize` because
/// `PyObjectRef` itself is neither `Send` nor `Sync`; builtin type objects are
/// immortal and process-global.
/// One slot per `ExcKind` variant.  Indexed by `kind as u8 as usize`,
/// so `EXC_KIND_COUNT - 1` is the largest valid index.  Public so
/// downstream crates (e.g. pyre-jit's GC init) can size per-kind
/// arrays against the same authoritative bound.  Anchored on the
/// highest-numbered variant so adding new ExcKinds at the end of the
/// enum extends the bound automatically.
pub const EXC_KIND_COUNT: usize = (ExcKind::EOFError as u8 as usize) + 1;

static EXC_CLASS_BY_KIND: [std::sync::atomic::AtomicUsize; EXC_KIND_COUNT] =
    [const { std::sync::atomic::AtomicUsize::new(0) }; EXC_KIND_COUNT];

/// Register `cls` for `kind` if the process-global slot is empty and return
/// the canonical class selected by the first writer.
pub fn register_exc_class_for_kind(kind: ExcKind, cls: PyObjectRef) -> PyObjectRef {
    let slot = &EXC_CLASS_BY_KIND[kind as u8 as usize];
    match slot.compare_exchange(
        0,
        cls as usize,
        std::sync::atomic::Ordering::AcqRel,
        std::sync::atomic::Ordering::Acquire,
    ) {
        Ok(_) => cls,
        Err(canonical) => canonical as PyObjectRef,
    }
}

/// Reads the process-global `EXC_CLASS_BY_KIND`, a runtime-mutable root the
/// tracer cannot type; the JIT residualises the read instead of tracing into
/// it (`@dont_look_inside`, `rlib/jit.py:139`). The residual call resolves its
/// address by qualified path in `jit_trace_fnaddrs`.
#[majit_macros::dont_look_inside]
pub fn lookup_exc_class_for_kind(kind: ExcKind) -> PyObjectRef {
    EXC_CLASS_BY_KIND[kind as u8 as usize].load(std::sync::atomic::Ordering::Acquire) as PyObjectRef
}

/// True when `cls` is one of the canonical process-global builtin exception
/// classes registered via `register_exc_class_for_kind` — i.e. its
/// constructor is the Rust `descr_init` (no Python `__init__`).
pub fn is_canonical_exc_class(cls: PyObjectRef) -> bool {
    !cls.is_null()
        && EXC_CLASS_BY_KIND
            .iter()
            .any(|slot| slot.load(std::sync::atomic::Ordering::Acquire) == cls as usize)
}

/// `interp_exceptions.py:153 W_BaseException.descr_getargs` parity —
///
/// ```python
/// def descr_getargs(self, space):
///     return space.newtuple(self.args_w)
/// ```
///
/// Returns a freshly-built tuple wrapping the items of the internal
/// list slot, or an empty tuple when the exception was constructed
/// without going through the public `descr_init` path (e.g. internal
/// `w_exception_new` callers in `gateway.rs` that leave `args_w` as
/// `PY_NULL`).  Each call materialises a *new* tuple, mirroring
/// PyPy's "list → fresh newtuple per read" idiom (so
/// `e.args is e.args` is False — see `descr_getargs` line 153).
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_get_args(obj: PyObjectRef) -> PyObjectRef {
    unsafe {
        let stored = (*(obj as *const W_BaseException)).args_w;
        if stored.is_null() {
            return crate::tupleobject::w_tuple_new(Vec::new());
        }
        // PyPy: `space.newtuple(self.args_w)`.  `args_w` is an
        // RPython list (pyre: `W_ListObject`); flatten its items into
        // a freshly-allocated tuple.
        let items: Vec<PyObjectRef> = if crate::pyobject::is_list(stored) {
            let len = crate::listobject::w_list_len(stored) as i64;
            let mut items = Vec::with_capacity(len as usize);
            for i in 0..len {
                items.push(
                    crate::listobject::w_list_getitem(stored, i)
                        .unwrap_or(crate::pyobject::PY_NULL),
                );
            }
            items
        } else if crate::pyobject::is_tuple(stored) {
            // Legacy compat — pre-list storage path; treat as already
            // a sequence and rebuild the tuple identically.
            let len = crate::tupleobject::w_tuple_len(stored) as i64;
            let mut items = Vec::with_capacity(len as usize);
            for i in 0..len {
                items.push(
                    crate::tupleobject::w_tuple_getitem(stored, i)
                        .unwrap_or(crate::pyobject::PY_NULL),
                );
            }
            items
        } else {
            Vec::new()
        };
        crate::tupleobject::w_tuple_new(items)
    }
}

/// Build the `args_w` storage list for an exception.
///
/// `interp_exceptions.py:114` declares `args_w = []` — an RPython
/// `list of W_Root`, i.e. a plain array of object pointers.  List
/// *strategies* are a `W_ListObject` feature of the app-level list type
/// (`objspace/std/listobject.py`) and have no counterpart in an RPython
/// list, so `args_w` must never take the unboxed `Integer` / `Float`
/// representation `w_list_new` would pick for `ValueError(7)`, nor the
/// `Empty` one it picks for `ValueError()`.
///
/// Beyond parity this is what keeps the slot readable: the `args` load
/// fold walks the object items block directly, and declines on any other
/// strategy, leaving `e.args` as a residual `getattr` call.
pub fn w_exception_args_new(items: Vec<PyObjectRef>) -> PyObjectRef {
    crate::listobject::w_list_new_object(items)
}

/// Raw `args_w` storage for JIT field mirrors.  Unlike
/// [`w_exception_get_args`], this does not allocate the public tuple view.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_get_args_storage(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_BaseException)).args_w }
}

/// `interp_exceptions.py:123-124 W_BaseException.descr_init` /
/// `:156-157 descr_setargs` parity —
///
/// ```python
/// def descr_init(self, space, args_w):
///     self.args_w = args_w
///
/// def descr_setargs(self, space, w_newargs):
///     self.args_w = space.fixedview(w_newargs)
/// ```
///
/// Stores a `W_ListObject` carrying the constructor / setter items.
/// Callers (`baseobjspace::coerce_to_list_for_args`) pre-flatten any
/// iterable into a list via `space.fixedview` semantics so the slot
/// always holds a list — matching PyPy's `args_w: list of W_Root`
/// type.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_set_args(obj: PyObjectRef, args_list: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_BaseException)).args_w = args_list;
        exception_write_barrier(obj);
    }
}

/// `interp_exceptions.py:163-164 descr_getcause` parity —
///
/// ```python
/// def descr_getcause(self, space):
///     return self.w_cause
/// ```
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_get_cause(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_BaseException)).w_cause }
}

/// `interp_exceptions.py:166-174 descr_setcause` parity — writes the
/// `w_cause` slot.  Type validation (None or BaseException subclass
/// instance) is enforced at the call site (`baseobjspace::setattr_str`).
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_set_cause(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_BaseException)).w_cause = value;
        exception_write_barrier(obj);
    }
}

/// `interp_exceptions.py:180-181 descr_getcontext` parity —
///
/// ```python
/// def descr_getcontext(self, space):
///     return self.w_context
/// ```
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_get_context(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_BaseException)).w_context }
}

/// `interp_exceptions.py:183-190 descr_setcontext` parity — writes
/// the `w_context` slot.  Type validation lives in
/// `baseobjspace::setattr_str`.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_set_context(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_BaseException)).w_context = value;
        exception_write_barrier(obj);
    }
}

/// The raw `self.w_traceback` slot read.  `descr_gettraceback`
/// (`interp_exceptions.py:196-201`) and `OperationError.get_traceback`
/// (`error.py:359-370`) are this read plus `tb.frame.mark_as_escaped()`;
/// that mark lives in `pytraceback::mark_traceback_escaped`, since the
/// frame type is not visible from this crate.  Callers mirroring either
/// getter pair the two; callers mirroring a direct `_application_traceback`
/// read (printing, chain trimming) use this alone.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_get_traceback(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_BaseException)).w_traceback }
}

/// `interp_exceptions.py:203-205 descr_settraceback` parity — writes
/// the `w_traceback` slot.  Type validation lives in
/// `baseobjspace::setattr_str`.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_set_traceback(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_BaseException)).w_traceback = value;
        exception_write_barrier(obj);
    }
}

/// `interp_exceptions.py:222-225 getdict` parity —
///
/// ```python
/// def getdict(self, space):
///     if self.w_dict is None:
///         self.w_dict = space.newdict(instance=True)
///     return self.w_dict
/// ```
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_getdict(obj: PyObjectRef) -> PyObjectRef {
    unsafe {
        let exc = obj as *mut W_BaseException;
        if (*exc).w_dict.is_null() {
            (*exc).w_dict = crate::dictmultiobject::w_dict_new_instance();
            exception_write_barrier(obj);
        }
        (*exc).w_dict
    }
}

/// `descr_reduce` reads `self.w_dict` WITHOUT allocating it — returns the
/// raw slot (`PY_NULL` when unset), so a `__reduce__` over an attribute-less
/// exception does not leave behind a fresh empty instance dict.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_peek_dict(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_BaseException)).w_dict }
}

/// `interp_exceptions.py:227-231 setdict` parity — writes the `w_dict`
/// slot.  The non-dict `TypeError` check lives in the caller
/// (`baseobjspace::setdict`).
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_setdict(obj: PyObjectRef, w_dict: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_BaseException)).w_dict = w_dict;
        exception_write_barrier(obj);
    }
}

/// Read the per-exception weakref lifeline.  Only `ExceptionGroup`'s type
/// advertises this storage; keeping it on the flattened exception payload
/// matches the object-owned lifeline used by PyPy heap instances.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_getweakref(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_BaseException)).w_weakreflifeline }
}

/// Store the per-exception weakref lifeline and remember an old-to-young edge.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_setweakref(obj: PyObjectRef, lifeline: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_BaseException)).w_weakreflifeline = lifeline;
        exception_write_barrier(obj);
    }
}

/// `interp_exceptions.py:212-213 descr_getsuppresscontext` parity —
///
/// ```python
/// def descr_getsuppresscontext(self, space):
///     return space.newbool(self.suppress_context)
/// ```
///
/// Returns the raw bool; the caller wraps with `w_bool_from`.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_get_suppress_context(obj: PyObjectRef) -> bool {
    unsafe { (*(obj as *const W_BaseException)).suppress_context }
}

/// `interp_exceptions.py:215-216 descr_setsuppresscontext` parity —
/// writes the `suppress_context` slot after the caller has resolved
/// `space.bool_w(w_value)` into a Rust bool.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_set_suppress_context(obj: PyObjectRef, value: bool) {
    unsafe {
        (*(obj as *mut W_BaseException)).suppress_context = value;
    }
}

// ─── Unicode*Error per-class field accessors ────────────────────────
//
// `interp_exceptions.py:468-471 W_UnicodeTranslateError.typedef`
// (and `:1080-1084 W_UnicodeDecodeError.typedef` /
// `:1200-1204 W_UnicodeEncodeError.typedef`) wire each field via
// `readwrite_attrproperty_w('w_object', ...)` etc.  Pyre's
// `baseobjspace::getattr_str` and `setattr` arms dispatch on the
// attribute name + ExcKind and route here.
//
// All five accessors return `space.w_None` (resolved by the caller)
// when the slot is `PY_NULL`, matching PyPy's class-default
// `w_object = None` etc. — `descr_str` checks `if self.object is
// None:` and short-circuits to `""`.

/// `interp_exceptions.py:468 readwrite_attrproperty_w('w_object', ...)`
/// — `e.object` reader.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_get_object(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_BaseException)).w_object }
}

/// `interp_exceptions.py:468 readwrite_attrproperty_w('w_object', ...)`
/// — `e.object = ...` writer.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_set_object(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_BaseException)).w_object = value;
        exception_write_barrier(obj);
    }
}

/// `interp_exceptions.py:469 readwrite_attrproperty_w('w_start', ...)`
/// — `e.start` reader.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_get_start(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_BaseException)).w_start }
}

/// `interp_exceptions.py:469 readwrite_attrproperty_w('w_start', ...)`
/// — `e.start = ...` writer.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_set_start(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_BaseException)).w_start = value;
        exception_write_barrier(obj);
    }
}

/// `interp_exceptions.py:470 readwrite_attrproperty_w('w_end', ...)`
/// — `e.end` reader.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_get_end(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_BaseException)).w_end }
}

/// `interp_exceptions.py:470 readwrite_attrproperty_w('w_end', ...)`
/// — `e.end = ...` writer.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_set_end(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_BaseException)).w_end = value;
        exception_write_barrier(obj);
    }
}

/// `interp_exceptions.py:471 readwrite_attrproperty_w('w_reason', ...)`
/// — `e.reason` reader.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_get_reason(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_BaseException)).w_reason }
}

/// `interp_exceptions.py:471 readwrite_attrproperty_w('w_reason', ...)`
/// — `e.reason = ...` writer.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_set_reason(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_BaseException)).w_reason = value;
        exception_write_barrier(obj);
    }
}

/// `interp_exceptions.py:1080 readwrite_attrproperty_w('w_encoding',
/// ...)` / `:1200 ...` — `e.encoding` reader (Decode / Encode only;
/// Translate has no encoding field but the slot is still backed by
/// `PY_NULL`).
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_get_encoding(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_BaseException)).w_encoding }
}

/// `interp_exceptions.py:1080 readwrite_attrproperty_w('w_encoding',
/// ...)` / `:1200 ...` — `e.encoding = ...` writer.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_set_encoding(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_BaseException)).w_encoding = value;
        exception_write_barrier(obj);
    }
}

/// `interp_exceptions.py:739 readwrite_attrproperty_w('w_errno', ...)`
/// — `e.errno` reader.  `PY_NULL` means the slot was never written
/// (the `errno` getattr arm then derives the value from `args_w`).
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_get_errno(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_BaseException)).w_errno }
}

/// `interp_exceptions.py:739 readwrite_attrproperty_w('w_errno', ...)`
/// — `e.errno = ...` writer.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_set_errno(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_BaseException)).w_errno = value;
        exception_write_barrier(obj);
    }
}

/// `interp_exceptions.py:725 readwrite_attrproperty_w('w_winerror', ...)`
/// — `e.winerror` reader.  `PY_NULL` means no Windows error code was
/// supplied, which is every instance off Windows and the ones built from
/// an errno on it.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_get_winerror(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_BaseException)).w_winerror }
}

/// `interp_exceptions.py:725 readwrite_attrproperty_w('w_winerror', ...)`
/// — `e.winerror = ...` writer.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_set_winerror(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_BaseException)).w_winerror = value;
        exception_write_barrier(obj);
    }
}

/// `interp_exceptions.py:740 readwrite_attrproperty_w('w_strerror', ...)`
/// — `e.strerror` reader.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_get_strerror(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_BaseException)).w_strerror }
}

/// `interp_exceptions.py:740 readwrite_attrproperty_w('w_strerror', ...)`
/// — `e.strerror = ...` writer.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_set_strerror(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_BaseException)).w_strerror = value;
        exception_write_barrier(obj);
    }
}

/// `interp_exceptions.py:741 readwrite_attrproperty_w('w_filename', ...)`
/// — `e.filename` reader.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_get_filename(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_BaseException)).w_filename }
}

/// `interp_exceptions.py:741 readwrite_attrproperty_w('w_filename', ...)`
/// — `e.filename = ...` writer.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_set_filename(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_BaseException)).w_filename = value;
        exception_write_barrier(obj);
    }
}

/// `interp_exceptions.py:742 readwrite_attrproperty_w('w_filename2', ...)`
/// — `e.filename2` reader.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_get_filename2(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_BaseException)).w_filename2 }
}

/// `interp_exceptions.py:742 readwrite_attrproperty_w('w_filename2', ...)`
/// — `e.filename2 = ...` writer.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_set_filename2(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_BaseException)).w_filename2 = value;
        exception_write_barrier(obj);
    }
}

/// `interp_exceptions.py:1006 readwrite_attrproperty_w('w_code', ...)`
/// — `e.code` reader.  `PY_NULL` means the slot was never written (the
/// `code` getattr arm then derives the value from `args_w`).
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_get_code(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_BaseException)).w_code }
}

/// `interp_exceptions.py:1006 readwrite_attrproperty_w('w_code', ...)`
/// — `e.code = ...` writer.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_set_code(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_BaseException)).w_code = value;
        exception_write_barrier(obj);
    }
}

/// `interp_exceptions.py:508 readwrite_attrproperty_w('w_value', ...)` —
/// `StopIteration.value` reader.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_get_value(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_BaseException)).w_value }
}

/// `interp_exceptions.py:508 readwrite_attrproperty_w('w_value', ...)` —
/// `StopIteration.value = ...` writer.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_set_value(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_BaseException)).w_value = value;
        exception_write_barrier(obj);
    }
}

/// Shared `e.name` reader for ImportError / NameError / AttributeError
/// (`readwrite_attrproperty_w('w_name', ...)`).
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_get_name(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_BaseException)).w_exc_name }
}

/// Shared `e.name = ...` writer for ImportError / NameError /
/// AttributeError.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_set_name(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_BaseException)).w_exc_name = value;
        exception_write_barrier(obj);
    }
}

/// `e.obj` reader (W_AttributeError).
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_get_attr_obj(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_BaseException)).w_attr_obj }
}

/// `e.obj = ...` writer (W_AttributeError).
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_set_attr_obj(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_BaseException)).w_attr_obj = value;
        exception_write_barrier(obj);
    }
}

/// `interp_exceptions.py:681 readwrite_attrproperty_w('w_path', ...)`
/// — `e.path` reader (W_ImportError).
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_get_import_path(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_BaseException)).w_import_path }
}

/// `e.path = ...` writer (W_ImportError).
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_set_import_path(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_BaseException)).w_import_path = value;
        exception_write_barrier(obj);
    }
}

/// `e.name_from` reader (W_ImportError).
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_get_import_name_from(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_BaseException)).w_import_name_from }
}

/// `e.name_from = ...` writer (W_ImportError).
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_set_import_name_from(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_BaseException)).w_import_name_from = value;
        exception_write_barrier(obj);
    }
}

/// `interp_exceptions.py:679 readwrite_attrproperty_w('w_msg', ...)`
/// — `e.msg` reader (W_ImportError).  `PY_NULL` means the slot was never
/// written (the `msg` getattr arm then derives the value from `args_w`).
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_get_import_msg(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_BaseException)).w_import_msg }
}

/// `e.msg = ...` writer (W_ImportError).
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_set_import_msg(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_BaseException)).w_import_msg = value;
        exception_write_barrier(obj);
    }
}

/// `interp_exceptions.py:827 W_SyntaxError.w_filename` reader.
#[inline]
pub unsafe fn w_exception_get_syntax_filename(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_BaseException)).w_syntax_filename }
}

/// `interp_exceptions.py:827 W_SyntaxError.w_filename` writer.
#[inline]
pub unsafe fn w_exception_set_syntax_filename(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_BaseException)).w_syntax_filename = value;
        exception_write_barrier(obj);
    }
}

/// `interp_exceptions.py:828 W_SyntaxError.w_lineno` reader.
#[inline]
pub unsafe fn w_exception_get_syntax_lineno(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_BaseException)).w_syntax_lineno }
}

/// `interp_exceptions.py:828 W_SyntaxError.w_lineno` writer.
#[inline]
pub unsafe fn w_exception_set_syntax_lineno(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_BaseException)).w_syntax_lineno = value;
        exception_write_barrier(obj);
    }
}

/// `interp_exceptions.py:829 W_SyntaxError.w_offset` reader.
#[inline]
pub unsafe fn w_exception_get_syntax_offset(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_BaseException)).w_syntax_offset }
}

/// `interp_exceptions.py:829 W_SyntaxError.w_offset` writer.
#[inline]
pub unsafe fn w_exception_set_syntax_offset(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_BaseException)).w_syntax_offset = value;
        exception_write_barrier(obj);
    }
}

/// `interp_exceptions.py:830 W_SyntaxError.w_text` reader.
#[inline]
pub unsafe fn w_exception_get_syntax_text(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_BaseException)).w_syntax_text }
}

/// `interp_exceptions.py:830 W_SyntaxError.w_text` writer.
#[inline]
pub unsafe fn w_exception_set_syntax_text(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_BaseException)).w_syntax_text = value;
        exception_write_barrier(obj);
    }
}

/// `interp_exceptions.py:831 W_SyntaxError.w_msg` reader.
#[inline]
pub unsafe fn w_exception_get_syntax_msg(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_BaseException)).w_syntax_msg }
}

/// `interp_exceptions.py:831 W_SyntaxError.w_msg` writer.
#[inline]
pub unsafe fn w_exception_set_syntax_msg(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_BaseException)).w_syntax_msg = value;
        exception_write_barrier(obj);
    }
}

/// `interp_exceptions.py:832 W_SyntaxError.w_print_file_and_line` reader.
#[inline]
pub unsafe fn w_exception_get_syntax_print_file_and_line(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_BaseException)).w_syntax_print_file_and_line }
}

/// `interp_exceptions.py:832 W_SyntaxError.w_print_file_and_line` writer.
#[inline]
pub unsafe fn w_exception_set_syntax_print_file_and_line(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_BaseException)).w_syntax_print_file_and_line = value;
        exception_write_barrier(obj);
    }
}

/// `interp_exceptions.py:833 W_SyntaxError.w_end_lineno` reader.
#[inline]
pub unsafe fn w_exception_get_syntax_end_lineno(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_BaseException)).w_syntax_end_lineno }
}

/// `interp_exceptions.py:833 W_SyntaxError.w_end_lineno` writer.
#[inline]
pub unsafe fn w_exception_set_syntax_end_lineno(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_BaseException)).w_syntax_end_lineno = value;
        exception_write_barrier(obj);
    }
}

/// `interp_exceptions.py:834 W_SyntaxError.w_end_offset` reader.
#[inline]
pub unsafe fn w_exception_get_syntax_end_offset(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_BaseException)).w_syntax_end_offset }
}

/// `interp_exceptions.py:834 W_SyntaxError.w_end_offset` writer.
#[inline]
pub unsafe fn w_exception_set_syntax_end_offset(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_BaseException)).w_syntax_end_offset = value;
        exception_write_barrier(obj);
    }
}

/// CPython 3.14 `SyntaxError._metadata` reader.
#[inline]
pub unsafe fn w_exception_get_syntax_metadata(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_BaseException)).w_syntax_metadata }
}

/// CPython 3.14 `SyntaxError._metadata` writer.
#[inline]
pub unsafe fn w_exception_set_syntax_metadata(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_BaseException)).w_syntax_metadata = value;
        exception_write_barrier(obj);
    }
}

/// `interp_group.py:19` `exc.w_message` reader.
#[inline]
pub unsafe fn w_exception_get_group_message(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_BaseException)).w_group_message }
}

/// `interp_group.py:19` `exc.w_message` writer.
#[inline]
pub unsafe fn w_exception_set_group_message(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_BaseException)).w_group_message = value;
        exception_write_barrier(obj);
    }
}

/// `interp_group.py:20` `exc.w_exceptions` reader.
#[inline]
pub unsafe fn w_exception_get_group_exceptions(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_BaseException)).w_group_exceptions }
}

/// `interp_group.py:20` `exc.w_exceptions` writer.
#[inline]
pub unsafe fn w_exception_set_group_exceptions(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_BaseException)).w_group_exceptions = value;
        exception_write_barrier(obj);
    }
}

/// Constructor-time `repr` of the sequence `descr_new` received.
#[inline]
pub unsafe fn w_exception_get_group_exceptions_repr(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_BaseException)).w_group_exceptions_repr }
}

/// Constructor-time `repr` of the sequence `descr_new` received.
#[inline]
pub unsafe fn w_exception_set_group_exceptions_repr(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_BaseException)).w_group_exceptions_repr = value;
        exception_write_barrier(obj);
    }
}

/// `compile.py:1090` `memory_error = MemoryError()` parity — module-level
/// singleton instance the JIT raises through
/// `PropagateExceptionDescr.handle_fail` when a malloc helper returns
/// NULL.  RPython allocates the singleton at translation time; pyre
/// allocates lazily on first OOM (most workloads never trigger it).
///
/// Stored as `usize` because `PyObjectRef` is `*mut PyObject`, which is
/// neither `Send` nor `Sync` — `OnceLock<usize>` is the standard escape
/// hatch.  The `W_BaseException` lives forever: it is cached in the
/// GC-invisible `OnceLock` and baked into JIT constant pools, so it must
/// stay immortal (`w_exception_new_empty_immortal`), never GC-swept.
pub fn memory_error_singleton() -> PyObjectRef {
    *MEMORY_ERROR_SINGLETON
        .get_or_init(|| w_exception_new_empty_immortal(ExcKind::MemoryError) as usize)
        as PyObjectRef
}

static MEMORY_ERROR_SINGLETON: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

static STANDARD_EXC_INSTANCES: [std::sync::OnceLock<usize>; EXC_KIND_COUNT] =
    [const { std::sync::OnceLock::new() }; EXC_KIND_COUNT];

/// Visit every immortal exception singleton that has actually been created.
///
/// The singletons themselves are `malloc_typed` and outlive every collection,
/// but the `args_w` / `w_traceback` / `w_context` a raise attaches to them are
/// ordinary GC-managed objects. Because the holder is not managed, neither the
/// write barrier nor major seeding reaches those children, so they survive only
/// while some raw carrier happens to be parked on the singleton. Enumerating
/// them here lets a root walker forward the children per object instead.
///
/// Only initialized slots are reported — reading through `get()` never forces
/// an allocation, which must not happen from inside a collection.
pub fn for_each_immortal_exception_singleton(mut visit: impl FnMut(PyObjectRef)) {
    if let Some(&raw) = MEMORY_ERROR_SINGLETON.get() {
        visit(raw as PyObjectRef);
    }
    for slot in STANDARD_EXC_INSTANCES.iter() {
        if let Some(&raw) = slot.get() {
            visit(raw as PyObjectRef);
        }
    }
}

/// `rpython/rtyper/exceptiondata.py:34-38 get_standard_ll_exc_instance`
/// parity — return the reusable prebuilt instance for `kind`.  RPython's
/// `r_inst.get_reusable_prebuilt_instance()` materialises a single
/// instance per classdef at rtyper construction time and reuses it for
/// every `flatten.py:165-170 self.emitline("raise", c)` call site (the
/// `_ovf` direct raise path).
///
/// Pyre allocates per `ExcKind` lazily on first access; the resulting
/// pointer is valid for the lifetime of the process and stable across
/// calls so a JIT'd constant pool can carry it as an immediate pointer.
/// Same `OnceLock<usize>` escape hatch as `memory_error_singleton`
/// because `PyObjectRef` is neither `Send` nor `Sync`.
pub fn standard_exc_instance(kind: ExcKind) -> PyObjectRef {
    let slot = &STANDARD_EXC_INSTANCES[kind as u8 as usize];
    *slot.get_or_init(|| w_exception_new_empty_immortal(kind) as usize) as PyObjectRef
}

/// Check if an object is an exception instance.
///
/// Uses `ll_isinstance` against the `BaseException` root
/// (`EXCEPTION_TYPE`); every per-kind exception `PyType` is registered
/// as a descendant via `all_foreign_pytypes`, so the
/// `subclassrange_{min,max}` check (`rclass.py:1133-1137`) matches
/// every subclass without pointer-identity coupling.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_exception(obj: PyObjectRef) -> bool {
    crate::pyobject::ensure_object_subclass_ranges_initialized();
    // `ll_issubclass` reads the ranges under the seqlock, so a concurrent
    // one-time batch re-stamp cannot make this spuriously false.
    unsafe { ll_isinstance(obj, &EXCEPTION_TYPE) }
}

/// Get the exception kind tag.
///
/// # Safety
/// `obj` must point to a valid `W_BaseException`.
#[inline]
pub unsafe fn w_exception_get_kind(obj: PyObjectRef) -> ExcKind {
    unsafe { (*(obj as *const W_BaseException)).kind }
}

/// The raw tag byte, read without interpreting it as an `ExcKind`.
///
/// # Safety
/// `obj` must be non-null and point to at least
/// `offset_of!(W_BaseException, kind) + 1` readable bytes.
#[inline]
pub unsafe fn w_exception_kind_byte(obj: PyObjectRef) -> u8 {
    unsafe {
        std::ptr::addr_of!((*(obj as *const W_BaseException)).kind)
            .cast::<u8>()
            .read()
    }
}

/// `w_exception_get_kind` for a value whose provenance is not proven — a raw
/// resume word handed back by the blackhole, say.
///
/// `blackhole.py:1679-1682 _exit_frame_with_exception` casts its value to
/// GCREF and every later classification runs through a genuine class lookup
/// (`bh_classof` / `space.exception_match`, i.e. the `rclass.py:1133-1137`
/// subclass ranges).  Pyre reads a `#[repr(u8)]` tag out of the object
/// instead, which turns a bad value into an out-of-range index into a
/// bounds-check-free jump table (`ExcKind::MAX_DISCRIMINANT`).  This restores
/// the class check in front of the tag read: `is_exception` is the
/// `ll_isinstance` port.
///
/// The class check runs *before* the tag range check. `kind` is a tail field
/// past `ob_header`, so reading it out of a value that is not an exception can
/// read past the object — `W_NoneObject` is a bare header and ends exactly
/// where `kind` starts. `ob_type` sits at offset 0, so the class check needs
/// only the header to be readable, and once `is_exception` holds the object is
/// a `W_BaseException` and the tag load is in bounds. The range check stays
/// after it because the class check does not prove the byte is a live
/// discriminant, and that byte is what indexes the jump table.
///
/// The screens stop at the shape of `ob_type`: they do not prove it addresses
/// a live type. `is_exception` reads the subclass range through it, so an
/// aligned non-null word that is not a type header still faults. Proving it
/// would need a heap-membership test, and the only one available (`is_tracked`)
/// runs as a `gc_op` — it leaves RUNNING and can park the mutator, which is not
/// something a classification on the blackhole resume path may do. The caller's
/// contract carries that obligation instead.
///
/// # Safety
/// `obj` must be null or point to at least `size_of::<PyObject>()` readable
/// bytes, and its `ob_type` must be null or address a readable type header.
#[inline]
pub unsafe fn w_exception_kind_checked(obj: PyObjectRef) -> Option<ExcKind> {
    // Every screen below precedes the dereference it protects, so that a value
    // which is not an object at all is rejected rather than faulting:
    // alignment before the `ob_type` load, `ob_type`'s own shape before
    // `ll_issubclass` reads the subclass range through it, and the class
    // before the tail-field tag load.
    if obj.is_null() || !(obj as usize).is_multiple_of(align_of::<W_BaseException>()) {
        return None;
    }
    let ob_type = unsafe { (*obj).ob_type };
    if ob_type.is_null()
        || !(ob_type as usize).is_multiple_of(align_of::<crate::pyobject::PyType>())
    {
        return None;
    }
    if !unsafe { is_exception(obj) } {
        return None;
    }
    if unsafe { w_exception_kind_byte(obj) } > ExcKind::MAX_DISCRIMINANT {
        return None;
    }
    Some(unsafe { w_exception_get_kind(obj) })
}

/// Reads the caught exception's `kind` discriminant as an integer, the
/// residual-callable twin of `w_exception_get_kind`.  The tracer cannot
/// model the raw pointer read, so the JIT residualises the call rather than
/// tracing into it (`@dont_look_inside`); the residual resolves its address
/// by qualified path in `jit_trace_fnaddrs`.  A non-inline standalone graph
/// (unlike the `#[inline]` accessor) is what the census residualises.
#[majit_macros::dont_look_inside]
pub fn exc_kind_discriminant(evalue: PyObjectRef) -> i64 {
    // Safety: `evalue` is a valid `W_BaseException` (the caught exception).
    unsafe { w_exception_get_kind(evalue) as i64 }
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
        ExcKind::UnboundLocalError => "UnboundLocalError",
        ExcKind::IndexError => "IndexError",
        ExcKind::KeyError => "KeyError",
        ExcKind::AttributeError => "AttributeError",
        ExcKind::RuntimeError => "RuntimeError",
        ExcKind::StopIteration => "StopIteration",
        ExcKind::StopAsyncIteration => "StopAsyncIteration",
        ExcKind::OverflowError => "OverflowError",
        ExcKind::ArithmeticError => "ArithmeticError",
        ExcKind::ImportError => "ImportError",
        ExcKind::ModuleNotFoundError => "ModuleNotFoundError",
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
        ExcKind::SystemError => "SystemError",
        ExcKind::EOFError => "EOFError",
        ExcKind::LookupError => "LookupError",
        ExcKind::UnicodeError => "UnicodeError",
        ExcKind::UnicodeTranslateError => "UnicodeTranslateError",
        ExcKind::SyntaxError => "SyntaxError",
        ExcKind::BufferError => "BufferError",
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
    if type_name == "NameError" {
        return matches!(kind, ExcKind::NameError | ExcKind::UnboundLocalError);
    }
    // ImportError hierarchy — ModuleNotFoundError is-a ImportError.
    if type_name == "ImportError" {
        return matches!(kind, ExcKind::ImportError | ExcKind::ModuleNotFoundError);
    }
    // OSError hierarchy — FileNotFoundError is-a OSError is-a Exception.
    // IOError / EnvironmentError are aliases for OSError in Python 3.
    if type_name == "OSError" || type_name == "IOError" || type_name == "EnvironmentError" {
        return matches!(kind, ExcKind::OSError | ExcKind::FileNotFoundError);
    }
    // Unicode errors are subclasses of UnicodeError which is a
    // subclass of ValueError, so "ValueError" matches everything in
    // the UnicodeError subtree too.
    if type_name == "ValueError" {
        return matches!(
            kind,
            ExcKind::ValueError
                | ExcKind::UnicodeError
                | ExcKind::UnicodeDecodeError
                | ExcKind::UnicodeEncodeError
                | ExcKind::UnicodeTranslateError
        );
    }
    if type_name == "UnicodeError" {
        return matches!(
            kind,
            ExcKind::UnicodeError
                | ExcKind::UnicodeDecodeError
                | ExcKind::UnicodeEncodeError
                | ExcKind::UnicodeTranslateError
        );
    }
    // LookupError is the intermediate parent of IndexError and KeyError
    // (`pypy/module/exceptions/interp_exceptions.py:474`).
    if type_name == "LookupError" {
        return matches!(
            kind,
            ExcKind::LookupError | ExcKind::IndexError | ExcKind::KeyError
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
        "UnboundLocalError" => Some(ExcKind::UnboundLocalError),
        "IndexError" => Some(ExcKind::IndexError),
        "KeyError" => Some(ExcKind::KeyError),
        "AttributeError" => Some(ExcKind::AttributeError),
        "RuntimeError" => Some(ExcKind::RuntimeError),
        "StopIteration" => Some(ExcKind::StopIteration),
        "StopAsyncIteration" => Some(ExcKind::StopAsyncIteration),
        "OverflowError" => Some(ExcKind::OverflowError),
        "ArithmeticError" => Some(ExcKind::ArithmeticError),
        "ImportError" => Some(ExcKind::ImportError),
        "ModuleNotFoundError" => Some(ExcKind::ModuleNotFoundError),
        "NotImplementedError" => Some(ExcKind::NotImplementedError),
        "AssertionError" => Some(ExcKind::AssertionError),
        "ReferenceError" => Some(ExcKind::ReferenceError),
        "GeneratorExit" => Some(ExcKind::GeneratorExit),
        // `rpython/rlib/rstackovf.py:10-14 StackOverflow` is a
        // `RuntimeError` subclass that RPython's rtyper synthesizes
        // catch/convert code for; `rpython/annotator/exception.py:3`
        // lists `_StackOverflow` in the standard set so
        // `get_standard_ll_exc_instance_by_class` has a prebuilt
        // instance for it.  Pyre doesn't have an LL-side StackOverflow
        // class — the stack-check slowpath raises a Python-level
        // `RecursionError` directly (`eval.rs:2979 stack_check_slow
        // path → pos_exception()`) — so we alias the RPython name to
        // pyre's `RecursionError` ExcKind: every consumer that looks
        // up the standard pointer receives the singleton instance
        // whose `kind` is the user-visible class, matching what user
        // code would catch.
        "RecursionError" | "_StackOverflow" | "StackOverflow" => Some(ExcKind::RecursionError),
        "OSError" | "IOError" | "EnvironmentError" => Some(ExcKind::OSError),
        "FileNotFoundError" => Some(ExcKind::FileNotFoundError),
        "UnicodeDecodeError" => Some(ExcKind::UnicodeDecodeError),
        "UnicodeEncodeError" => Some(ExcKind::UnicodeEncodeError),
        "SystemExit" => Some(ExcKind::SystemExit),
        "MemoryError" => Some(ExcKind::MemoryError),
        "SystemError" => Some(ExcKind::SystemError),
        "EOFError" => Some(ExcKind::EOFError),
        "LookupError" => Some(ExcKind::LookupError),
        "UnicodeError" => Some(ExcKind::UnicodeError),
        "UnicodeTranslateError" => Some(ExcKind::UnicodeTranslateError),
        "SyntaxError" => Some(ExcKind::SyntaxError),
        "BufferError" => Some(ExcKind::BufferError),
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
            // The message is stored as the single constructor arg.
            let args = w_exception_get_args(obj);
            assert_eq!(crate::tupleobject::w_tuple_len(args), 1);
            let arg0 = crate::tupleobject::w_tuple_getitem(args, 0).unwrap();
            assert_eq!(
                crate::unicodeobject::w_str_get_wtf8(arg0),
                Wtf8::new("bad value")
            );
        }
    }

    /// A byte that is not a live discriminant must not reach `kind_from_exc`:
    /// that match has no wildcard arm, so an out-of-range byte indexes a
    /// bounds-check-free jump table.
    ///
    /// The subject is a stack copy of a real exception, so it clears the
    /// alignment and class screens and the tag range is the gate under test.
    /// A `[u8; N]` buffer would not: its alignment is 1 and its header is
    /// null, so either of the earlier screens could be the one that rejects.
    #[test]
    fn test_kind_checked_rejects_out_of_range_tag() {
        let live = w_exception_new(ExcKind::ValueError, "bad value");
        let mut copy: W_BaseException = unsafe { std::ptr::read(live as *const W_BaseException) };
        let obj = std::ptr::addr_of_mut!(copy) as PyObjectRef;
        assert_eq!(
            unsafe { w_exception_kind_checked(obj) },
            Some(ExcKind::ValueError),
            "the copy must pass every screen before the tag is corrupted"
        );
        let tag = std::ptr::addr_of_mut!(copy.kind).cast::<u8>();
        for raw in [ExcKind::MAX_DISCRIMINANT + 1, 128, 160, u8::MAX] {
            unsafe { tag.write(raw) };
            assert_eq!(
                unsafe { w_exception_kind_checked(obj) },
                None,
                "tag byte {raw} must be rejected"
            );
        }
    }

    /// The class screen rejects a word whose header is not an exception type
    /// without ever loading the tail field the tag lives in.
    #[test]
    fn test_kind_checked_rejects_non_exception_header() {
        let buf = [0usize; std::mem::size_of::<W_BaseException>() / 8];
        assert_eq!(
            unsafe { w_exception_kind_checked(buf.as_ptr() as PyObjectRef) },
            None
        );
        assert_eq!(
            unsafe { w_exception_kind_checked(crate::noneobject::w_none()) },
            None
        );
    }

    #[test]
    fn test_kind_checked_accepts_real_exception() {
        for kind in [ExcKind::ValueError, ExcKind::EOFError] {
            let obj = w_exception_new(kind, "bad value");
            assert_eq!(unsafe { w_exception_kind_checked(obj) }, Some(kind));
        }
        assert_eq!(
            unsafe { w_exception_kind_checked(std::ptr::null_mut()) },
            None
        );
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
        assert!(exc_kind_matches(
            ExcKind::UnboundLocalError,
            "UnboundLocalError"
        ));
        assert!(exc_kind_matches(ExcKind::UnboundLocalError, "NameError"));
        assert!(exc_kind_matches(ExcKind::UnboundLocalError, "Exception"));
        assert!(exc_kind_matches(
            ExcKind::UnboundLocalError,
            "BaseException"
        ));
    }

    #[test]
    fn test_exc_kind_from_name_roundtrip() {
        // Every variant of ExcKind must round-trip through
        // exc_kind_name → exc_kind_from_name so the per-kind class
        // registry (`register_exc_class_for_kind`) plumbed by
        // pyre-interpreter::builtins::register_exc_class can install a
        // class pointer for every `w_exception_new(kind, ...)` callsite.
        // A gap here would leave that kind's `ob_header.w_class` at the
        // generic `EXCEPTION_TYPE` stub, breaking the "the object's
        // class is the exception type" invariant on the w_class read
        // path.
        for kind in [
            ExcKind::BaseException,
            ExcKind::Exception,
            ExcKind::TypeError,
            ExcKind::ValueError,
            ExcKind::ZeroDivisionError,
            ExcKind::NameError,
            ExcKind::UnboundLocalError,
            ExcKind::IndexError,
            ExcKind::KeyError,
            ExcKind::AttributeError,
            ExcKind::RuntimeError,
            ExcKind::StopIteration,
            ExcKind::StopAsyncIteration,
            ExcKind::OverflowError,
            ExcKind::ArithmeticError,
            ExcKind::ImportError,
            ExcKind::NotImplementedError,
            ExcKind::AssertionError,
            ExcKind::ReferenceError,
            ExcKind::GeneratorExit,
            ExcKind::RecursionError,
            ExcKind::OSError,
            ExcKind::FileNotFoundError,
            ExcKind::UnicodeDecodeError,
            ExcKind::UnicodeEncodeError,
            ExcKind::SystemExit,
            ExcKind::MemoryError,
            ExcKind::SystemError,
            ExcKind::EOFError,
            ExcKind::LookupError,
            ExcKind::UnicodeError,
            ExcKind::UnicodeTranslateError,
            ExcKind::BufferError,
        ] {
            let name = exc_kind_name(kind);
            assert_eq!(
                exc_kind_from_name(name),
                Some(kind),
                "exc_kind_from_name({name:?}) round-trip failed for {kind:?}",
            );
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
            // Empty message → no constructor args (`args == ()`).
            assert_eq!(crate::tupleobject::w_tuple_len(w_exception_get_args(a)), 0);
        }
    }

    #[test]
    fn standard_exc_instance_is_idempotent_and_per_kind_distinct() {
        // RPython `get_standard_ll_exc_instance` returns the same
        // prebuilt instance pointer across repeated lookups (it's the
        // `_reusable_prebuilt_instance` slot on the InstanceRepr).
        // Pyre matches by caching per-`ExcKind`; the test pins both
        // the idempotence (same kind → same pointer) and the per-kind
        // distinctness (different kinds → different pointers, so the
        // JIT cannot accidentally merge `raise OverflowError` and
        // `raise ZeroDivisionError` into the same singleton).
        let overflow_a = standard_exc_instance(ExcKind::OverflowError);
        let overflow_b = standard_exc_instance(ExcKind::OverflowError);
        assert_eq!(
            overflow_a as usize, overflow_b as usize,
            "per-kind singleton must be stable across calls"
        );
        let zerodiv = standard_exc_instance(ExcKind::ZeroDivisionError);
        assert_ne!(
            overflow_a as usize, zerodiv as usize,
            "distinct ExcKinds must yield distinct singleton pointers"
        );
        unsafe {
            assert!(is_exception(overflow_a));
            assert_eq!(w_exception_get_kind(overflow_a), ExcKind::OverflowError);
            assert_eq!(w_exception_get_kind(zerodiv), ExcKind::ZeroDivisionError);
        }
    }

    #[test]
    fn immortal_singleton_enumeration_reports_created_and_forces_none() {
        // The root walker forwards each reported singleton's reference slots,
        // so the enumeration must cover every singleton that exists — and must
        // never itself create one, since it runs from inside a collection.
        let mem = memory_error_singleton();
        let key = standard_exc_instance(ExcKind::KeyError);

        let mut seen = Vec::new();
        for_each_immortal_exception_singleton(|exc| seen.push(exc as usize));
        assert!(
            seen.contains(&(mem as usize)),
            "MemoryError singleton must be reported once created"
        );
        assert!(
            seen.contains(&(key as usize)),
            "per-kind singleton must be reported once created"
        );
        assert!(
            seen.len() <= EXC_KIND_COUNT + 1,
            "enumeration is bounded by the per-kind slots plus MemoryError"
        );

        // Enumerating must not initialize a slot: a second pass still reports
        // every singleton the first one did, stays within the bound, and hands
        // back real exception objects.  The slots are process-global and the
        // test binary runs its cases concurrently, so a sibling case creating
        // another kind in between makes the second set a superset — comparing
        // the two for equality would fail on that alone.
        let mut again = Vec::new();
        for_each_immortal_exception_singleton(|exc| again.push(exc as usize));
        for raw in &seen {
            assert!(again.contains(raw), "enumeration must not drop a singleton");
        }
        assert!(
            again.len() <= EXC_KIND_COUNT + 1,
            "enumeration is bounded by the per-kind slots plus MemoryError"
        );
        for &raw in &again {
            assert!(unsafe { is_exception(raw as PyObjectRef) });
        }
    }

    #[test]
    fn w_exception_gc_type_id_matches_descr() {
        assert_eq!(W_BASE_EXCEPTION_GC_TYPE_ID, 31);
        assert_eq!(
            <W_BaseException as crate::lltype::GcType>::type_id(),
            W_BASE_EXCEPTION_GC_TYPE_ID
        );
        assert_eq!(
            <W_BaseException as crate::lltype::GcType>::SIZE,
            W_BASE_EXCEPTION_SIZE
        );
    }
}
