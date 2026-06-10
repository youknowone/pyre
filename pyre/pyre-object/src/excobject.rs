//! W_ExceptionObject — Python exception instance.
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
pub static EXC_INDEX_ERROR_TYPE: PyType = crate::pyobject::new_pytype("IndexError");
pub static EXC_KEY_ERROR_TYPE: PyType = crate::pyobject::new_pytype("KeyError");
pub static EXC_ATTRIBUTE_ERROR_TYPE: PyType = crate::pyobject::new_pytype("AttributeError");
pub static EXC_RUNTIME_ERROR_TYPE: PyType = crate::pyobject::new_pytype("RuntimeError");
pub static EXC_STOP_ITERATION_TYPE: PyType = crate::pyobject::new_pytype("StopIteration");
pub static EXC_IMPORT_ERROR_TYPE: PyType = crate::pyobject::new_pytype("ImportError");
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
/// PyPy `pypy/module/exceptions/interp_exceptions.py:474
/// W_LookupError = _new_exception('LookupError', W_Exception, ...)`
/// — intermediate parent for IndexError and KeyError.
pub static EXC_LOOKUP_ERROR_TYPE: PyType = crate::pyobject::new_pytype("LookupError");
/// PyPy `pypy/module/exceptions/interp_exceptions.py:418
/// W_UnicodeError = _new_exception('UnicodeError', W_ValueError, ...)`
/// — intermediate parent for UnicodeDecodeError and UnicodeEncodeError.
pub static EXC_UNICODE_ERROR_TYPE: PyType = crate::pyobject::new_pytype("UnicodeError");

/// Per-`ExcKind` `ob_type` resolver. `w_exception_new` writes the
/// returned pointer into the allocated `W_ExceptionObject` so the
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
        ExcKind::IndexError => &EXC_INDEX_ERROR_TYPE,
        ExcKind::KeyError => &EXC_KEY_ERROR_TYPE,
        ExcKind::AttributeError => &EXC_ATTRIBUTE_ERROR_TYPE,
        ExcKind::RuntimeError => &EXC_RUNTIME_ERROR_TYPE,
        ExcKind::StopIteration => &EXC_STOP_ITERATION_TYPE,
        ExcKind::ImportError => &EXC_IMPORT_ERROR_TYPE,
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
        ExcKind::LookupError => &EXC_LOOKUP_ERROR_TYPE,
        ExcKind::UnicodeError => &EXC_UNICODE_ERROR_TYPE,
        ExcKind::UnicodeTranslateError => &EXC_UNICODE_TRANSLATE_ERROR_TYPE,
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
    /// GC type id for `W_ExceptionObject`, with every per-subclass slot
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
}

impl ExcKind {
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
    /// interp_exceptions.py:433/1041/1159).
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
pub struct W_ExceptionObject {
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
    /// Pyre flattens them onto `W_ExceptionObject` to keep a single
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
    /// `interp_exceptions.py:525 W_OSError.w_strerror` /
    /// `:740 readwrite_attrproperty_w('w_strerror', W_OSError)`.
    pub w_strerror: PyObjectRef,
    /// `interp_exceptions.py:526 W_OSError.w_filename` /
    /// `:741 readwrite_attrproperty_w('w_filename', W_OSError)`.
    pub w_filename: PyObjectRef,
    /// `interp_exceptions.py:527 W_OSError.w_filename2` /
    /// `:742 readwrite_attrproperty_w('w_filename2', W_OSError)`.
    pub w_filename2: PyObjectRef,
}

pub const EXC_KIND_OFFSET: usize = std::mem::offset_of!(W_ExceptionObject, kind);
pub const EXC_ARGS_W_OFFSET: usize = std::mem::offset_of!(W_ExceptionObject, args_w);
pub const EXC_W_CAUSE_OFFSET: usize = std::mem::offset_of!(W_ExceptionObject, w_cause);
pub const EXC_W_CONTEXT_OFFSET: usize = std::mem::offset_of!(W_ExceptionObject, w_context);
pub const EXC_W_TRACEBACK_OFFSET: usize = std::mem::offset_of!(W_ExceptionObject, w_traceback);
pub const EXC_W_OBJECT_OFFSET: usize = std::mem::offset_of!(W_ExceptionObject, w_object);
pub const EXC_W_START_OFFSET: usize = std::mem::offset_of!(W_ExceptionObject, w_start);
pub const EXC_W_END_OFFSET: usize = std::mem::offset_of!(W_ExceptionObject, w_end);
pub const EXC_W_REASON_OFFSET: usize = std::mem::offset_of!(W_ExceptionObject, w_reason);
pub const EXC_W_ENCODING_OFFSET: usize = std::mem::offset_of!(W_ExceptionObject, w_encoding);
pub const EXC_W_ERRNO_OFFSET: usize = std::mem::offset_of!(W_ExceptionObject, w_errno);
pub const EXC_W_STRERROR_OFFSET: usize = std::mem::offset_of!(W_ExceptionObject, w_strerror);
pub const EXC_W_FILENAME_OFFSET: usize = std::mem::offset_of!(W_ExceptionObject, w_filename);
pub const EXC_W_FILENAME2_OFFSET: usize = std::mem::offset_of!(W_ExceptionObject, w_filename2);

/// GC trace offsets for `W_ExceptionObject` — `args_w` plus the three
/// `PyObjectRef`-shaped chained-exception slots per
/// `interp_exceptions.py:113-117 W_BaseException` class defaults,
/// plus the five Unicode*Error per-class slots (w_object / w_start /
/// w_end / w_reason / w_encoding) that PyPy distributes across the
/// W_UnicodeTranslateError / W_UnicodeDecodeError / W_UnicodeEncodeError
/// subclasses, plus the four W_OSError per-class slots (w_errno /
/// w_strerror / w_filename / w_filename2).  `kind` is a `u8` tag,
/// `message` is a `*mut String` (raw heap), and `suppress_context` is
/// a bool — none of those are GC-traced.
pub const W_EXCEPTION_GC_PTR_OFFSETS: [usize; 13] = [
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
    EXC_W_STRERROR_OFFSET,
    EXC_W_FILENAME_OFFSET,
    EXC_W_FILENAME2_OFFSET,
];

/// GC type id assigned to `W_ExceptionObject` at JitDriver init time.
pub const W_EXCEPTION_GC_TYPE_ID: u32 = 31;

/// Fixed payload size (`framework.py:811`).
pub const W_EXCEPTION_OBJECT_SIZE: usize = std::mem::size_of::<W_ExceptionObject>();

impl crate::lltype::GcType for W_ExceptionObject {
    fn type_id() -> u32 {
        W_EXCEPTION_GC_TYPE_ID
    }
    const SIZE: usize = W_EXCEPTION_OBJECT_SIZE;
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
        let arg = crate::strobject::w_str_new(message);
        unsafe { w_exception_set_args(exc, crate::listobject::w_list_new(vec![arg])) };
    }
    exc
}

/// Like `w_exception_new` but stores an arbitrary WTF-8 message,
/// preserving lone surrogates that a `&str` message cannot carry.
pub fn w_exception_new_wtf8(kind: ExcKind, message: &Wtf8) -> PyObjectRef {
    let exc = w_exception_new_empty(kind);
    if !message.is_empty() {
        let arg = crate::strobject::w_str_from_wtf8(message.to_wtf8_buf());
        unsafe { w_exception_set_args(exc, crate::listobject::w_list_new(vec![arg])) };
    }
    exc
}

/// Allocate a `W_ExceptionObject` of `kind` with no constructor args
/// (`args_w = PY_NULL`).  The public Python `__new__` path
/// (`exc_constructor`) and the message helpers above attach `args_w`
/// afterwards via `w_exception_set_args`.
pub fn w_exception_new_empty(kind: ExcKind) -> PyObjectRef {
    let w_class = lookup_exc_class_for_kind(kind);
    let w_class = if w_class != PY_NULL {
        w_class
    } else {
        get_instantiate(&EXCEPTION_TYPE)
    };
    crate::lltype::malloc_typed(W_ExceptionObject {
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
        // `w_errno = w_strerror = w_filename = w_filename2 = None`.
        w_errno: PY_NULL,
        w_strerror: PY_NULL,
        w_filename: PY_NULL,
        w_filename2: PY_NULL,
    }) as PyObjectRef
}

/// Per-`ExcKind` class-pointer registry. Populated by
/// `pyre-interpreter::builtins::register_exc_class` during
/// `install_default_builtins`; consumed by `w_exception_new` so each
/// builtin-raised exception's `ob_header.w_class` points at the
/// specific class object (rather than the generic `EXCEPTION_TYPE`).
/// PyPy's equivalent is the `space.w_TypeError` / `space.w_ValueError`
/// / ... attributes on `ObjSpace`.
///
/// Stored as `thread_local!` because pyre's `W_TypeObject` identities
/// are also per-thread (each cargo test thread re-runs
/// `init_typeobjects` and gets its own `W_TypeObject` pointers via
/// `TYPEOBJECT_CACHE`). A global `AtomicPtr` cache on
/// `PyType.instantiate` would let one test thread's write race ahead
/// of another's, causing `exception_match` on thread A to compare
/// against thread B's W_TypeObject identity — they'd never match.
/// One slot per `ExcKind` variant.  Indexed by `kind as u8 as usize`,
/// so `EXC_KIND_COUNT - 1` is the largest valid index.  Public so
/// downstream crates (e.g. pyre-jit's GC init) can size per-kind
/// arrays against the same authoritative bound.  Anchored on the
/// highest-numbered variant so adding new ExcKinds at the end of the
/// enum extends the bound automatically.
pub const EXC_KIND_COUNT: usize = (ExcKind::UnicodeTranslateError as u8 as usize) + 1;

thread_local! {
    static EXC_CLASS_BY_KIND: std::cell::Cell<[PyObjectRef; EXC_KIND_COUNT]> =
        const { std::cell::Cell::new([PY_NULL; EXC_KIND_COUNT]) };
}

pub fn register_exc_class_for_kind(kind: ExcKind, cls: PyObjectRef) {
    EXC_CLASS_BY_KIND.with(|cell| {
        let mut table = cell.get();
        table[kind as u8 as usize] = cls;
        cell.set(table);
    });
}

pub fn lookup_exc_class_for_kind(kind: ExcKind) -> PyObjectRef {
    EXC_CLASS_BY_KIND.with(|cell| cell.get()[kind as u8 as usize])
}

/// True when `cls` is one of the canonical per-kind builtin exception
/// classes registered via `register_exc_class_for_kind` — i.e. its
/// constructor is the Rust `descr_init` (no Python `__init__`).
pub fn is_canonical_exc_class(cls: PyObjectRef) -> bool {
    !cls.is_null() && EXC_CLASS_BY_KIND.with(|cell| cell.get().contains(&cls))
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
/// `obj` must point to a valid `W_ExceptionObject`.
#[inline]
pub unsafe fn w_exception_get_args(obj: PyObjectRef) -> PyObjectRef {
    unsafe {
        let stored = (*(obj as *const W_ExceptionObject)).args_w;
        if stored.is_null() {
            return crate::tupleobject::w_tuple_new(Vec::new());
        }
        // PyPy: `space.newtuple(self.args_w)`.  `args_w` is an
        // RPython list (pyre: `W_ListObject`); flatten its items into
        // a freshly-allocated tuple.
        let items: Vec<PyObjectRef> = if crate::pyobject::is_list(stored) {
            let len = crate::listobject::w_list_len(stored) as i64;
            (0..len)
                .map(|i| {
                    crate::listobject::w_list_getitem(stored, i).unwrap_or(crate::pyobject::PY_NULL)
                })
                .collect()
        } else if crate::pyobject::is_tuple(stored) {
            // Legacy compat — pre-list storage path; treat as already
            // a sequence and rebuild the tuple identically.
            let len = crate::tupleobject::w_tuple_len(stored) as i64;
            (0..len)
                .map(|i| {
                    crate::tupleobject::w_tuple_getitem(stored, i)
                        .unwrap_or(crate::pyobject::PY_NULL)
                })
                .collect()
        } else {
            Vec::new()
        };
        crate::tupleobject::w_tuple_new(items)
    }
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
/// `obj` must point to a valid `W_ExceptionObject`.
#[inline]
pub unsafe fn w_exception_set_args(obj: PyObjectRef, args_list: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_ExceptionObject)).args_w = args_list;
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
/// `obj` must point to a valid `W_ExceptionObject`.
#[inline]
pub unsafe fn w_exception_get_cause(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_ExceptionObject)).w_cause }
}

/// `interp_exceptions.py:166-174 descr_setcause` parity — writes the
/// `w_cause` slot.  Type validation (None or BaseException subclass
/// instance) is enforced at the call site (`baseobjspace::setattr_str`).
///
/// # Safety
/// `obj` must point to a valid `W_ExceptionObject`.
#[inline]
pub unsafe fn w_exception_set_cause(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_ExceptionObject)).w_cause = value;
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
/// `obj` must point to a valid `W_ExceptionObject`.
#[inline]
pub unsafe fn w_exception_get_context(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_ExceptionObject)).w_context }
}

/// `interp_exceptions.py:183-190 descr_setcontext` parity — writes
/// the `w_context` slot.  Type validation lives in
/// `baseobjspace::setattr_str`.
///
/// # Safety
/// `obj` must point to a valid `W_ExceptionObject`.
#[inline]
pub unsafe fn w_exception_set_context(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_ExceptionObject)).w_context = value;
    }
}

/// `interp_exceptions.py:196-201 descr_gettraceback` parity (minus
/// the `PyTraceback.frame.mark_as_escaped()` callback, which pyre
/// does not have yet — see TODO on
/// `baseobjspace::getattr_str`'s `__traceback__` arm).
///
/// # Safety
/// `obj` must point to a valid `W_ExceptionObject`.
#[inline]
pub unsafe fn w_exception_get_traceback(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_ExceptionObject)).w_traceback }
}

/// `interp_exceptions.py:203-205 descr_settraceback` parity — writes
/// the `w_traceback` slot.  Type validation lives in
/// `baseobjspace::setattr_str`.
///
/// # Safety
/// `obj` must point to a valid `W_ExceptionObject`.
#[inline]
pub unsafe fn w_exception_set_traceback(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_ExceptionObject)).w_traceback = value;
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
/// `obj` must point to a valid `W_ExceptionObject`.
#[inline]
pub unsafe fn w_exception_get_suppress_context(obj: PyObjectRef) -> bool {
    unsafe { (*(obj as *const W_ExceptionObject)).suppress_context }
}

/// `interp_exceptions.py:215-216 descr_setsuppresscontext` parity —
/// writes the `suppress_context` slot after the caller has resolved
/// `space.bool_w(w_value)` into a Rust bool.
///
/// # Safety
/// `obj` must point to a valid `W_ExceptionObject`.
#[inline]
pub unsafe fn w_exception_set_suppress_context(obj: PyObjectRef, value: bool) {
    unsafe {
        (*(obj as *mut W_ExceptionObject)).suppress_context = value;
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
/// `obj` must point to a valid `W_ExceptionObject`.
#[inline]
pub unsafe fn w_exception_get_object(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_ExceptionObject)).w_object }
}

/// `interp_exceptions.py:468 readwrite_attrproperty_w('w_object', ...)`
/// — `e.object = ...` writer.
///
/// # Safety
/// `obj` must point to a valid `W_ExceptionObject`.
#[inline]
pub unsafe fn w_exception_set_object(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_ExceptionObject)).w_object = value;
    }
}

/// `interp_exceptions.py:469 readwrite_attrproperty_w('w_start', ...)`
/// — `e.start` reader.
///
/// # Safety
/// `obj` must point to a valid `W_ExceptionObject`.
#[inline]
pub unsafe fn w_exception_get_start(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_ExceptionObject)).w_start }
}

/// `interp_exceptions.py:469 readwrite_attrproperty_w('w_start', ...)`
/// — `e.start = ...` writer.
///
/// # Safety
/// `obj` must point to a valid `W_ExceptionObject`.
#[inline]
pub unsafe fn w_exception_set_start(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_ExceptionObject)).w_start = value;
    }
}

/// `interp_exceptions.py:470 readwrite_attrproperty_w('w_end', ...)`
/// — `e.end` reader.
///
/// # Safety
/// `obj` must point to a valid `W_ExceptionObject`.
#[inline]
pub unsafe fn w_exception_get_end(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_ExceptionObject)).w_end }
}

/// `interp_exceptions.py:470 readwrite_attrproperty_w('w_end', ...)`
/// — `e.end = ...` writer.
///
/// # Safety
/// `obj` must point to a valid `W_ExceptionObject`.
#[inline]
pub unsafe fn w_exception_set_end(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_ExceptionObject)).w_end = value;
    }
}

/// `interp_exceptions.py:471 readwrite_attrproperty_w('w_reason', ...)`
/// — `e.reason` reader.
///
/// # Safety
/// `obj` must point to a valid `W_ExceptionObject`.
#[inline]
pub unsafe fn w_exception_get_reason(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_ExceptionObject)).w_reason }
}

/// `interp_exceptions.py:471 readwrite_attrproperty_w('w_reason', ...)`
/// — `e.reason = ...` writer.
///
/// # Safety
/// `obj` must point to a valid `W_ExceptionObject`.
#[inline]
pub unsafe fn w_exception_set_reason(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_ExceptionObject)).w_reason = value;
    }
}

/// `interp_exceptions.py:1080 readwrite_attrproperty_w('w_encoding',
/// ...)` / `:1200 ...` — `e.encoding` reader (Decode / Encode only;
/// Translate has no encoding field but the slot is still backed by
/// `PY_NULL`).
///
/// # Safety
/// `obj` must point to a valid `W_ExceptionObject`.
#[inline]
pub unsafe fn w_exception_get_encoding(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_ExceptionObject)).w_encoding }
}

/// `interp_exceptions.py:1080 readwrite_attrproperty_w('w_encoding',
/// ...)` / `:1200 ...` — `e.encoding = ...` writer.
///
/// # Safety
/// `obj` must point to a valid `W_ExceptionObject`.
#[inline]
pub unsafe fn w_exception_set_encoding(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_ExceptionObject)).w_encoding = value;
    }
}

/// `interp_exceptions.py:739 readwrite_attrproperty_w('w_errno', ...)`
/// — `e.errno` reader.  `PY_NULL` means the slot was never written
/// (the `errno` getattr arm then derives the value from `args_w`).
///
/// # Safety
/// `obj` must point to a valid `W_ExceptionObject`.
#[inline]
pub unsafe fn w_exception_get_errno(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_ExceptionObject)).w_errno }
}

/// `interp_exceptions.py:739 readwrite_attrproperty_w('w_errno', ...)`
/// — `e.errno = ...` writer.
///
/// # Safety
/// `obj` must point to a valid `W_ExceptionObject`.
#[inline]
pub unsafe fn w_exception_set_errno(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_ExceptionObject)).w_errno = value;
    }
}

/// `interp_exceptions.py:740 readwrite_attrproperty_w('w_strerror', ...)`
/// — `e.strerror` reader.
///
/// # Safety
/// `obj` must point to a valid `W_ExceptionObject`.
#[inline]
pub unsafe fn w_exception_get_strerror(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_ExceptionObject)).w_strerror }
}

/// `interp_exceptions.py:740 readwrite_attrproperty_w('w_strerror', ...)`
/// — `e.strerror = ...` writer.
///
/// # Safety
/// `obj` must point to a valid `W_ExceptionObject`.
#[inline]
pub unsafe fn w_exception_set_strerror(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_ExceptionObject)).w_strerror = value;
    }
}

/// `interp_exceptions.py:741 readwrite_attrproperty_w('w_filename', ...)`
/// — `e.filename` reader.
///
/// # Safety
/// `obj` must point to a valid `W_ExceptionObject`.
#[inline]
pub unsafe fn w_exception_get_filename(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_ExceptionObject)).w_filename }
}

/// `interp_exceptions.py:741 readwrite_attrproperty_w('w_filename', ...)`
/// — `e.filename = ...` writer.
///
/// # Safety
/// `obj` must point to a valid `W_ExceptionObject`.
#[inline]
pub unsafe fn w_exception_set_filename(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_ExceptionObject)).w_filename = value;
    }
}

/// `interp_exceptions.py:742 readwrite_attrproperty_w('w_filename2', ...)`
/// — `e.filename2` reader.
///
/// # Safety
/// `obj` must point to a valid `W_ExceptionObject`.
#[inline]
pub unsafe fn w_exception_get_filename2(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_ExceptionObject)).w_filename2 }
}

/// `interp_exceptions.py:742 readwrite_attrproperty_w('w_filename2', ...)`
/// — `e.filename2 = ...` writer.
///
/// # Safety
/// `obj` must point to a valid `W_ExceptionObject`.
#[inline]
pub unsafe fn w_exception_set_filename2(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_ExceptionObject)).w_filename2 = value;
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
/// hatch.  The `W_ExceptionObject` lives forever (`malloc_typed` is
/// `Box::into_raw` today; future GC integration must root it).
pub fn memory_error_singleton() -> PyObjectRef {
    static MEMORY_ERROR_SINGLETON: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MEMORY_ERROR_SINGLETON.get_or_init(|| w_exception_new(ExcKind::MemoryError, "") as usize)
        as PyObjectRef
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
    static INSTANCES: [std::sync::OnceLock<usize>; EXC_KIND_COUNT] =
        [const { std::sync::OnceLock::new() }; EXC_KIND_COUNT];
    let slot = &INSTANCES[kind as u8 as usize];
    *slot.get_or_init(|| w_exception_new(kind, "") as usize) as PyObjectRef
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
    unsafe { ll_isinstance(obj, &EXCEPTION_TYPE) }
}

/// Get the exception kind tag.
///
/// # Safety
/// `obj` must point to a valid `W_ExceptionObject`.
#[inline]
pub unsafe fn w_exception_get_kind(obj: PyObjectRef) -> ExcKind {
    unsafe { (*(obj as *const W_ExceptionObject)).kind }
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
        ExcKind::SystemError => "SystemError",
        ExcKind::LookupError => "LookupError",
        ExcKind::UnicodeError => "UnicodeError",
        ExcKind::UnicodeTranslateError => "UnicodeTranslateError",
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
        "LookupError" => Some(ExcKind::LookupError),
        "UnicodeError" => Some(ExcKind::UnicodeError),
        "UnicodeTranslateError" => Some(ExcKind::UnicodeTranslateError),
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
                crate::strobject::w_str_get_wtf8(arg0),
                Wtf8::new("bad value")
            );
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
            ExcKind::IndexError,
            ExcKind::KeyError,
            ExcKind::AttributeError,
            ExcKind::RuntimeError,
            ExcKind::StopIteration,
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
            ExcKind::LookupError,
            ExcKind::UnicodeError,
            ExcKind::UnicodeTranslateError,
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
    fn w_exception_gc_type_id_matches_descr() {
        assert_eq!(W_EXCEPTION_GC_TYPE_ID, 31);
        assert_eq!(
            <W_ExceptionObject as crate::lltype::GcType>::type_id(),
            W_EXCEPTION_GC_TYPE_ID
        );
        assert_eq!(
            <W_ExceptionObject as crate::lltype::GcType>::SIZE,
            W_EXCEPTION_OBJECT_SIZE
        );
    }
}
