use pyre_object::PyObjectRef;
use pyre_object::interp_exceptions::{ExcKind, exc_kind_name, w_exception_new};
use ruff_text_size::Ranged;
use rustpython_compiler::{ast, parser};
use std::io::Write;
use std::sync::OnceLock;

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

    /// pypy/interpreter/error.py:180-249 `normalize_exception`.
    ///
    /// ```python
    /// def normalize_exception(self, space):
    ///     w_type = self.w_type
    ///     w_value = self.get_w_value(space)
    ///     if space.exception_is_valid_obj_as_class_w(w_type):
    ///         if space.is_w(w_value, space.w_None):
    ///             w_value = space.call_function(w_type)
    ///             w_type = self._exception_getclass(space, w_value)
    ///         else:
    ///             w_valuetype = space.exception_getclass(w_value)
    ///             if space.exception_issubclass_w(w_valuetype, w_type):
    ///                 w_type = w_valuetype
    ///             else:
    ///                 if space.isinstance_w(w_value, space.w_tuple):
    ///                     w_value = space.call(w_type, w_value)
    ///                 else:
    ///                     w_value = space.call_function(w_type, w_value)
    ///                 w_type = self._exception_getclass(space, w_value)
    ///         if self._application_traceback:
    ///             from pypy.interpreter.pytraceback import PyTraceback
    ///             from pypy.module.exceptions.interp_exceptions import W_BaseException
    ///             tb = self._application_traceback
    ///             if (isinstance(w_value, W_BaseException) and
    ///                 isinstance(tb, PyTraceback)):
    ///                 # traceback hasn't escaped yet
    ///                 w_value.w_traceback = tb
    ///             else:
    ///                 # traceback has escaped
    ///                 space.setattr(w_value, space.newtext("__traceback__"),
    ///                               self.get_w_traceback(space))
    ///     else:
    ///         w_inst = w_type
    ///         w_instclass = self._exception_getclass(space, w_inst)
    ///         if not space.is_w(w_value, space.w_None):
    ///             raise oefmt(space.w_TypeError, ...)
    ///         w_value = w_inst
    ///         w_type = w_instclass
    ///     self.w_type = w_type
    ///     self._w_value = w_value
    ///     return w_value
    /// ```
    ///
    /// Mutates `self` to install the normalized `(w_type, w_value)` and
    /// returns the new `w_value`.
    ///
    /// `pypy/interpreter/error.py:225-236 normalize_exception` traceback
    /// attach: the W_BaseException-typed fast path writes
    /// `w_value.w_traceback = tb` directly; the generic fallback is
    /// `space.setattr(w_value, "__traceback__", tb)`.  Pyre's
    /// W_BaseException grew the typed `w_traceback` slot, so the
    /// fast path is now reachable — `w_exception_set_traceback` is
    /// invoked when `w_value` is an exception instance, falling back
    /// to the generic setattr for anything else.
    pub fn normalize_exception(&mut self, space: PyObjectRef) -> Result<PyObjectRef, PyError> {
        let mut w_type = self.w_type;
        let mut w_value = self.get_w_value(space);
        unsafe {
            if crate::baseobjspace::exception_is_valid_obj_as_class_w(w_type) {
                if w_value.is_null() || w_value == pyre_object::w_none() {
                    // error.py:208-210 (Class, None): instantiate Class()
                    w_value = crate::baseobjspace::call_function(w_type, &[]);
                    if w_value.is_null() {
                        return Err(crate::call::take_call_error()
                            .unwrap_or_else(|| PyError::type_error("constructor failed")));
                    }
                    w_type = self._exception_getclass_from_call(w_type, w_value)?;
                } else {
                    // error.py:212 w_valuetype = space.exception_getclass(w_value)
                    let w_valuetype = crate::baseobjspace::exception_getclass(w_value);
                    if !w_valuetype.is_null()
                        && crate::baseobjspace::exception_issubclass_w(w_valuetype, w_type)
                    {
                        // error.py:213-215 (Class, inst): use inst's exact class
                        w_type = w_valuetype;
                    } else {
                        // error.py:217 if space.isinstance_w(w_value, space.w_tuple):
                        let w_tuple_cls =
                            crate::typedef::gettypeobject(&pyre_object::pyobject::TUPLE_TYPE);
                        if !w_tuple_cls.is_null()
                            && crate::baseobjspace::isinstance_w(w_value, w_tuple_cls)
                        {
                            // error.py:218-220 (Class, tuple): Class(*tuple)
                            let items = pyre_object::w_tuple_items_copy_as_vec(w_value);
                            w_value = crate::baseobjspace::call_function(w_type, &items);
                        } else {
                            // error.py:221-223 (Class, x): Class(x)
                            w_value = crate::baseobjspace::call_function(w_type, &[w_value]);
                        }
                        if w_value.is_null() {
                            return Err(crate::call::take_call_error()
                                .unwrap_or_else(|| PyError::type_error("constructor failed")));
                        }
                        w_type = self._exception_getclass_from_call(w_type, w_value)?;
                    }
                }
                // error.py:225-236 traceback attach — fast path writes
                // `w_value.w_traceback = tb` when `w_value` is a
                // `W_BaseException`, otherwise falls through to the
                // generic `space.setattr(w_value, "__traceback__", tb)`.
                if let Some(tb) = self._application_traceback {
                    if pyre_object::is_exception(w_value) {
                        pyre_object::interp_exceptions::w_exception_set_traceback(w_value, tb);
                    } else {
                        crate::baseobjspace::setattr_str(w_value, "__traceback__", tb)?;
                    }
                }
            } else {
                // error.py:238-245 (inst, None) — `raise inst`
                let w_inst = w_type;
                let w_instclass = self._exception_getclass(w_inst)?;
                if !w_value.is_null() && w_value != pyre_object::w_none() {
                    return Err(PyError::type_error(
                        "instance exception may not have a separate value",
                    ));
                }
                w_value = w_inst;
                w_type = w_instclass;
            }
        }
        self.w_type = w_type;
        self.w_value = w_value;
        Ok(w_value)
    }

    /// pypy/interpreter/error.py:251-257 `_exception_getclass(space, w_inst, what="exceptions")`.
    ///
    /// ```python
    /// def _exception_getclass(self, space, w_inst, what="exceptions"):
    ///     w_type = space.exception_getclass(w_inst)
    ///     if not space.exception_is_valid_class_w(w_type):
    ///         raise oefmt(space.w_TypeError, ...)
    ///     return w_type
    /// ```
    fn _exception_getclass(&self, w_inst: PyObjectRef) -> Result<PyObjectRef, PyError> {
        let w_type = crate::baseobjspace::exception_getclass(w_inst);
        if w_type.is_null() || !unsafe { crate::baseobjspace::exception_is_valid_class_w(w_type) } {
            return Err(PyError::type_error(
                "exceptions must derive from BaseException",
            ));
        }
        Ok(w_type)
    }

    /// Python 3.14 exception normalization diagnostic for a class call whose
    /// `__new__` returned a non-exception.  This is the class-construction
    /// counterpart of `_exception_getclass`; direct `raise non_exception`
    /// retains the ordinary "exceptions must derive" wording.
    fn _exception_getclass_from_call(
        &self,
        w_constructor: PyObjectRef,
        w_inst: PyObjectRef,
    ) -> Result<PyObjectRef, PyError> {
        let w_type = crate::baseobjspace::exception_getclass(w_inst);
        if w_type.is_null() || !unsafe { crate::baseobjspace::exception_is_valid_class_w(w_type) } {
            let constructor = unsafe { crate::display::py_repr(w_constructor) }
                .unwrap_or_else(|_| "<exception class>".to_string());
            let returned_type = if w_type.is_null() {
                "<unknown type>".to_string()
            } else {
                unsafe { crate::display::py_repr(w_type) }
                    .unwrap_or_else(|_| "<unknown type>".to_string())
            };
            return Err(PyError::type_error(format!(
                "calling {constructor} should have returned an instance of BaseException, not {returned_type}"
            )));
        }
        Ok(w_type)
    }

    /// `pypy/interpreter/error.py:422-434 chain_exceptions` parity:
    ///
    /// ```python
    /// def chain_exceptions(self, space, context):
    ///     w_value = self.normalize_exception(space)
    ///     w_context = context.normalize_exception(space)
    ///     if not space.is_w(w_value, w_context):
    ///         if not isinstance(w_value, W_BaseException):
    ///             raise oefmt(space.w_SystemError, "not an instance of Exception: %T", w_value)
    ///         if w_value.w_context is None:
    ///             _break_context_cycle(space, w_value, w_context)
    ///             w_value.descr_setcontext(space, w_context)
    /// ```
    ///
    /// Writes flow through the typed `w_context` slot on
    /// `W_BaseException` (`pyre-object/src/interp_exceptions.rs:113-117
    /// W_BaseException class defaults`).
    pub fn chain_exceptions(
        &mut self,
        space: PyObjectRef,
        context: &mut OperationError,
    ) -> Result<(), PyError> {
        let w_value = self.normalize_exception(space)?;
        let w_context = context.normalize_exception(space)?;
        if std::ptr::eq(w_value, w_context) {
            return Ok(());
        }
        if !unsafe { pyre_object::is_exception(w_value) } {
            return Err(PyError::new(
                crate::PyErrorKind::SystemError,
                "not an instance of Exception".to_string(),
            ));
        }
        // `:432-434` — only set __context__ when it isn't already
        // stamped; mirrors CPython's `_PyErr_ChainExceptions` precedent.
        let existing = unsafe { pyre_object::interp_exceptions::w_exception_get_context(w_value) };
        if existing.is_null() {
            _break_context_cycle(w_value, w_context)?;
            unsafe { pyre_object::interp_exceptions::w_exception_set_context(w_value, w_context) };
        }
        Ok(())
    }
}

/// `pypy/interpreter/error.py:478-509 _break_context_cycle` parity —
/// Floyd cycle-detection over the `__context__` chain, breaking the
/// loop by writing `None` into the offending link before the new
/// `w_context` is attached.
fn _break_context_cycle(w_value: PyObjectRef, w_context: PyObjectRef) -> Result<(), PyError> {
    let mut w_rabbit = w_context;
    let mut w_tortoise = w_context;
    let mut update_tortoise_toggle = false;
    loop {
        if !unsafe { pyre_object::is_exception(w_rabbit) } {
            return Err(PyError::new(
                crate::PyErrorKind::SystemError,
                "not an instance of Exception".to_string(),
            ));
        }
        let w_next = unsafe { pyre_object::interp_exceptions::w_exception_get_context(w_rabbit) };
        if w_next.is_null() || unsafe { pyre_object::is_none(w_next) } {
            break;
        }
        if std::ptr::eq(w_next, w_value) {
            // `:497-498` — `w_rabbit.descr_setcontext(space, space.w_None)`
            // (a real None, mirroring PyPy's "internal None" → user-
            // visible None via the typed slot).
            unsafe {
                pyre_object::interp_exceptions::w_exception_set_context(
                    w_rabbit,
                    pyre_object::w_none(),
                )
            };
            break;
        }
        w_rabbit = w_next;
        if std::ptr::eq(w_rabbit, w_tortoise) {
            // `:502-503` — pre-existing cycle; don't set anything to None.
            break;
        }
        if update_tortoise_toggle {
            if !unsafe { pyre_object::is_exception(w_tortoise) } {
                return Err(PyError::new(
                    crate::PyErrorKind::RuntimeError,
                    "not an instance of Exception".to_string(),
                ));
            }
            w_tortoise =
                unsafe { pyre_object::interp_exceptions::w_exception_get_context(w_tortoise) };
        }
        update_tortoise_toggle = !update_tortoise_toggle;
    }
    Ok(())
}

/// Record `active` as `exc.__context__` when `exc` is raised while `active`
/// is the exception currently being handled — the shared primitive for both
/// explicit (`raise`) and implicit (builtin/operator) raises.  Only writes
/// when no `__context__` is already stamped (so an exception that already
/// carries one — e.g. an explicit `raise ... from` or a re-raise — is left
/// untouched) and skips self-context; any cycle through the context chain is
/// broken first via `_break_context_cycle`.
pub fn chain_context(exc: PyObjectRef, active: PyObjectRef) {
    if exc.is_null()
        || active.is_null()
        || std::ptr::eq(exc, active)
        || unsafe { pyre_object::is_none(active) }
        || !unsafe { pyre_object::is_exception(exc) }
        || !unsafe { pyre_object::is_exception(active) }
    {
        return;
    }
    let existing = unsafe { pyre_object::interp_exceptions::w_exception_get_context(exc) };
    if existing.is_null() {
        let _ = _break_context_cycle(exc, active);
        unsafe { pyre_object::interp_exceptions::w_exception_set_context(exc, active) };
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
            attach_tb: true,
            reraise_lasti: -1,
            w_name_context: std::ptr::null_mut(),
            w_obj_context: std::ptr::null_mut(),
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
    /// Cached W_BaseException pointer — reused by to_exc_object()
    /// to avoid re-allocating an exception object that already exists.
    pub exc_object: PyObjectRef,
    /// `pypy/interpreter/pyopcode.py:122 handle_operation_error(..., attach_tb=True)`
    /// parity.  RERAISE opcode (`pyopcode.py:1348-1376 RERAISE`)
    /// surfaces the operror as `RaiseWithExplicitTraceback` which
    /// PyPy's bytecode dispatcher routes through
    /// `handle_operation_error(attach_tb=False)` to avoid stamping a
    /// spurious traceback frame on the cleanup path.  Pyre routes the
    /// same intent through this field: the `reraise` opcode
    /// (`eval.rs::reraise`) clears `attach_tb` so the
    /// re-fired `handle_exception` skips
    /// `record_application_traceback`.  Default `true` for any
    /// normally-constructed PyError; reraise flips it off.
    pub attach_tb: bool,
    /// `pypy/interpreter/pyopcode.py:122 handle_operation_error(..., reraise_lasti=-1)`
    /// parity.  RERAISE N reads the original raise-site lasti from the
    /// value stack and carries it through `RaiseWithExplicitTraceback`
    /// so the next exception-table dispatch can push the original
    /// raise-site offset (not the RERAISE instruction itself) as the
    /// `lasti` value, and so the no-handler propagation path can
    /// restore `last_instr` for correct `f_lineno`.  `-1` means "no
    /// reraise lasti carried" (default for primary raises).
    pub reraise_lasti: i32,
    /// The `name` context attribute for a freshly raised NameError or
    /// AttributeError (Python 3.10+): the undefined name (NameError) or
    /// the failed attribute name (AttributeError).  Exception
    /// materialisation is deferred to `to_exc_object`, so the raise site
    /// records the context here (`PY_NULL` = unset) and `to_exc_object`
    /// stamps it onto the built instance — the lazy equivalent of
    /// setting `exc.name` right after the raise
    /// (`_PyEval_FormatExcCheckArg` / `set_attribute_error_context`).
    pub w_name_context: PyObjectRef,
    /// The `obj` context attribute for a freshly raised AttributeError
    /// (Python 3.10+): the object whose attribute lookup failed.
    /// Carried and applied alongside `w_name_context`; `PY_NULL` = unset.
    pub w_obj_context: PyObjectRef,
}

impl PyError {
    /// Forward the up-to-three GC-managed references a `PyError` holds — the
    /// cached exception object and the lazy NameError/AttributeError name/obj
    /// context — to a root-walk visitor. The precise collector does not reach
    /// these through the Rust struct, so any `PyError` parked in a TLS slot
    /// across a collection must be walked through here. Each non-null slot is
    /// forwarded BY ADDRESS so the visitor relocates a moved child in place;
    /// the lazy-null `exc_object` is never materialised. `PyErrorKind` carries
    /// no object payload, so these three fields are the complete GC-ref set.
    pub fn walk_gc_refs(&mut self, visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
        let mut forward = |r: &mut PyObjectRef| {
            if !r.is_null() {
                // SAFETY: `PyObjectRef` and `GcRef` are layout-compatible.
                unsafe { visitor(&mut *(r as *mut PyObjectRef as *mut majit_ir::GcRef)) };
            }
        };
        forward(&mut self.exc_object);
        forward(&mut self.w_name_context);
        forward(&mut self.w_obj_context);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PyErrorKind {
    TypeError,
    ValueError,
    ZeroDivisionError,
    NameError,
    UnboundLocalError,
    IndexError,
    KeyError,
    AttributeError,
    RuntimeError,
    StopIteration,
    StopAsyncIteration,
    OverflowError,
    ArithmeticError,
    ImportError,
    /// Subclass of ImportError raised when a module cannot be found.
    ModuleNotFoundError,
    NotImplementedError,
    AssertionError,
    /// Raised by `_weakref` when a proxy is dereferenced after the
    /// referent has been collected.
    /// pypy/module/_weakref/interp__weakref.py:347
    /// `oefmt(space.w_ReferenceError, "weakly referenced object no longer exists")`.
    ReferenceError,
    /// BaseException subclass — raised inside generators by gen.close().
    /// Not a subclass of Exception, so `except Exception` does not catch it.
    GeneratorExit,
    RecursionError,
    /// Internal: RETURN_GENERATOR unwind signal (not a real exception).
    /// Carries the generator PyObjectRef as message.
    GeneratorReturn,
    /// Internal parity marker for pypy/interpreter/pycode.py:25
    /// `BytecodeCorruption`. This should not be raised as a user-level
    /// Python exception; it signals malformed bytecode in the interpreter.
    BytecodeCorruption,
    /// Base class for all operating-system errors.
    /// pypy/module/exceptions/interp_exceptions.py W_OSError.
    OSError,
    /// Subclass of OSError raised when a file or directory is not found.
    FileNotFoundError,
    /// pypy/interpreter/baseobjspace.py:419-420 DescrMismatch.
    ///
    /// ```python
    /// class DescrMismatch(Exception):
    ///     pass
    /// ```
    ///
    /// Internal control-flow exception raised by `space.descr_self_interp_w`
    /// when a descriptor's typecheck wrapper sees an instance of the wrong
    /// class. Caught by `GetSetProperty.descr_property_get/set/del` which
    /// then re-raises a user-visible TypeError via `descr_call_mismatch`.
    DescrMismatch,
    /// Raised by sys.exit(). Not a subclass of Exception.
    SystemExit,
    /// rpython/jit/metainterp/compile.py:1090 `memory_error = MemoryError()`
    /// — raised by `PropagateExceptionDescr.handle_fail` when a JIT
    /// malloc helper returns NULL (true OOM).  Not raised by user code
    /// in pyre; the runtime allocator never returns NULL except on
    /// genuine system OOM.
    MemoryError,
    /// `pypy/module/exceptions/interp_exceptions.py W_SystemError` —
    /// PyPy uses this for interpreter-internal invariant violations
    /// (e.g. `chain_exceptions` rejecting non-BaseException context
    /// at `error.py:429`).
    SystemError,
    /// Internal JIT trace-abort signal. This is not a Python exception
    /// class and must be intercepted before user-visible error handling.
    TraceAbort,
    /// `pypy/module/exceptions/interp_exceptions.py:474 W_LookupError`
    /// — intermediate parent of IndexError / KeyError.  Distinct
    /// PyErrorKind variant so `from_exc_object(LookupError)` /
    /// `to_exc_object()` / `render_exception` round-trip preserves
    /// the exact class rather than collapsing it onto IndexError or
    /// Exception.
    LookupError,
    /// `pypy/module/exceptions/interp_exceptions.py:418 W_UnicodeError`
    /// — intermediate parent of UnicodeDecodeError / UnicodeEncodeError,
    /// itself a subclass of ValueError.
    UnicodeError,
    UnicodeDecodeError,
    UnicodeEncodeError,
    /// `pypy/module/exceptions/interp_exceptions.py:426
    /// W_UnicodeTranslateError` — subclass of UnicodeError.
    /// Identity-only port: dedicated PyErrorKind / ExcKind / PyType
    /// so render_exception preserves the class.  W_UnicodeTranslateError's
    /// 5-arg `(object, start, end, reason)` `__init__` and custom
    /// `__str__` are TODO.
    UnicodeTranslateError,
    /// `pypy/module/exceptions/interp_exceptions.py W_SyntaxError` —
    /// raised by `compile`/`exec`/`eval`/`ast.parse` on malformed source.
    /// Identity-only port (dedicated PyErrorKind / ExcKind / PyType); the
    /// `msg`/`filename`/`lineno`/`offset`/`text` slots are TODO.
    SyntaxError,
    /// Raised when a buffer-related operation cannot proceed, e.g.
    /// `PyByteArray_Resize` resizing a `bytearray` whose storage backs a
    /// live `memoryview` ("Existing exports of data: object cannot be
    /// re-sized").  Direct subclass of Exception.
    BufferError,
}

impl PyError {
    pub fn new(kind: PyErrorKind, message: impl Into<String>) -> Self {
        PyError {
            kind,
            message: message.into(),
            exc_object: std::ptr::null_mut(),
            attach_tb: true,
            reraise_lasti: -1,
            w_name_context: std::ptr::null_mut(),
            w_obj_context: std::ptr::null_mut(),
        }
    }

    pub fn type_error(msg: impl Into<String>) -> Self {
        Self::new(PyErrorKind::TypeError, msg)
    }

    pub fn attribute_error(msg: impl Into<String>) -> Self {
        Self::new(PyErrorKind::AttributeError, msg)
    }

    /// `set_attribute_error_context` parity — an AttributeError raised by
    /// a failed attribute lookup, carrying the failed attribute `name`
    /// and the `obj` it was looked up on so `e.name` / `e.obj` read back
    /// once the instance is materialised (Python 3.10+).
    pub fn attribute_error_with_context(
        msg: impl Into<String>,
        w_obj: PyObjectRef,
        name: &str,
    ) -> Self {
        let mut err = Self::new(PyErrorKind::AttributeError, msg);
        err.w_name_context = pyre_object::w_str_new(name);
        err.w_obj_context = w_obj;
        err
    }

    pub fn value_error(msg: impl Into<String>) -> Self {
        Self::new(PyErrorKind::ValueError, msg)
    }

    pub fn syntax_error(msg: impl Into<String>) -> Self {
        Self::new(PyErrorKind::SyntaxError, msg)
    }

    /// Build a compile-time SyntaxError carrying the parser location so
    /// `e.msg` / `e.filename` / `e.lineno` / `e.offset` / `e.text` /
    /// `e.end_lineno` / `e.end_offset` read back once the instance is
    /// materialised.  `args` becomes `(msg, (filename, lineno, offset,
    /// text, end_lineno, end_offset))`, the shape
    /// `interp_exceptions.py:836 W_SyntaxError.descr_init` stores and
    /// `syntax_error_attr` / `descr_str` read from.  `text` is the
    /// offending source line (`None` → `text` reads back as `None`).
    #[allow(clippy::too_many_arguments)]
    pub fn syntax_error_located(
        msg: impl Into<String>,
        filename: &str,
        lineno: i64,
        offset: i64,
        end_lineno: i64,
        end_offset: i64,
        text: Option<&str>,
    ) -> Self {
        let message = msg.into();
        // Root the fresh exception across the args/details allocations: `exc`
        // lives only in this Rust local while `w_str_new` / `w_int_new` /
        // `w_tuple_new` / `w_list_new` run, so a collection there could sweep
        // the unrooted exception before `w_exception_set_args` writes it.
        let _roots = pyre_object::gc_roots::push_roots();
        let exc = w_exception_new(ExcKind::SyntaxError, &message);
        pyre_object::gc_roots::pin_root(exc);
        // Build the `(filename, lineno, offset, text, end_lineno, end_offset)`
        // details tuple element by element. Each is pinned as it is created and
        // reloaded before the tuple is assembled: an int / str allocation for a
        // later element can trigger a minor collection that relocates an earlier
        // young element (the filename or text string, or an uncached line/col
        // int), leaving its raw local pointing at the old address.
        let filename_slot = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(pyre_object::w_str_new(filename));
        let lineno_slot = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(pyre_object::w_int_new(lineno));
        let offset_slot = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(pyre_object::w_int_new(offset));
        let text_slot = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(match text {
            Some(t) => pyre_object::w_str_new(t),
            None => pyre_object::w_none(),
        });
        let end_lineno_slot = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(pyre_object::w_int_new(end_lineno));
        let end_offset_slot = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(pyre_object::w_int_new(end_offset));
        let details = pyre_object::w_tuple_new(vec![
            pyre_object::gc_roots::shadow_stack_get(filename_slot),
            pyre_object::gc_roots::shadow_stack_get(lineno_slot),
            pyre_object::gc_roots::shadow_stack_get(offset_slot),
            pyre_object::gc_roots::shadow_stack_get(text_slot),
            pyre_object::gc_roots::shadow_stack_get(end_lineno_slot),
            pyre_object::gc_roots::shadow_stack_get(end_offset_slot),
        ]);
        // Root the details tuple across the message allocation below before it
        // becomes `args[1]`.
        let details_slot = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(details);
        let w_msg = pyre_object::w_str_new(&message);
        let details = pyre_object::gc_roots::shadow_stack_get(details_slot);
        let args_list = pyre_object::w_list_new(vec![w_msg, details]);
        unsafe { pyre_object::interp_exceptions::w_exception_set_args(exc, args_list) };
        PyError {
            kind: PyErrorKind::SyntaxError,
            // Leave the display message empty so `message_text` derives it from
            // `exc_object` via the SyntaxError `descr_str`, which appends the
            // `(filename, line N)` suffix.
            message: String::new(),
            exc_object: exc,
            attach_tb: true,
            reraise_lasti: -1,
            w_name_context: std::ptr::null_mut(),
            w_obj_context: std::ptr::null_mut(),
        }
    }

    pub fn zero_division(msg: impl Into<String>) -> Self {
        Self::new(PyErrorKind::ZeroDivisionError, msg)
    }

    pub fn overflow_error(msg: impl Into<String>) -> Self {
        Self::new(PyErrorKind::OverflowError, msg)
    }

    pub fn runtime_error(msg: impl Into<String>) -> Self {
        Self::new(PyErrorKind::RuntimeError, msg)
    }

    pub fn not_implemented(msg: impl Into<String>) -> Self {
        Self::new(PyErrorKind::NotImplementedError, msg)
    }

    /// `_PyEval_FormatExcCheckArg` parity — a NameError carrying the
    /// undefined `name` so `e.name` reads back once the instance is
    /// materialised (Python 3.10+).
    pub fn name_error_with_name(msg: impl Into<String>, name: &str) -> Self {
        let mut err = Self::new(PyErrorKind::NameError, msg);
        err.w_name_context = pyre_object::w_str_new(name);
        err
    }

    /// UnboundLocalError(NameError) carrying the undefined local name.
    pub fn unbound_local_error_with_name(msg: impl Into<String>, name: &str) -> Self {
        let mut err = Self::new(PyErrorKind::UnboundLocalError, msg);
        err.w_name_context = pyre_object::w_str_new(name);
        err
    }

    /// A ModuleNotFoundError carrying the unresolved module `name` so
    /// `e.name` reads back once the instance is materialised — the import
    /// machinery raises `ModuleNotFoundError(msg, name=fullname)`.  The
    /// `name` rides the shared `w_n` slot (ImportError / NameError /
    /// AttributeError), stamped by `to_exc_object`.
    pub fn module_not_found_with_name(msg: impl Into<String>, name: &str) -> Self {
        let mut err = Self::new(PyErrorKind::ModuleNotFoundError, msg);
        err.w_name_context = pyre_object::w_str_new(name);
        err
    }

    pub fn internal_trace_abort(reason: impl Into<String>) -> Self {
        let mut err = Self::new(PyErrorKind::TraceAbort, reason);
        err.attach_tb = false;
        err
    }

    pub fn key_error(msg: impl Into<String>) -> Self {
        Self::new(PyErrorKind::KeyError, msg)
    }

    /// `baseobjspace.py:1284 raise_key_error(w_key)` parity — builds
    /// `KeyError(w_key)` with the key itself (not a stringified copy)
    /// as args[0], so callers reading `e.args[0]` get back the missing
    /// key object.  The display message is the key's repr, matching
    /// what `KeyError.__str__` yields.
    pub fn key_error_with_key(key: PyObjectRef) -> Self {
        // Root the caller's key and the fresh exception across the repr,
        // exception, and args allocations below: they live only in Rust
        // locals, which the precise collector does not scan, so a collection
        // inside `py_repr` / `w_exception_new` / `w_list_new` could sweep the
        // key or the exception before `w_exception_set_args` stamps them.
        // The key is pinned before `py_repr` because that repr itself
        // allocates.
        let _roots = pyre_object::gc_roots::push_roots();
        let key_slot = pyre_object::gc_roots::shadow_stack_len();
        if !key.is_null() {
            pyre_object::gc_roots::pin_root(key);
        }
        let message = if key.is_null() {
            "<null>".to_string()
        } else {
            unsafe { crate::display::py_repr(key) }
                .unwrap_or_else(|_| "<unrepresentable>".to_string())
        };
        let exc = pyre_object::interp_exceptions::w_exception_new(ExcKind::KeyError, &message);
        pyre_object::gc_roots::pin_root(exc);
        if !key.is_null() {
            // Reload the key after the repr / exception allocations: the pin
            // keeps it alive, but a minor collection may have relocated the
            // young key, leaving this raw local pointing at the old address.
            let key = pyre_object::gc_roots::shadow_stack_get(key_slot);
            let args_list = pyre_object::w_list_new(vec![key]);
            unsafe { pyre_object::interp_exceptions::w_exception_set_args(exc, args_list) };
        }
        PyError {
            kind: PyErrorKind::KeyError,
            message,
            exc_object: exc,
            attach_tb: true,
            reraise_lasti: -1,
            w_name_context: std::ptr::null_mut(),
            w_obj_context: std::ptr::null_mut(),
        }
    }

    pub fn index_error(msg: impl Into<String>) -> Self {
        Self::new(PyErrorKind::IndexError, msg)
    }

    pub fn lookup_error(msg: impl Into<String>) -> Self {
        Self::new(PyErrorKind::LookupError, msg)
    }

    pub fn os_error(msg: impl Into<String>) -> Self {
        Self::new(PyErrorKind::OSError, msg)
    }

    /// Rust's OS-error `Display` is `"{strerror} (os error {errno})"`;
    /// recover the bare strerror the way `rposix.strerror` reports it.
    fn clean_strerror(errno: i32) -> String {
        // `from_raw_os_error` reads its argument as a Win32 code on Windows,
        // where these errnos are the POSIX ones `io_error_posix_errno`
        // translated to, so it would describe an unrelated system error (17 is
        // EEXIST but also ERROR_NOT_SAME_DEVICE).  The C runtime keeps the
        // errno-keyed table `strerror` reports from.
        #[cfg(windows)]
        {
            let msg = unsafe { libc::strerror(errno) };
            if !msg.is_null() {
                return unsafe { std::ffi::CStr::from_ptr(msg) }
                    .to_string_lossy()
                    .into_owned();
            }
        }
        let full = std::io::Error::from_raw_os_error(errno).to_string();
        match full.rfind(" (os error ") {
            Some(idx) => full[..idx].to_string(),
            None => full,
        }
    }

    /// `error.py:_wrap_oserror2` — build the OSError a failed syscall raises:
    /// `args = (errno, strerror)`, the errno-specific subclass (`ERRNO_MAP`),
    /// the clean strerror, and the `.errno` / `.strerror` / `.filename` slots.
    /// `w_filename` is the affected path, or `PY_NULL` for none.
    pub fn os_error_syscall(errno: i32, w_filename: PyObjectRef) -> Self {
        Self::os_error_syscall2(errno, w_filename, pyre_object::PY_NULL)
    }

    /// `os_error_syscall` for the two-path syscalls (`rename`, `replace`,
    /// `link`, `symlink`): `w_filename2` becomes the `.filename2` slot, which
    /// `str(e)` renders as the `" -> 'dest'"` suffix.
    pub fn os_error_syscall2(
        errno: i32,
        w_filename: PyObjectRef,
        w_filename2: PyObjectRef,
    ) -> Self {
        let strerror = Self::clean_strerror(errno);
        let subclass = crate::builtins::os_error_errno_subclass(errno as i64);
        // The dedicated FileNotFoundError kind carries its own str/repr; the
        // other errno subclasses share OSError's and differ only by class.
        let (kind, exc_kind) = if matches!(subclass, Some("FileNotFoundError")) {
            (PyErrorKind::FileNotFoundError, ExcKind::FileNotFoundError)
        } else {
            (PyErrorKind::OSError, ExcKind::OSError)
        };
        let message = format!("[Errno {errno}] {strerror}");
        // Root the caller's filename and the fresh exception across the
        // exception and args allocations below: they live only in Rust locals,
        // which the precise collector does not scan, so a collection inside
        // `w_exception_new` / `w_list_new` could sweep them before the setters
        // stamp them onto `exc`.
        let _roots = pyre_object::gc_roots::push_roots();
        let pin = |obj: PyObjectRef| -> Option<usize> {
            if obj.is_null() {
                return None;
            }
            let slot = pyre_object::gc_roots::shadow_stack_len();
            pyre_object::gc_roots::pin_root(obj);
            Some(slot)
        };
        let filename_slot = pin(w_filename);
        let filename2_slot = pin(w_filename2);
        let exc = w_exception_new(exc_kind, &message);
        pyre_object::gc_roots::pin_root(exc);
        // Retag to the errno subclass through `w_class`, like
        // `os_error_family_new`: the subclasses share OSError's layout, so
        // the ExcKind stays OSError and only the class differs.
        if let Some(w_target) = subclass.and_then(crate::builtins::lookup_exc_class) {
            unsafe {
                (*(exc as *mut pyre_object::PyObject)).w_class = w_target;
            }
        }
        let args_list = pyre_object::w_list_new(vec![
            pyre_object::w_int_new(errno as i64),
            pyre_object::w_str_new(&strerror),
        ]);
        unsafe {
            pyre_object::interp_exceptions::w_exception_set_args(exc, args_list);
            pyre_object::interp_exceptions::w_exception_set_errno(
                exc,
                pyre_object::w_int_new(errno as i64),
            );
            pyre_object::interp_exceptions::w_exception_set_strerror(
                exc,
                pyre_object::w_str_new(&strerror),
            );
            // Reload the paths after the args allocations: the pins keep them
            // alive, but a minor collection may have relocated the young
            // strings, leaving the raw locals pointing at the old addresses.
            if let Some(slot) = filename_slot {
                let w_filename = pyre_object::gc_roots::shadow_stack_get(slot);
                pyre_object::interp_exceptions::w_exception_set_filename(exc, w_filename);
            }
            if let Some(slot) = filename2_slot {
                let w_filename2 = pyre_object::gc_roots::shadow_stack_get(slot);
                pyre_object::interp_exceptions::w_exception_set_filename2(exc, w_filename2);
            }
        }
        PyError {
            kind,
            // Leave the display message empty so `message_text` derives it from
            // `exc_object` via `W_OSError.descr_str`, which appends the
            // `: 'filename'` suffix; the bare "[Errno N] strerror" would bypass
            // it and the uncaught-traceback header would drop the filename.
            message: String::new(),
            exc_object: exc,
            attach_tb: true,
            reraise_lasti: -1,
            w_name_context: std::ptr::null_mut(),
            w_obj_context: std::ptr::null_mut(),
        }
    }

    /// Raise the structured OSError for `errno` (no filename). The caller's
    /// platform message is superseded by the errno-derived strerror.
    pub fn os_error_with_errno(errno: i32, _msg: impl Into<String>) -> Self {
        Self::os_error_syscall(errno, pyre_object::PY_NULL)
    }

    /// Raise an OSError carrying the C-level `(errno, strerror)` pair,
    /// matching `OSError.__init__`'s 2-argument form: `args` becomes
    /// `(errno, strerror)`, `str(e)` is `"[Errno N] strerror"`, and
    /// `e.errno` / `e.strerror` read back the two values.  ENOENT keeps
    /// the FileNotFoundError subclass mapping of `os_error_with_errno`.
    pub fn os_error_errno_strerror(errno: i32, strerror: impl Into<String>) -> Self {
        let strerror = strerror.into();
        let kind = if errno == 2 {
            PyErrorKind::FileNotFoundError
        } else {
            PyErrorKind::OSError
        };
        let exc_kind = if errno == 2 {
            ExcKind::FileNotFoundError
        } else {
            ExcKind::OSError
        };
        let message = format!("[Errno {errno}] {strerror}");
        // Root the fresh exception across the args allocation below: `exc`
        // lives only in this Rust local while `w_int_new` / `w_str_new` /
        // `w_list_new` run, so a collection there could sweep the unrooted
        // exception before `w_exception_set_args` writes through it.
        let _roots = pyre_object::gc_roots::push_roots();
        let exc = w_exception_new(exc_kind, &message);
        pyre_object::gc_roots::pin_root(exc);
        let args_list = pyre_object::w_list_new(vec![
            pyre_object::w_int_new(errno as i64),
            pyre_object::w_str_new(&strerror),
        ]);
        unsafe { pyre_object::interp_exceptions::w_exception_set_args(exc, args_list) };
        PyError {
            kind,
            message,
            exc_object: exc,
            attach_tb: true,
            reraise_lasti: -1,
            w_name_context: std::ptr::null_mut(),
            w_obj_context: std::ptr::null_mut(),
        }
    }

    /// `error.py:new_import_error` — `ImportError(msg, name=name, path=path)`.
    /// A failed `from X import Y` (pyopcode.py import_from) raises through
    /// this so `.name` (the package) and `.path` (its file) are readable,
    /// which `PyError::new(ImportError, msg)` cannot carry.
    pub fn import_error_name_path(
        msg: impl Into<String>,
        w_name: PyObjectRef,
        w_path: PyObjectRef,
    ) -> Self {
        let message = msg.into();
        // Root the caller's name/path and the fresh exception across the
        // exception and message allocations below: they live only in Rust
        // locals, which the precise collector does not scan, so a collection
        // inside `w_exception_new` / `w_str_new` could sweep them before the
        // setters stamp them onto `exc`.
        let _roots = pyre_object::gc_roots::push_roots();
        let name_slot = pyre_object::gc_roots::shadow_stack_len();
        if !w_name.is_null() {
            pyre_object::gc_roots::pin_root(w_name);
        }
        let path_slot = pyre_object::gc_roots::shadow_stack_len();
        if !w_path.is_null() {
            pyre_object::gc_roots::pin_root(w_path);
        }
        let exc = w_exception_new(ExcKind::ImportError, &message);
        pyre_object::gc_roots::pin_root(exc);
        // `ImportError.__init__` mirrors args[0] into the dedicated `msg`
        // slot; the prebuilt-instance path bypasses it, so stamp it here.
        let w_msg = if message.is_empty() {
            pyre_object::w_none()
        } else {
            pyre_object::w_str_new(&message)
        };
        // Reload name/path after the message allocation: the pins keep them
        // alive, but a minor collection may have relocated the young objects,
        // leaving these raw locals stale. A null (absent) ref was never pinned,
        // so it has no slot and stays null.
        let w_name = if w_name.is_null() {
            w_name
        } else {
            pyre_object::gc_roots::shadow_stack_get(name_slot)
        };
        let w_path = if w_path.is_null() {
            w_path
        } else {
            pyre_object::gc_roots::shadow_stack_get(path_slot)
        };
        unsafe {
            pyre_object::interp_exceptions::w_exception_set_import_msg(exc, w_msg);
            pyre_object::interp_exceptions::w_exception_set_name(exc, w_name);
            pyre_object::interp_exceptions::w_exception_set_import_path(exc, w_path);
        }
        PyError {
            kind: PyErrorKind::ImportError,
            message,
            exc_object: exc,
            attach_tb: true,
            reraise_lasti: -1,
            w_name_context: std::ptr::null_mut(),
            w_obj_context: std::ptr::null_mut(),
        }
    }

    /// pypy/module/_weakref/interp__weakref.py:347 — raised by `force()`
    /// when the referent of a proxy is no longer alive.
    pub fn reference_error(msg: impl Into<String>) -> Self {
        Self::new(PyErrorKind::ReferenceError, msg)
    }

    pub fn recursion_error(msg: impl Into<String>) -> Self {
        Self::new(PyErrorKind::RecursionError, msg)
    }

    /// rpython/jit/metainterp/compile.py:1090 `memory_error = MemoryError()`
    /// — module-level singleton instance the JIT raises through
    /// `PropagateExceptionDescr.handle_fail` when a malloc helper
    /// returns NULL.
    pub fn memory_error(msg: impl Into<String>) -> Self {
        Self::new(PyErrorKind::MemoryError, msg)
    }

    pub fn stop_iteration() -> Self {
        Self::new(PyErrorKind::StopIteration, String::new())
    }

    pub fn stop_async_iteration() -> Self {
        Self::new(PyErrorKind::StopAsyncIteration, String::new())
    }

    /// Convert to a W_BaseException for pushing onto the value stack.
    /// Reuses the cached object from from_exc_object() if available, otherwise
    /// materialises it once and memoises it into `self.exc_object` so repeat
    /// calls return the same instance (`get_w_value` write-once, error.py:349).
    ///
    /// Mirrors `pypy/interpreter/error.py:OperationError.get_w_value`'s
    /// upgrade-to-exception-instance path: when the OperationError
    /// carries a raw message (the `oefmt` shape), the materialised
    /// exception instance gets `args = (msg,)` per
    /// `pypy/module/exceptions/interp_exceptions.py:123-124
    /// W_BaseException.descr_init` — `self.args_w = args_w`.  Pyre
    /// stores `args_w` as a `W_ListObject`, so we stamp a one-element
    /// list `[msg_str]` here so `str(e)` and `repr(e)` and
    /// `e.args == (msg,)` all line up with PyPy.
    pub fn to_exc_object(&mut self) -> PyObjectRef {
        if !self.exc_object.is_null() {
            return self.exc_object;
        }
        // Root the deferred name/obj context references across the exception
        // allocation below: they live only in this Rust `PyError`, which the
        // precise collector does not scan, so a collection inside
        // `w_exception_new` could sweep them before they are stamped onto `exc`
        // at the `set_name` / `set_attr_obj` calls.
        let _roots = pyre_object::gc_roots::push_roots();
        let name_ctx_slot = pyre_object::gc_roots::shadow_stack_len();
        if !self.w_name_context.is_null() {
            pyre_object::gc_roots::pin_root(self.w_name_context);
        }
        let obj_ctx_slot = pyre_object::gc_roots::shadow_stack_len();
        if !self.w_obj_context.is_null() {
            pyre_object::gc_roots::pin_root(self.w_obj_context);
        }
        let exc = w_exception_new(self.to_exc_kind(), &self.message);
        // Root the fresh managed exception across the message/context stamping
        // below: `exc` lives only in this Rust local while `w_list_new` (and the
        // setters) run, so a collection there could sweep the unrooted
        // (non-moving oldgen) exception before it is written through.
        pyre_object::gc_roots::pin_root(exc);
        if !self.message.is_empty() {
            let msg_slot = pyre_object::gc_roots::shadow_stack_len();
            let msg = pyre_object::w_str_new(&self.message);
            pyre_object::gc_roots::pin_root(msg);
            let args_list = pyre_object::w_list_new(vec![msg]);
            unsafe { pyre_object::interp_exceptions::w_exception_set_args(exc, args_list) };
            // `ImportError` / `ModuleNotFoundError` expose the message through a
            // dedicated `msg` slot (`ImportError.__init__` stores `args[0]`
            // there). The raw-message raise path bypasses `__init__`, so stamp
            // `msg` here too — otherwise `e.msg` reads back `None`.
            if matches!(
                self.kind,
                PyErrorKind::ImportError | PyErrorKind::ModuleNotFoundError
            ) {
                // Reload `msg` after the args allocation: the pin keeps it
                // alive, but `w_list_new` may have relocated the young string,
                // leaving this raw local pointing at the old address.
                let msg = pyre_object::gc_roots::shadow_stack_get(msg_slot);
                unsafe { pyre_object::interp_exceptions::w_exception_set_import_msg(exc, msg) };
            }
        }
        // Stamp the deferred `name` / `obj` context onto the freshly
        // materialised NameError / AttributeError instance, the lazy
        // equivalent of `_PyEval_FormatExcCheckArg` /
        // `set_attribute_error_context` (Python 3.10+).
        // Reload the deferred context refs after the exception / args
        // allocations: the pins keep them alive, but a minor collection may
        // have relocated the young objects, leaving the stored fields stale.
        if !self.w_name_context.is_null() {
            let w_name_context = pyre_object::gc_roots::shadow_stack_get(name_ctx_slot);
            unsafe { pyre_object::interp_exceptions::w_exception_set_name(exc, w_name_context) };
        }
        if !self.w_obj_context.is_null() {
            let w_obj_context = pyre_object::gc_roots::shadow_stack_get(obj_ctx_slot);
            unsafe { pyre_object::interp_exceptions::w_exception_set_attr_obj(exc, w_obj_context) };
        }
        // Write-once memo (`get_w_value` self.w_value): cache the materialised
        // instance so a second call returns the same object (identity `e1 is e2`)
        // instead of a fresh allocation.
        self.exc_object = exc;
        exc
    }

    pub fn normalize_exception(&mut self, _space: PyObjectRef) -> Result<PyObjectRef, PyError> {
        let w_value = self.to_exc_object();
        if w_value.is_null() || !unsafe { pyre_object::is_exception(w_value) } {
            return Err(PyError::type_error("exception is not a BaseException"));
        }
        self.exc_object = w_value;
        Ok(w_value)
    }

    /// pypy/interpreter/error.py:263-317 `write_unraisable`, with the
    /// `with_traceback=False` first-line shape.
    pub fn write_unraisable(
        &mut self,
        space: PyObjectRef,
        where_desc: &str,
        w_object: PyObjectRef,
    ) {
        let w_value = self
            .normalize_exception(space)
            .unwrap_or_else(|_| self.to_exc_object());
        let mut w_type = crate::baseobjspace::exception_getclass(w_value);
        if w_type.is_null() {
            w_type = pyre_object::w_none();
        }
        let mut w_tb = if !w_value.is_null() && unsafe { pyre_object::is_exception(w_value) } {
            unsafe { pyre_object::interp_exceptions::w_exception_get_traceback(w_value) }
        } else {
            pyre_object::PY_NULL
        };
        if w_tb.is_null() {
            w_tb = pyre_object::w_none();
        }
        let mut w_object = if w_object.is_null() {
            pyre_object::w_none()
        } else {
            w_object
        };
        let mut first_line = if where_desc.is_empty() {
            String::new()
        } else if where_desc.starts_with("Exception ignored ") {
            where_desc.to_string()
        } else {
            format!("Exception ignored in: {where_desc}")
        };
        // vm.py:19-25 also carries `extra_line`; pyre's hook-args
        // structseq targets the 5-field shape, so only the default printer
        // receives the Rust-side extra line.
        let hook_args = crate::_structseq::new_instance(
            unraisable_hook_args_type(),
            vec![
                w_type,
                w_value,
                w_tb,
                if first_line.is_empty() {
                    pyre_object::w_none()
                } else {
                    pyre_object::w_str_new(&first_line)
                },
                w_object,
            ],
        );
        if let Some(sys_mod) = crate::importing::get_sys_module("sys") {
            if let Ok(w_hook) = crate::baseobjspace::getattr_str(sys_mod, "unraisablehook") {
                if !w_hook.is_null() && !unsafe { pyre_object::is_none(w_hook) } {
                    match crate::call::call_function_impl_result(w_hook, &[hook_args]) {
                        Ok(_) => return,
                        Err(mut hook_err) => {
                            first_line = "Exception ignored in sys.unraisablehook".to_string();
                            w_object = w_hook;
                            let hook_value = hook_err
                                .normalize_exception(space)
                                .unwrap_or_else(|_| hook_err.to_exc_object());
                            w_type = crate::baseobjspace::exception_getclass(hook_value);
                            if w_type.is_null() {
                                w_type = pyre_object::w_none();
                            }
                            w_tb = if !hook_value.is_null()
                                && unsafe { pyre_object::is_exception(hook_value) }
                            {
                                unsafe {
                                    pyre_object::interp_exceptions::w_exception_get_traceback(
                                        hook_value,
                                    )
                                }
                            } else {
                                pyre_object::PY_NULL
                            };
                            if w_tb.is_null() {
                                w_tb = pyre_object::w_none();
                            }
                            Self::write_unraisable_default(
                                space,
                                w_type,
                                hook_value,
                                w_tb,
                                &first_line,
                                w_object,
                                "",
                            );
                            return;
                        }
                    }
                }
            }
        }
        Self::write_unraisable_default(space, w_type, w_value, w_tb, &first_line, w_object, "");
    }

    fn write_unraisable_to_sys_stderr(buf: &[u8]) -> bool {
        let Some(sys_mod) = crate::importing::get_sys_module("sys") else {
            return false;
        };
        let stderr = match crate::baseobjspace::getattr_str(sys_mod, "stderr") {
            Ok(w) if !w.is_null() && !unsafe { pyre_object::is_none(w) } => w,
            _ => return false,
        };
        let text = String::from_utf8_lossy(buf).into_owned();
        let w_text = pyre_object::w_str_new(&text);
        let result = crate::baseobjspace::call_method(stderr, "write", &[w_text]);
        if result.is_null() {
            let _ = crate::call::take_call_error();
            return false;
        }
        true
    }

    /// pypy/interpreter/error.py:318-347 `write_unraisable_default`.
    pub fn write_unraisable_default(
        space: PyObjectRef,
        w_type: PyObjectRef,
        w_value: PyObjectRef,
        w_tb: PyObjectRef,
        first_line: &str,
        w_object: PyObjectRef,
        extra_line: &str,
    ) {
        let _ = (space, w_type);
        let mut first_line = if first_line.is_empty() {
            "Exception ignored in:".to_string()
        } else {
            first_line.to_string()
        };
        if !w_object.is_null() && !unsafe { pyre_object::is_none(w_object) } {
            let objrepr = unsafe { crate::display::py_repr(w_object) }
                .unwrap_or_else(|_| "<object repr() failed>".to_string());
            first_line = format!("{first_line} {objrepr}");
        }
        let extra_line = if extra_line.is_empty() {
            "\n".to_string()
        } else {
            format!("{extra_line}:\n")
        };
        let mut buf = Vec::new();
        let _ = write!(&mut buf, "{first_line}");
        if !extra_line.is_empty() {
            let _ = write!(&mut buf, "{extra_line}");
        }
        if !w_value.is_null() && unsafe { pyre_object::is_exception(w_value) } {
            if !w_tb.is_null() && !unsafe { pyre_object::is_none(w_tb) } {
                unsafe { pyre_object::interp_exceptions::w_exception_set_traceback(w_value, w_tb) };
            }
            let err = unsafe { PyError::from_exc_object(w_value) };
            let _ = write_exception(&mut buf, &err, true);
        }
        if !Self::write_unraisable_to_sys_stderr(&buf) {
            crate::host_seam::emit_stderr(&buf);
        }
    }

    fn to_exc_kind(&self) -> ExcKind {
        match self.kind {
            PyErrorKind::TypeError => ExcKind::TypeError,
            PyErrorKind::ValueError => ExcKind::ValueError,
            PyErrorKind::ZeroDivisionError => ExcKind::ZeroDivisionError,
            PyErrorKind::NameError => ExcKind::NameError,
            PyErrorKind::UnboundLocalError => ExcKind::UnboundLocalError,
            PyErrorKind::IndexError => ExcKind::IndexError,
            PyErrorKind::KeyError => ExcKind::KeyError,
            PyErrorKind::AttributeError => ExcKind::AttributeError,
            PyErrorKind::RuntimeError => ExcKind::RuntimeError,
            PyErrorKind::StopIteration => ExcKind::StopIteration,
            PyErrorKind::StopAsyncIteration => ExcKind::StopAsyncIteration,
            PyErrorKind::OverflowError => ExcKind::OverflowError,
            PyErrorKind::ArithmeticError => ExcKind::ArithmeticError,
            PyErrorKind::ImportError => ExcKind::ImportError,
            PyErrorKind::ModuleNotFoundError => ExcKind::ModuleNotFoundError,
            PyErrorKind::NotImplementedError => ExcKind::NotImplementedError,
            PyErrorKind::AssertionError => ExcKind::AssertionError,
            PyErrorKind::ReferenceError => ExcKind::ReferenceError,
            PyErrorKind::GeneratorExit => ExcKind::GeneratorExit,
            PyErrorKind::RecursionError => ExcKind::RecursionError,
            PyErrorKind::GeneratorReturn => ExcKind::RuntimeError,
            // Internal-only marker. If it escapes to object-space conversion,
            // degrade to RuntimeError rather than inventing a new exception type.
            PyErrorKind::BytecodeCorruption => ExcKind::RuntimeError,
            PyErrorKind::OSError => ExcKind::OSError,
            PyErrorKind::FileNotFoundError => ExcKind::FileNotFoundError,
            // DescrMismatch is a control-flow exception caught by
            // GetSetProperty.descr_property_get/set/del. If it ever escapes
            // to user code without being converted to TypeError it surfaces
            // as a TypeError, matching PyPy's eventual descr_call_mismatch.
            PyErrorKind::DescrMismatch => ExcKind::TypeError,
            PyErrorKind::SystemExit => ExcKind::SystemExit,
            PyErrorKind::MemoryError => ExcKind::MemoryError,
            PyErrorKind::SystemError => ExcKind::SystemError,
            PyErrorKind::TraceAbort => ExcKind::RuntimeError,
            PyErrorKind::LookupError => ExcKind::LookupError,
            PyErrorKind::UnicodeError => ExcKind::UnicodeError,
            PyErrorKind::UnicodeDecodeError => ExcKind::UnicodeDecodeError,
            PyErrorKind::UnicodeEncodeError => ExcKind::UnicodeEncodeError,
            PyErrorKind::UnicodeTranslateError => ExcKind::UnicodeTranslateError,
            PyErrorKind::SyntaxError => ExcKind::SyntaxError,
            PyErrorKind::BufferError => ExcKind::BufferError,
        }
    }

    /// Create a PyError from a W_BaseException.
    ///
    /// # Safety
    /// `obj` must point to a valid `W_BaseException`.
    pub unsafe fn from_exc_object(obj: PyObjectRef) -> Self {
        unsafe {
            let kind = pyre_object::interp_exceptions::w_exception_get_kind(obj);
            // The display string is derived lazily from `exc_object`
            // (`message_text`), never here: conversion runs on every
            // raise propagation, and stringifying the args eagerly would
            // execute their `__str__` at raise time instead of at
            // display time.
            PyError {
                kind: Self::kind_from_exc(kind),
                message: String::new(),
                exc_object: obj,
                attach_tb: true,
                reraise_lasti: -1,
                w_name_context: std::ptr::null_mut(),
                w_obj_context: std::ptr::null_mut(),
            }
        }
    }

    pub fn kind_from_exc(kind: ExcKind) -> PyErrorKind {
        match kind {
            ExcKind::TypeError => PyErrorKind::TypeError,
            ExcKind::ValueError => PyErrorKind::ValueError,
            ExcKind::ZeroDivisionError => PyErrorKind::ZeroDivisionError,
            ExcKind::NameError => PyErrorKind::NameError,
            ExcKind::UnboundLocalError => PyErrorKind::UnboundLocalError,
            ExcKind::IndexError => PyErrorKind::IndexError,
            ExcKind::KeyError => PyErrorKind::KeyError,
            ExcKind::AttributeError => PyErrorKind::AttributeError,
            ExcKind::RuntimeError => PyErrorKind::RuntimeError,
            ExcKind::StopIteration => PyErrorKind::StopIteration,
            ExcKind::StopAsyncIteration => PyErrorKind::StopAsyncIteration,
            ExcKind::OverflowError => PyErrorKind::OverflowError,
            ExcKind::ArithmeticError => PyErrorKind::ArithmeticError,
            ExcKind::ImportError => PyErrorKind::ImportError,
            ExcKind::ModuleNotFoundError => PyErrorKind::ModuleNotFoundError,
            ExcKind::NotImplementedError => PyErrorKind::NotImplementedError,
            ExcKind::AssertionError => PyErrorKind::AssertionError,
            ExcKind::ReferenceError => PyErrorKind::ReferenceError,
            ExcKind::GeneratorExit => PyErrorKind::GeneratorExit,
            ExcKind::RecursionError => PyErrorKind::RecursionError,
            ExcKind::BaseException | ExcKind::Exception => PyErrorKind::RuntimeError,
            ExcKind::OSError => PyErrorKind::OSError,
            ExcKind::FileNotFoundError => PyErrorKind::FileNotFoundError,
            // `pypy/module/exceptions/interp_exceptions.py:418`
            // W_UnicodeError = _new_exception('UnicodeError',
            // W_ValueError, ...).  Each Unicode subclass round-trips
            // through its own PyErrorKind variant so render_exception
            // preserves the exact class.
            ExcKind::UnicodeError => PyErrorKind::UnicodeError,
            ExcKind::UnicodeDecodeError => PyErrorKind::UnicodeDecodeError,
            ExcKind::UnicodeEncodeError => PyErrorKind::UnicodeEncodeError,
            // `pypy/module/exceptions/interp_exceptions.py:426`
            // W_UnicodeTranslateError — subclass of UnicodeError.
            ExcKind::UnicodeTranslateError => PyErrorKind::UnicodeTranslateError,
            ExcKind::SystemExit => PyErrorKind::SystemExit,
            ExcKind::MemoryError => PyErrorKind::MemoryError,
            ExcKind::SystemError => PyErrorKind::SystemError,
            // `pypy/module/exceptions/interp_exceptions.py:474`
            // W_LookupError = _new_exception('LookupError', W_Exception,
            // ...) — intermediate parent of IndexError / KeyError.
            ExcKind::LookupError => PyErrorKind::LookupError,
            ExcKind::SyntaxError => PyErrorKind::SyntaxError,
            ExcKind::BufferError => PyErrorKind::BufferError,
        }
    }

    /// The exception text without the class name.  Rust-constructed
    /// errors carry it in `message`; object-backed errors
    /// (`from_exc_object`) derive it from `exc_object` on demand —
    /// `W_BaseException.descr_str` formats `args_w`, so the args'
    /// `__str__` runs at display time, not at raise time.
    pub fn message_text(&self) -> String {
        if !self.message.is_empty() || self.exc_object.is_null() {
            return self.message.clone();
        }
        // Infallible Display-side context: a raising `__str__` degrades to
        // the placeholder rather than propagating, and a lone surrogate is
        // backslash-escaped rather than panicking.
        unsafe { crate::display::py_str_display(self.exc_object) }
    }

    pub fn render_exception(&self) -> String {
        let name = exc_object_class_name(self.exc_object)
            .unwrap_or_else(|| exc_kind_name(self.to_exc_kind()).to_string());
        let message = self.message_text();
        if message.is_empty() {
            name
        } else {
            format!("{name}: {message}")
        }
    }
}

fn unraisable_hook_args_type() -> PyObjectRef {
    thread_local! {
        static TYPE: OnceLock<PyObjectRef> = const { OnceLock::new() };
    }
    TYPE.with(|cell| {
        *cell.get_or_init(|| {
            crate::_structseq::make_struct_seq(
                "sys.UnraisableHookArgs",
                &[
                    "exc_type",
                    "exc_value",
                    "exc_traceback",
                    "err_msg",
                    "object",
                ],
            )
        })
    })
}

/// Resolve an exception instance's actual Python class name for display.
///
/// `crate::typedef::type` trusts the instance's `w_class`, which the
/// `__new__` wrapper sets to the exact class that was raised — including
/// user subclasses (`class MyErr(ValueError)`) and class-only builtins
/// such as `KeyboardInterrupt` whose `ExcKind` tag degrades to
/// `BaseException`.  Reading the class name here makes the printed header
/// match `type(e).__name__` instead of the coarse kind tag.  Returns
/// `None` for a null pointer or non-exception object so callers fall back
/// to the kind-based name.
fn exc_object_class_name(exc: PyObjectRef) -> Option<String> {
    if exc.is_null() || !unsafe { pyre_object::is_exception(exc) } {
        return None;
    }
    crate::typedef::r#type(exc).map(|tp| unsafe { pyre_object::w_type_get_name(tp).to_string() })
}

impl std::fmt::Display for PyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.render_exception())
    }
}

pub fn write_exception<W: Write>(
    writer: &mut W,
    err: &PyError,
    include_traceback: bool,
) -> std::io::Result<()> {
    if include_traceback {
        // `traceback.py:171-194` __cause__ / __context__ chain
        // printing.  Recurse into the older exception first, emit
        // the bridging banner, then print the current exception.
        if !err.exc_object.is_null() {
            write_chained_context(writer, err.exc_object)?;
        }
        writeln!(writer, "Traceback (most recent call last):")?;
        write_traceback_chain(writer, err)?;
        writeln!(writer, "{}", err.render_exception())?;
        write_exception_notes(writer, err.exc_object)
    } else {
        writeln!(writer, "{}", err.render_exception())
    }
}

/// Render an uncaught compile-time SyntaxError as the top-level banner
/// (`print_error_text`): the `File "…", line N` header, the offending
/// source line, a caret under the offending column span, and
/// `<Class>: <msg>` (the raw `msg`, not `str(e)`'s `(file, line N)`
/// suffix — the header already carries the location).  There is no
/// traceback frame chain: a malformed source never began executing.
/// Falls back to the plain header when the instance lacks structured
/// location (a location-less codegen error).
pub fn write_syntax_error<W: Write>(writer: &mut W, err: &PyError) -> std::io::Result<()> {
    let exc = err.exc_object;
    if exc.is_null() || !unsafe { pyre_object::is_exception(exc) } {
        return writeln!(writer, "{}", err.render_exception());
    }
    let attr = |name: &str| crate::baseobjspace::syntax_error_attr(exc, name);
    let str_of = |w: PyObjectRef| -> Option<String> {
        (!w.is_null() && unsafe { pyre_object::is_str(w) })
            .then(|| unsafe { pyre_object::w_str_get_value(w) }.to_string())
    };
    let int_of = |w: PyObjectRef| -> Option<i64> {
        (!w.is_null() && unsafe { pyre_object::is_int(w) })
            .then(|| unsafe { pyre_object::intobject::w_int_get_value(w) })
    };
    let filename = str_of(attr("filename"));
    let lineno = int_of(attr("lineno"));
    if let (Some(fname), Some(lineno)) = (filename.as_ref(), lineno) {
        writeln!(writer, "  File \"{fname}\", line {lineno}")?;
    }
    if let Some(text) = str_of(attr("text")) {
        let raw = text.trim_end_matches(['\n', '\r']);
        let shown = raw.trim_start();
        let lead = raw.len() - shown.len();
        writeln!(writer, "    {shown}")?;
        if let Some(offset) = int_of(attr("offset")) {
            let start_col = (offset.max(1) as usize) - 1;
            let end_col = int_of(attr("end_offset"))
                .filter(|&e| e >= offset)
                .map_or(start_col, |e| (e as usize) - 1);
            let caret_start = start_col.saturating_sub(lead);
            let caret_len = end_col.saturating_sub(start_col).max(1);
            let mut caret = String::from("    ");
            caret.push_str(&" ".repeat(caret_start));
            caret.push_str(&"^".repeat(caret_len));
            writeln!(writer, "{caret}")?;
        }
    }
    let name =
        exc_object_class_name(exc).unwrap_or_else(|| exc_kind_name(err.to_exc_kind()).to_string());
    match str_of(attr("msg")) {
        Some(msg) if !msg.is_empty() => writeln!(writer, "{name}: {msg}"),
        _ => writeln!(writer, "{name}"),
    }
}

/// Emit `write_syntax_error` through the mediated stderr seam, mirroring
/// `eprint_exception`.
pub fn eprint_syntax_error(err: &PyError) {
    let mut buf: Vec<u8> = Vec::new();
    let _ = write_syntax_error(&mut buf, err);
    crate::host_seam::emit_stderr(&buf);
}

/// `Lib/traceback.py:171-200 TracebackException._format_*` parity —
/// walk `__cause__` (or `__context__` when `__suppress_context__` is
/// False) chains and emit each older exception with the appropriate
/// bridging banner before the current one.
fn write_chained_context<W: Write>(writer: &mut W, exc: PyObjectRef) -> std::io::Result<()> {
    if exc.is_null() || !unsafe { pyre_object::is_exception(exc) } {
        return Ok(());
    }
    let cause = unsafe { pyre_object::interp_exceptions::w_exception_get_cause(exc) };
    let context = unsafe { pyre_object::interp_exceptions::w_exception_get_context(exc) };
    let suppress = unsafe { pyre_object::interp_exceptions::w_exception_get_suppress_context(exc) };

    // `traceback.py:184-191` — `__cause__` wins over `__context__`;
    // `__suppress_context__` (set by `raise X from None`) hides
    // `__context__` only when no explicit cause was attached.
    let (older, banner) = if !cause.is_null() && !unsafe { pyre_object::is_none(cause) } {
        (
            cause,
            "\nThe above exception was the direct cause of the following exception:\n",
        )
    } else if !suppress && !context.is_null() && !unsafe { pyre_object::is_none(context) } {
        (
            context,
            "\nDuring handling of the above exception, another exception occurred:\n",
        )
    } else {
        return Ok(());
    };

    write_chained_context(writer, older)?;
    write_single_exception(writer, older)?;
    writeln!(writer, "{}", banner)?;
    Ok(())
}

/// Print one W_BaseException as `Traceback ...\n<frames>\n<header>`.
/// Used by `write_chained_context` when recursing through chained
/// __cause__ / __context__ predecessors.
fn write_single_exception<W: Write>(writer: &mut W, exc: PyObjectRef) -> std::io::Result<()> {
    writeln!(writer, "Traceback (most recent call last):")?;
    write_traceback_chain_from_exc(writer, exc)?;
    let render = render_exc_object(exc);
    writeln!(writer, "{}", render)?;
    write_exception_notes(writer, exc)
}

/// `Lib/traceback.py:_format_final_exc_line` notes block — each entry of
/// `e.__notes__` is printed on its own line after the exception
/// header.  Notes are stored in the exception's instance dict as a
/// list of strings by `BaseException.add_note` (PEP 678).
fn write_exception_notes<W: Write>(writer: &mut W, exc: PyObjectRef) -> std::io::Result<()> {
    if exc.is_null() || !unsafe { pyre_object::is_exception(exc) } {
        return Ok(());
    }
    let notes = match crate::baseobjspace::getattr_str(exc, "__notes__") {
        Ok(v) if !v.is_null() => v,
        _ => return Ok(()),
    };
    if !unsafe { pyre_object::is_list(notes) } {
        return Ok(());
    }
    let len = unsafe { pyre_object::w_list_len(notes) } as i64;
    for i in 0..len {
        let item = unsafe { pyre_object::w_list_getitem(notes, i) }.unwrap_or(pyre_object::PY_NULL);
        if item.is_null() {
            continue;
        }
        let s = unsafe { crate::display::py_str(item) }
            .unwrap_or_else(|_| "<unprintable note>".to_string());
        for line in s.lines() {
            writeln!(writer, "{}", line)?;
        }
    }
    Ok(())
}

/// Compose the `ExcName: msg` header for a W_BaseException —
/// equivalent to `traceback.format_exception_only`'s last line.
fn render_exc_object(exc: PyObjectRef) -> String {
    if exc.is_null() || !unsafe { pyre_object::is_exception(exc) } {
        return String::from("<no exception>");
    }
    let name = exc_object_class_name(exc).unwrap_or_else(|| {
        let kind = unsafe { pyre_object::interp_exceptions::w_exception_get_kind(exc) };
        exc_kind_name(kind).to_string()
    });
    // `str(e)` semantically — pyre stores args as a list; the
    // first arg's str repr is the message.  Empty args produces
    // bare `ExcName` per `traceback.format_exception_only`.
    let args = unsafe { pyre_object::interp_exceptions::w_exception_get_args(exc) };
    let msg = unsafe {
        if args.is_null() || !pyre_object::is_tuple(args) {
            String::new()
        } else {
            let len = pyre_object::w_tuple_len(args);
            if len == 0 {
                String::new()
            } else if len == 1 {
                let first = pyre_object::w_tuple_getitem(args, 0).unwrap_or(pyre_object::PY_NULL);
                if first.is_null() {
                    String::new()
                } else {
                    crate::display::py_str_display(first)
                }
            } else {
                // Multi-arg exceptions render as tuple repr — matches
                // BaseException.__str__ (`interp_exceptions.py:142`).
                let items: Vec<String> = (0..len as i64)
                    .filter_map(|i| pyre_object::w_tuple_getitem(args, i))
                    .map(|w| {
                        crate::display::py_repr(w).unwrap_or_else(|_| "<unprintable>".to_string())
                    })
                    .collect();
                format!("({})", items.join(", "))
            }
        }
    };
    if msg.is_empty() {
        name
    } else {
        format!("{name}: {msg}")
    }
}

/// `pypy/interpreter/error.py:125-158 print_app_tb_only` parity —
/// walk the chained `PyTraceback` head→tail (outermost → innermost)
/// and print each frame as `File "<path>", line N, in <name>` plus the
/// source line.  Silently no-ops when `err.exc_object` is null, the
/// traceback slot is empty, or the source file can't be read.
fn write_traceback_chain<W: Write>(writer: &mut W, err: &PyError) -> std::io::Result<()> {
    if err.exc_object.is_null() {
        return Ok(());
    }
    write_traceback_chain_from_exc(writer, err.exc_object)
}

fn write_traceback_chain_from_exc<W: Write>(
    writer: &mut W,
    exc: PyObjectRef,
) -> std::io::Result<()> {
    if exc.is_null() || !unsafe { pyre_object::is_exception(exc) } {
        return Ok(());
    }
    let mut tb = unsafe { pyre_object::interp_exceptions::w_exception_get_traceback(exc) };
    while !tb.is_null() {
        if !unsafe { crate::pytraceback::is_pytraceback(tb) } {
            break;
        }
        let w_code = unsafe { crate::pytraceback::w_pytraceback_get_w_code(tb) };
        let lineno = unsafe { crate::pytraceback::w_pytraceback_get_lineno(tb) };
        let lasti = unsafe { crate::pytraceback::w_pytraceback_get_lasti(tb) };
        let (filename, funcname, location) = if w_code.is_null() {
            (String::from("<unknown>"), String::from("<unknown>"), None)
        } else {
            // `w_code` is a GC-rooted `PyCode` pointer captured
            // at `record_application_traceback` time; the inner
            // `CodeObject` lives as long as `w_code` is reachable.
            let code_obj = unsafe { crate::w_code_get_ptr(w_code) } as *const crate::CodeObject;
            if code_obj.is_null() {
                (String::from("<unknown>"), String::from("<unknown>"), None)
            } else {
                let code = unsafe { &*code_obj };
                let location = usize::try_from(lasti)
                    .ok()
                    .and_then(|index| code.locations.get(index))
                    .map(|(start, end)| {
                        (
                            start.line.get(),
                            end.line.get(),
                            start.character_offset.get().saturating_sub(1),
                            end.character_offset.get().saturating_sub(1),
                        )
                    });
                (
                    code.source_path.to_string(),
                    code.obj_name.to_string(),
                    location,
                )
            }
        };
        writeln!(
            writer,
            "  File \"{}\", line {}, in {}",
            filename, lineno, funcname
        )?;
        if let Some(line) = read_source_line(&filename, lineno) {
            let raw_line = line.trim_end_matches(['\n', '\r']);
            let shown_line = raw_line.trim_start();
            writeln!(writer, "    {shown_line}")?;

            if let Some((start_line, end_line, start_col, end_col)) = location
                && usize::try_from(lineno).ok() == Some(start_line)
                && end_line == start_line
                && end_col > start_col
                && start_col <= raw_line.len()
                && end_col <= raw_line.len()
                && raw_line.is_char_boundary(start_col)
                && raw_line.is_char_boundary(end_col)
            {
                let lead = raw_line.len() - shown_line.len();
                let anchors = traceback_anchors(raw_line, start_col, end_col);
                // `traceback.py:_should_show_carets` — a plain caret span covering
                // the whole stripped line (nothing but whitespace before the start
                // or after the end, and no binary-op/subscript sub-anchor) carries
                // no information, so it is suppressed. A `<name> = <call>(…)` or
                // `return <call>(…)` whose call value exactly covers the span is
                // likewise suppressed even though it has a call sub-anchor.
                let before_ws_only = raw_line
                    .get(..start_col)
                    .unwrap_or("")
                    .trim_start()
                    .is_empty();
                let after_ws_only = raw_line.get(end_col..).unwrap_or("").trim_end().is_empty();
                let full_line_call = spawns_full_line_call(
                    shown_line,
                    start_col.saturating_sub(lead),
                    end_col.saturating_sub(lead),
                );
                if !full_line_call && (anchors.is_some() || !before_ws_only || !after_ws_only) {
                    let mut caret_line = String::from("    ");
                    caret_line.push_str(&" ".repeat(start_col.saturating_sub(lead)));
                    for col in start_col..end_col {
                        let marker = if let Some((lo, hi)) = anchors {
                            if (lo..hi).contains(&col) { '^' } else { '~' }
                        } else {
                            '^'
                        };
                        caret_line.push(marker);
                    }
                    writeln!(writer, "{caret_line}")?;
                }
            }
        }
        tb = unsafe { crate::pytraceback::w_pytraceback_get_w_next(tb) };
    }
    Ok(())
}

/// `traceback.py:_should_show_carets` special case: a `<name> = <call>(…)` or
/// `return <call>(…)` statement whose call value exactly spans the failing
/// instruction (byte offsets `seg_start..seg_end` into `line`, the already-
/// dedented source line). Such a caret would merely re-underline the whole
/// right-hand side, so it is suppressed.
fn spawns_full_line_call(line: &str, seg_start: usize, seg_end: usize) -> bool {
    let Ok(parsed) = parser::parse_module(line) else {
        return false;
    };
    let body = &parsed.syntax().body;
    let [stmt] = body.as_slice() else {
        return false;
    };
    let call = match stmt {
        ast::Stmt::Assign(assign) => {
            if assign.targets.len() != 1
                || !matches!(assign.targets.first(), Some(ast::Expr::Name(_)))
                || !matches!(assign.value.as_ref(), ast::Expr::Call(_))
            {
                return false;
            }
            assign.value.range()
        }
        ast::Stmt::Return(ret) => match ret.value.as_deref() {
            Some(ast::Expr::Call(call)) if matches!(call.func.as_ref(), ast::Expr::Name(_)) => {
                call.range()
            }
            _ => return false,
        },
        _ => return false,
    };
    call.start().to_usize() == seg_start && call.end().to_usize() == seg_end
}

/// Return the secondary-marker span used by `traceback.py` for an expression.
fn traceback_anchors(raw_line: &str, start_col: usize, end_col: usize) -> Option<(usize, usize)> {
    let segment = raw_line.get(start_col..end_col)?;
    let wrapped = format!("(\n{segment}\n)");
    let parsed = parser::parse_expression(&wrapped).ok()?;
    let expr = parsed.syntax().body.as_ref();

    let map_offset = |offset: usize| {
        offset
            .checked_sub(2)
            .and_then(|offset| offset.checked_add(start_col))
            .map(|offset| offset.clamp(start_col, end_col))
    };
    let map_range = |lo: usize, hi: usize| {
        let lo = map_offset(lo)?;
        let hi = map_offset(hi)?;
        (lo < hi).then_some((lo, hi))
    };

    match expr {
        ast::Expr::BinOp(bin_op) => {
            let left_end = bin_op.left.range().end().to_usize();
            let right_start = bin_op.right.range().start().to_usize();
            if left_end > right_start || right_start > wrapped.len() {
                return None;
            }
            let between = wrapped.as_bytes().get(left_end..right_start)?;
            let op_relative = between
                .iter()
                .position(|byte| !byte.is_ascii_whitespace() && *byte != b')')?;
            let op_start = left_end.checked_add(op_relative)?;
            let mut op_end = op_start;
            while op_end < right_start {
                let byte = *wrapped.as_bytes().get(op_end)?;
                if byte.is_ascii_whitespace()
                    || byte == b')'
                    || byte.is_ascii_alphanumeric()
                    || byte == b'_'
                {
                    break;
                }
                op_end = op_end.checked_add(1)?;
            }
            map_range(op_start, op_end)
        }
        ast::Expr::Subscript(subscript) => map_range(
            subscript.value.range().end().to_usize(),
            expr.range().end().to_usize(),
        ),
        ast::Expr::Call(call) => map_range(
            call.func.range().end().to_usize(),
            expr.range().end().to_usize(),
        ),
        _ => None,
    }
}

/// Open `filename` and return its `lineno`-th line (1-indexed).  Returns
/// `None` for synthetic / unreadable sources — matches PyPy's silent
/// `linecache.getline` fallback at `error.py:150`.
fn read_source_line(filename: &str, lineno: i64) -> Option<String> {
    if lineno <= 0 || filename.is_empty() || filename.starts_with('<') {
        return None;
    }
    #[cfg(all(feature = "host_env", not(target_arch = "wasm32")))]
    {
        // Read through the import machinery's source provider, not std::fs:
        // under sandbox that routes the read through the seam to the controller
        // VFS, so a guest-controlled traceback path cannot leak a host file.
        let content =
            crate::importing::read_source_to_string(std::path::Path::new(filename)).ok()?;
        content
            .lines()
            .nth((lineno - 1) as usize)
            .map(|s| s.to_string())
    }
    #[cfg(any(not(feature = "host_env"), target_arch = "wasm32"))]
    {
        // Sandbox-intentional: PyPy's `error.py:150 linecache.getline`
        // also returns silently when the source can't be read; with
        // host_env off the interpreter must not reach `std::fs`
        // directly, so we treat every source as unreadable and let the
        // traceback render `^^^` markers without the offending line.
        let _ = (filename, lineno);
        None
    }
}

pub fn eprint_exception(err: &PyError, include_traceback: bool) {
    // Buffer then emit through the host_seam so the traceback rides the same
    // mediated stderr as sys.stderr under sandbox (raw fd 2 would bypass the
    // controller / corrupt nothing but escape the seam).
    let mut buf: Vec<u8> = Vec::new();
    let _ = write_exception(&mut buf, err, include_traceback);
    crate::host_seam::emit_stderr(&buf);
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

#[cfg(test)]
mod tests {
    use super::{PyError, PyErrorKind, write_exception};

    #[test]
    fn render_exception_omits_empty_message_separator() {
        let err = PyError::new(PyErrorKind::StopIteration, "");
        assert_eq!(err.render_exception(), "StopIteration");
    }

    #[test]
    fn write_exception_includes_traceback_header() {
        let err = PyError::type_error("bad operand");
        let mut out = Vec::new();
        write_exception(&mut out, &err, true).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("Traceback (most recent call last):"));
        assert!(text.contains("TypeError: bad operand"));
    }
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
