//! The C exception indicator -- PyPy `cpyext/state.py` and `cpyext/pyerrors.py`.
//!
//! Upstream parks an `OperationError` on the execution context
//! (`state.py:14 ExecutionContext.cpyext_operror`) and normalizes it only when
//! the interpreter takes it back.  Pyre's [`crate::PyError`] carries the
//! exception *instance*, so the indicator is the instance's mirror: the mirror
//! census already roots the link, which is why no separate root walker is
//! needed for the pending error.  Normalizing at `PyErr_Set*` time rather than
//! at hand-back time is the visible consequence — a C function that sets and
//! then clears an exception has still run the exception class's constructor.

use super::pyobject::{self, CPyObject, REFCNT_IMMORTAL};
use pyre_object::{PY_NULL, PyObjectRef};
use std::cell::Cell;
use std::ffi::{CStr, c_char, c_int};

thread_local! {
    /// The pending exception instance's mirror, owned.
    static PENDING: Cell<*mut CPyObject> = const { Cell::new(std::ptr::null_mut()) };
}

/// Replace the indicator, releasing whatever it held.
///
/// Every caller hands in a reference of its own — a fresh one from `make_ref`,
/// or the one `PyErr_Restore` was given to steal — so the release happens even
/// when the replacement is the same mirror.  Skipping it there would keep one
/// reference per repeat of the same exception instance.
fn set_pending_raw(raw: *mut CPyObject) {
    let previous = PENDING.with(|slot| slot.replace(raw));
    if !previous.is_null() {
        unsafe { pyobject::decref(previous) };
    }
}

fn pending_raw() -> *mut CPyObject {
    PENDING.with(|slot| slot.get())
}

/// `State.set_exception` — record an interpreter-level error for the C caller.
pub fn set_pending_error(mut error: crate::PyError) {
    let instance = error.to_exc_object();
    set_pending_raw(pyobject::make_ref(instance));
}

/// `State.clear_exception` — take the indicator back as an interpreter error.
pub fn take_pending_error() -> Option<crate::PyError> {
    let raw = pending_raw();
    if raw.is_null() {
        return None;
    }
    let instance = unsafe { pyobject::from_ref(raw) };
    // Built before the mirror is released: `from_exc_object` only copies the
    // pointer, and releasing the mirror cannot collect.
    let error = unsafe { crate::PyError::from_exc_object(instance) };
    set_pending_raw(std::ptr::null_mut());
    Some(error)
}

pub fn has_pending_error() -> bool {
    !pending_raw().is_null()
}

/// Record an interpreter-level failure and report it to C as `None`.
///
/// Every C entry point that can fail funnels through this, which is what
/// `@cpython_api`'s generated wrapper does upstream (`api.py`: the wrapper
/// catches `OperationError` and stores it in the state).
pub(super) fn trap<T>(result: Result<T, crate::PyError>) -> Option<T> {
    match result {
        Ok(value) => Some(value),
        Err(error) => {
            set_pending_error(error);
            None
        }
    }
}

/// Raise a `TypeError` from a C entry point handed the wrong kind of object.
pub(super) fn bad_argument<T>(function: &str) -> Option<T> {
    set_pending_error(crate::PyError::type_error(format!(
        "bad argument type for built-in operation {function}()"
    )));
    None
}

// ── the exception type mirrors ──────────────────────────────────────────

macro_rules! exception_mirrors {
    ($($symbol:ident => $name:literal,)*) => {
        $(
            /// The mirror of the like-named builtin exception class.  Null
            /// until the first extension load, and null forever for a class
            /// this build does not register.
            #[unsafe(no_mangle)]
            pub static mut $symbol: *mut CPyObject = std::ptr::null_mut();
        )*

        /// Bind every exception mirror to its builtin class.
        pub fn init_exception_mirrors() {
            $(
                unsafe {
                    if $symbol.is_null()
                        && let Some(class) = crate::builtins::lookup_exc_class($name)
                    {
                        let raw = pyobject::make_ref(class);
                        // Handed out borrowed to every `PyErr_*` caller, so it
                        // must outlive them all.
                        (*raw).ob_refcnt = REFCNT_IMMORTAL;
                        $symbol = raw;
                    }
                }
            )*
        }

        fn ensure_mirrors_linked() {
            $( std::hint::black_box(&raw const $symbol); )*
        }
    };
}

exception_mirrors! {
    PyExc_BaseException => "BaseException",
    PyExc_Exception => "Exception",
    PyExc_ArithmeticError => "ArithmeticError",
    PyExc_AssertionError => "AssertionError",
    PyExc_AttributeError => "AttributeError",
    PyExc_BaseExceptionGroup => "BaseExceptionGroup",
    PyExc_BlockingIOError => "BlockingIOError",
    PyExc_BrokenPipeError => "BrokenPipeError",
    PyExc_BufferError => "BufferError",
    PyExc_ChildProcessError => "ChildProcessError",
    PyExc_ConnectionAbortedError => "ConnectionAbortedError",
    PyExc_ConnectionError => "ConnectionError",
    PyExc_ConnectionRefusedError => "ConnectionRefusedError",
    PyExc_ConnectionResetError => "ConnectionResetError",
    PyExc_EOFError => "EOFError",
    PyExc_FileExistsError => "FileExistsError",
    PyExc_FileNotFoundError => "FileNotFoundError",
    PyExc_FloatingPointError => "FloatingPointError",
    PyExc_GeneratorExit => "GeneratorExit",
    PyExc_ImportError => "ImportError",
    PyExc_IndentationError => "IndentationError",
    PyExc_IndexError => "IndexError",
    PyExc_InterruptedError => "InterruptedError",
    PyExc_IsADirectoryError => "IsADirectoryError",
    PyExc_KeyError => "KeyError",
    PyExc_KeyboardInterrupt => "KeyboardInterrupt",
    PyExc_LookupError => "LookupError",
    PyExc_MemoryError => "MemoryError",
    PyExc_ModuleNotFoundError => "ModuleNotFoundError",
    PyExc_NameError => "NameError",
    PyExc_NotADirectoryError => "NotADirectoryError",
    PyExc_NotImplementedError => "NotImplementedError",
    PyExc_OSError => "OSError",
    PyExc_OverflowError => "OverflowError",
    PyExc_PermissionError => "PermissionError",
    PyExc_ProcessLookupError => "ProcessLookupError",
    PyExc_PythonFinalizationError => "PythonFinalizationError",
    PyExc_RecursionError => "RecursionError",
    PyExc_ReferenceError => "ReferenceError",
    PyExc_RuntimeError => "RuntimeError",
    PyExc_StopAsyncIteration => "StopAsyncIteration",
    PyExc_StopIteration => "StopIteration",
    PyExc_SyntaxError => "SyntaxError",
    PyExc_SystemError => "SystemError",
    PyExc_SystemExit => "SystemExit",
    PyExc_TabError => "TabError",
    PyExc_TimeoutError => "TimeoutError",
    PyExc_TypeError => "TypeError",
    PyExc_UnboundLocalError => "UnboundLocalError",
    PyExc_UnicodeDecodeError => "UnicodeDecodeError",
    PyExc_UnicodeEncodeError => "UnicodeEncodeError",
    PyExc_UnicodeError => "UnicodeError",
    PyExc_UnicodeTranslateError => "UnicodeTranslateError",
    PyExc_ValueError => "ValueError",
    PyExc_ZeroDivisionError => "ZeroDivisionError",
    // `OSError` under the two names it answered to before 3.3.
    PyExc_EnvironmentError => "OSError",
    PyExc_IOError => "OSError",
    // The warning classes.
    PyExc_Warning => "Warning",
    PyExc_BytesWarning => "BytesWarning",
    PyExc_DeprecationWarning => "DeprecationWarning",
    PyExc_EncodingWarning => "EncodingWarning",
    PyExc_FutureWarning => "FutureWarning",
    PyExc_ImportWarning => "ImportWarning",
    PyExc_PendingDeprecationWarning => "PendingDeprecationWarning",
    PyExc_ResourceWarning => "ResourceWarning",
    PyExc_RuntimeWarning => "RuntimeWarning",
    PyExc_SyntaxWarning => "SyntaxWarning",
    PyExc_UnicodeWarning => "UnicodeWarning",
    PyExc_UserWarning => "UserWarning",
}

// ── the public entry points ─────────────────────────────────────────────

/// Build the exception instance `PyErr_SetObject` describes.
///
/// `pyerrors.py:PyErr_SetObject` stores the pair unnormalized; the shape below
/// is `PyErr_NormalizeException`'s, run eagerly for the reason at the top of
/// this module.
fn normalized(w_type: PyObjectRef, w_value: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let type_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(w_type);
    let value_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(w_value);
    if !w_value.is_null()
        && unsafe { pyre_object::is_exception(w_value) }
        && crate::baseobjspace::isinstance(
            w_value,
            pyre_object::gc_roots::shadow_stack_get(type_slot),
        )?
    {
        return Ok(pyre_object::gc_roots::shadow_stack_get(value_slot));
    }
    let w_value = pyre_object::gc_roots::shadow_stack_get(value_slot);
    let w_type = pyre_object::gc_roots::shadow_stack_get(type_slot);
    let instance = if w_value.is_null() || unsafe { pyre_object::is_none(w_value) } {
        crate::call::call_function_impl_result(w_type, &[])?
    } else if unsafe { pyre_object::is_tuple(w_value) } {
        let items = unsafe { pyre_object::tupleobject::w_tuple_items_copy_as_vec(w_value) };
        crate::call::call_function_impl_result(
            pyre_object::gc_roots::shadow_stack_get(type_slot),
            &items,
        )?
    } else {
        crate::call::call_function_impl_result(w_type, &[w_value])?
    };
    if !unsafe { pyre_object::is_exception(instance) } {
        return Err(crate::PyError::type_error(
            "calling an exception class did not return an exception instance",
        ));
    }
    Ok(instance)
}

/// `_PyErr_SetObject` — normalize the pair and make the instance the indicator.
///
/// The chaining is that function's too: an exception raised while another is
/// being handled records the one being handled as its `__context__`, so the C
/// caller's error does not hide it.
fn set_normalized(class: PyObjectRef, value: PyObjectRef) {
    let Some(instance) = trap(normalized(class, value)) else {
        return;
    };
    crate::error::chain_context(instance, handled_exception());
    set_pending_raw(pyobject::make_ref(instance));
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyErr_SetObject(w_type: *mut CPyObject, value: *mut CPyObject) {
    super::object::realize_all([w_type, value]);
    let Some(class) = class_argument(w_type) else {
        return;
    };
    set_normalized(class, unsafe { pyobject::from_ref(value) });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyErr_SetString(w_type: *mut CPyObject, message: *const c_char) {
    let Some(class) = class_argument(w_type) else {
        return;
    };
    let text = if message.is_null() {
        PY_NULL
    } else {
        pyre_object::w_str_new(&unsafe { CStr::from_ptr(message) }.to_string_lossy())
    };
    set_normalized(class, text);
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyErr_SetNone(w_type: *mut CPyObject) {
    let Some(class) = class_argument(w_type) else {
        return;
    };
    set_normalized(class, PY_NULL);
}

/// The exception class a `PyErr_Set*` argument names, or `None` after
/// recording the `SystemError` an unusable one deserves.
fn class_argument(w_type: *mut CPyObject) -> Option<PyObjectRef> {
    let class = unsafe { pyobject::from_ref(w_type) };
    if class.is_null() || !unsafe { crate::baseobjspace::exception_is_valid_obj_as_class_w(class) }
    {
        set_pending_error(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "exception class expected",
        ));
        return None;
    }
    Some(class)
}

/// The pending exception's class, borrowed, or NULL when none is set.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyErr_Occurred() -> *mut CPyObject {
    let raw = pending_raw();
    if raw.is_null() {
        return std::ptr::null_mut();
    }
    unsafe { (*raw).ob_type as *mut CPyObject }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyErr_Clear() {
    set_pending_raw(std::ptr::null_mut());
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyErr_NoMemory() -> *mut CPyObject {
    set_pending_error(crate::PyError::new(
        crate::PyErrorKind::MemoryError,
        "out of memory",
    ));
    std::ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyErr_BadArgument() -> c_int {
    set_pending_error(crate::PyError::type_error(
        "bad argument type for built-in operation",
    ));
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyErr_BadInternalCall() {
    set_pending_error(crate::PyError::new(
        crate::PyErrorKind::SystemError,
        "bad argument to internal function",
    ));
}

/// The spelling the header's `PyErr_BadInternalCall()` macro expands to, so
/// that the report names the caller's own file and line.
///
/// A call made from inside this layer has no such place to name and goes
/// through the plain entry point above, which is why the same mistake reads
/// differently depending on which side made it.
///
/// # Safety
/// `filename` must be null or NUL-terminated.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _PyErr_BadInternalCall(filename: *const c_char, lineno: c_int) {
    if filename.is_null() {
        unsafe { PyErr_BadInternalCall() };
        return;
    }
    let filename = unsafe { CStr::from_ptr(filename) }.to_string_lossy();
    set_pending_error(crate::PyError::new(
        crate::PyErrorKind::SystemError,
        format!("{filename}:{lineno}: bad argument to internal function"),
    ));
}

/// `PyErr_GivenExceptionMatches` for the pending exception.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyErr_ExceptionMatches(expected: *mut CPyObject) -> c_int {
    let raw = pending_raw();
    if raw.is_null() {
        return 0;
    }
    unsafe { PyErr_GivenExceptionMatches(raw, expected) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyErr_GivenExceptionMatches(
    given: *mut CPyObject,
    expected: *mut CPyObject,
) -> c_int {
    super::object::realize_all([given, expected]);
    let given = unsafe { pyobject::from_ref(given) };
    let expected = unsafe { pyobject::from_ref(expected) };
    if given.is_null() || expected.is_null() {
        return 0;
    }
    let class = if unsafe { pyre_object::is_exception(given) } {
        crate::baseobjspace::exception_getclass(given)
    } else {
        given
    };
    crate::baseobjspace::exception_match(class, expected) as c_int
}

/// Hand the indicator out as the classic `(type, value, traceback)` triple.
///
/// Each slot receives a new reference and the indicator is cleared, which is
/// `PyErr_Fetch`'s contract.  All three are derived from the one instance the
/// indicator holds: the class it is of, itself, and its `__traceback__`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyErr_Fetch(
    ptype: *mut *mut CPyObject,
    pvalue: *mut *mut CPyObject,
    ptraceback: *mut *mut CPyObject,
) {
    let raw = pending_raw();
    let (class, value, traceback) = if raw.is_null() {
        (
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    } else {
        let class = unsafe { (*raw).ob_type as *mut CPyObject };
        unsafe { pyobject::incref(class) };
        let traceback = unsafe { traceback_reference(pyobject::from_ref(raw)) };
        // The indicator's own reference is transferred to `pvalue`.
        PENDING.with(|slot| slot.set(std::ptr::null_mut()));
        (class, raw, traceback)
    };
    unsafe {
        if !ptype.is_null() {
            *ptype = class;
        } else {
            pyobject::decref(class);
        }
        if !pvalue.is_null() {
            *pvalue = value;
        } else {
            pyobject::decref(value);
        }
        if !ptraceback.is_null() {
            *ptraceback = traceback;
        } else {
            pyobject::decref(traceback);
        }
    }
}

/// An exception instance's `__traceback__` as a new reference, or NULL.
fn traceback_reference(instance: PyObjectRef) -> *mut CPyObject {
    if instance.is_null() || !unsafe { pyre_object::is_exception(instance) } {
        return std::ptr::null_mut();
    }
    let stored = unsafe { pyre_object::interp_exceptions::w_exception_get_traceback(instance) };
    if stored.is_null() {
        return std::ptr::null_mut();
    }
    unsafe { crate::pytraceback::mark_traceback_escaped(stored) };
    pyobject::make_ref(stored)
}

/// Detach the indicator and hand it to the caller, which is fetch-and-clear
/// with no triple in the way.
///
/// The reference the indicator held becomes the caller's — nothing is increfed
/// here — and NULL is the answer when nothing was pending, not an error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyErr_GetRaisedException() -> *mut CPyObject {
    let raw = pending_raw();
    PENDING.with(|slot| slot.set(std::ptr::null_mut()));
    raw
}

/// Make `exception` the indicator, stealing the reference and releasing
/// whatever was pending.
///
/// A NULL argument is the clear spelling.  Upstream performs no check at all
/// and a foreign object reaches the unwinder as a garbage read; pyre's
/// indicator is a typed exception instance, so anything else is refused here
/// with the `SystemError` `_PyErr_SetObject` uses for the same mistake.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyErr_SetRaisedException(exception: *mut CPyObject) {
    super::object::realize_all([exception]);
    let instance = unsafe { pyobject::from_ref(exception) };
    if exception.is_null() {
        set_pending_raw(std::ptr::null_mut());
        return;
    }
    if !unsafe { pyre_object::is_exception(instance) } {
        unsafe { pyobject::decref(exception) };
        set_pending_error(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            format!(
                "PyErr_SetRaisedException: exception {} is not a BaseException instance",
                crate::type_methods::arg_type_name(instance)
            ),
        ));
        return;
    }
    set_pending_raw(exception);
}

/// The exception currently being handled, as a new reference, or NULL.
///
/// `sys_exc_info` walks through the suspended generators' saved slots, so this
/// answers the topmost handler's exception rather than only the current
/// frame's, and it clears nothing.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyErr_GetHandledException() -> *mut CPyObject {
    let handled = handled_exception();
    match handled.is_null() {
        true => std::ptr::null_mut(),
        false => pyobject::make_ref(handled),
    }
}

/// The handled exception, or a null pointer when nothing is being handled.
///
/// `None` never survives in the slot — the setters below store it as the empty
/// slot — but a bare `raise` reaching an unset slot writes one, so it is mapped
/// back here rather than handed out as an exception.
fn handled_exception() -> PyObjectRef {
    let handled = crate::eval::get_sys_exception();
    match handled.is_null() || unsafe { !pyre_object::is_exception(handled) } {
        true => PY_NULL,
        false => handled,
    }
}

/// Replace the handled exception, keeping the caller's reference.
///
/// This is the one setter in the family that borrows rather than steals, and
/// `None` is stored as the empty slot.  It writes the current execution
/// context's slot, which is not always the one [`PyErr_GetHandledException`]
/// reads: inside a generator the read falls through to the caller's.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyErr_SetHandledException(exception: *mut CPyObject) {
    super::object::realize_all([exception]);
    let instance = unsafe { pyobject::from_ref(exception) };
    if instance.is_null() || unsafe { pyre_object::is_none(instance) } {
        crate::eval::set_current_exception(PY_NULL);
        return;
    }
    if !unsafe { pyre_object::is_exception(instance) } {
        set_pending_error(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            format!(
                "PyErr_SetHandledException: exception {} is not a BaseException instance",
                crate::type_methods::arg_type_name(instance)
            ),
        ));
        return;
    }
    crate::eval::set_current_exception(instance);
}

/// The handled exception as the classic triple, all three derived from it.
///
/// The empty state is the asymmetric one: the class and traceback slots
/// receive `None` while the value slot receives NULL, which is what tells this
/// apart from `sys.exc_info()`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyErr_GetExcInfo(
    ptype: *mut *mut CPyObject,
    pvalue: *mut *mut CPyObject,
    ptraceback: *mut *mut CPyObject,
) {
    let handled = handled_exception();
    // `Py_None` is immortal, so the two slots that can receive it need no
    // reference of their own — which is why upstream hands it over bare.
    let none = pyobject::borrow_mirror(pyre_object::w_none());
    let (class, value, traceback) = if handled.is_null() {
        (none, std::ptr::null_mut(), none)
    } else {
        let value = pyobject::make_ref(handled);
        let class = unsafe { (*value).ob_type as *mut CPyObject };
        unsafe { pyobject::incref(class) };
        let traceback = traceback_reference(handled);
        let traceback = match traceback.is_null() {
            true => none,
            false => traceback,
        };
        (class, value, traceback)
    };
    unsafe {
        store_or_release(ptype, class);
        store_or_release(pvalue, value);
        store_or_release(ptraceback, traceback);
    }
}

/// Hand `value` to `slot`, or release it when there is no slot to take it.
unsafe fn store_or_release(slot: *mut *mut CPyObject, value: *mut CPyObject) {
    match slot.is_null() {
        true => unsafe { pyobject::decref(value) },
        false => unsafe { *slot = value },
    }
}

/// The three-argument spelling of [`PyErr_SetHandledException`], which is all
/// it is: only the value is stored, and all three references are stolen.
///
/// Whatever is passed as the class or the traceback is unobservable — a later
/// [`PyErr_GetExcInfo`] derives both from the value again — so the two are
/// released and nothing else.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyErr_SetExcInfo(
    ptype: *mut CPyObject,
    pvalue: *mut CPyObject,
    ptraceback: *mut CPyObject,
) {
    unsafe {
        PyErr_SetHandledException(pvalue);
        pyobject::decref(pvalue);
        pyobject::decref(ptype);
        pyobject::decref(ptraceback);
    }
}

/// The inverse of [`PyErr_Fetch`]; every argument's reference is stolen.
///
/// The traceback is written onto the instance rather than dropped, which is
/// what makes the pair lossless: `Fetch` reads that same slot, so a caller
/// that saves the triple and hands it straight back keeps the traceback the
/// exception was carrying.  Nothing is chained here — the exception being
/// restored is an older one rather than a new raise.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyErr_Restore(
    ptype: *mut CPyObject,
    pvalue: *mut CPyObject,
    ptraceback: *mut CPyObject,
) {
    super::object::realize_all([ptype, pvalue, ptraceback]);
    let traceback = unsafe { pyobject::from_ref(ptraceback) };
    // `errors.c:62-67` — a NULL class clears the indicator whatever the value
    // is, so restoring what `PyErr_Fetch` handed back on a clear indicator
    // cannot leave an older exception standing.
    if ptype.is_null() {
        unsafe {
            pyobject::decref(pvalue);
            pyobject::decref(ptraceback);
        }
        set_pending_raw(std::ptr::null_mut());
        return;
    }
    let value = unsafe { pyobject::from_ref(pvalue) };
    if !value.is_null() && unsafe { pyre_object::is_exception(value) } {
        restore_traceback(value, traceback);
        set_pending_raw(pvalue);
        unsafe {
            pyobject::decref(ptype);
            pyobject::decref(ptraceback);
        }
        return;
    }
    // `errors.c:77-86` — anything else, a NULL value included, is built into an
    // instance by calling the class, so the restored pair is consumed rather
    // than stored.  A class whose mirror no longer answers leaves nothing to
    // raise, which is the clear indicator rather than the older exception.
    let class = unsafe { pyobject::from_ref(ptype) };
    if class.is_null() {
        set_pending_raw(std::ptr::null_mut());
    } else if let Some(instance) = trap(normalized(class, value)) {
        restore_traceback(instance, traceback);
        set_pending_raw(pyobject::make_ref(instance));
    }
    unsafe {
        pyobject::decref(ptype);
        pyobject::decref(pvalue);
        pyobject::decref(ptraceback);
    }
}

/// Write a restored traceback onto the instance that is about to be raised.
///
/// A NULL one leaves the slot alone, since `errors.c:87` only writes when the
/// triple carried one; anything that is not a traceback is nothing to write.
fn restore_traceback(instance: PyObjectRef, traceback: PyObjectRef) {
    if traceback.is_null() || !unsafe { crate::pytraceback::is_pytraceback(traceback) } {
        return;
    }
    unsafe { pyre_object::interp_exceptions::w_exception_set_traceback(instance, traceback) };
}

/// Build an `ImportError` of `class` and make it the indicator.
///
/// The instance is built by calling the class as `class(message, name=...,
/// path=..., name_from=...)`, so a subclass whose `__init__` does not take
/// those three keywords refuses rather than silently losing them.  Every
/// argument is borrowed, and the answer is always NULL — the return value is
/// the convention `return PyErr_SetImportError(...)` relies on.
fn set_import_error(
    class: *mut CPyObject,
    message: *mut CPyObject,
    name: *mut CPyObject,
    path: *mut CPyObject,
) -> Option<()> {
    super::object::realize_all([class, message, name, path]);
    let class = super::object::argument(class)?;
    let import_error = crate::builtins::lookup_exc_class("ImportError")?;
    let roots = pyre_object::gc_roots::push_roots();
    let class_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(class);
    if !trap(crate::baseobjspace::issubclass(class, import_error))? {
        set_pending_error(crate::PyError::type_error(
            "expected a subclass of ImportError",
        ));
        return None;
    }
    if message.is_null() {
        set_pending_error(crate::PyError::type_error("expected a message argument"));
        return None;
    }
    let named: Vec<(rustpython_wtf8::Wtf8Buf, PyObjectRef)> = [("name", name), ("path", path)]
        .into_iter()
        .map(|(key, raw)| {
            let value = unsafe { pyobject::from_ref(raw) };
            let value = match value.is_null() {
                true => pyre_object::w_none(),
                false => value,
            };
            (rustpython_wtf8::Wtf8Buf::from_string(key.to_owned()), value)
        })
        .chain(std::iter::once((
            rustpython_wtf8::Wtf8Buf::from_string("name_from".to_owned()),
            pyre_object::w_none(),
        )))
        .collect();
    let message = unsafe { pyobject::from_ref(message) };
    let instance = trap(crate::eval::CURRENT_FRAME.with(|current| {
        let frame = current.get();
        if frame.is_null() {
            return Err(crate::PyError::runtime_error(
                "cpyext keyword call has no current frame",
            ));
        }
        crate::call::call_with_kwargs(
            unsafe { &mut *frame },
            pyre_object::gc_roots::shadow_stack_get(class_slot),
            &[message],
            &named,
        )
    }))?;
    if !unsafe { pyre_object::is_exception(instance) } {
        set_pending_error(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            format!(
                "_PyErr_SetObject: exception {} is not a BaseException subclass",
                crate::type_methods::arg_type_name(instance)
            ),
        ));
        return None;
    }
    // `_PyErr_SetObject`'s implicit chaining: the exception being handled
    // becomes the new one's context, so the ImportError does not hide it.
    crate::error::chain_context(instance, handled_exception());
    set_pending_raw(pyobject::make_ref(instance));
    Some(())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyErr_SetImportError(
    message: *mut CPyObject,
    name: *mut CPyObject,
    path: *mut CPyObject,
) -> *mut CPyObject {
    let import_error = unsafe { PyExc_ImportError };
    set_import_error(import_error, message, name, path);
    std::ptr::null_mut()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyErr_SetImportErrorSubclass(
    class: *mut CPyObject,
    message: *mut CPyObject,
    name: *mut CPyObject,
    path: *mut CPyObject,
) -> *mut CPyObject {
    set_import_error(class, message, name, path);
    std::ptr::null_mut()
}

/// `PyErr_NormalizeException` — replace the pair with the exception instance
/// the class and value name.
///
/// An indicator this module set is normalized already, but the triple handed in
/// need not have come from one: an extension may build `(PyExc_ValueError, "a
/// message", NULL)` itself.  Both slots are read as owned references and left
/// holding new ones, which is the documented transfer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyErr_NormalizeException(
    ptype: *mut *mut CPyObject,
    pvalue: *mut *mut CPyObject,
    _ptraceback: *mut *mut CPyObject,
) {
    if ptype.is_null() || pvalue.is_null() {
        return;
    }
    let (was_type, was_value) = unsafe { (*ptype, *pvalue) };
    super::object::realize_all([was_type, was_value]);
    let class = unsafe { pyobject::from_ref(was_type) };
    if class.is_null() {
        return;
    }
    let value = unsafe { pyobject::from_ref(was_value) };
    let Some(instance) = trap(normalized(class, value)) else {
        return;
    };
    let roots = pyre_object::gc_roots::push_roots();
    let instance_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(instance);
    let Some(w_class) = crate::typedef::r#type(instance) else {
        return;
    };
    let class_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(w_class.as_ptr());
    let value_ref = pyobject::make_ref(pyre_object::gc_roots::shadow_stack_get(instance_slot));
    let type_ref = pyobject::make_ref(pyre_object::gc_roots::shadow_stack_get(class_slot));
    unsafe {
        *ptype = type_ref;
        *pvalue = value_ref;
        pyobject::decref(was_type);
        pyobject::decref(was_value);
    }
}

/// `_PyErr_ChainExceptions1(exc)` — make `exc` the indicator, or the context
/// of whatever is already pending.
///
/// The reference is stolen either way.  A caller reaches for this having just
/// caught something of its own: the error it is about to report must not hide
/// the one that was already on its way out.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _PyErr_ChainExceptions1(exception: *mut CPyObject) {
    if exception.is_null() {
        return;
    }
    let pending = unsafe { PyErr_GetRaisedException() };
    if pending.is_null() {
        unsafe { PyErr_SetRaisedException(exception) };
        return;
    }
    unsafe { super::exception::PyException_SetContext(pending, exception) };
    unsafe { PyErr_SetRaisedException(pending) };
}

/// `write_unraisable` over whatever the indicator holds, which the two
/// entry points below share.
///
/// The indicator is taken first: reporting runs `sys.unraisablehook`, and a
/// hook that raises has to find the slot free to leave its own error in.
fn write_unraisable(context: Option<String>, object: *mut CPyObject) {
    let Some(mut error) = take_pending_error() else {
        return;
    };
    let object = unsafe { pyobject::from_ref(object) };
    let object = if object.is_null() {
        pyre_object::w_none()
    } else {
        object
    };
    let context = context.unwrap_or_default();
    error.write_unraisable(
        pyre_object::w_none(),
        rustpython_wtf8::Wtf8::new(context.as_str()),
        object,
    );
}

/// `PyErr_WriteUnraisable(object)` — report the pending exception through
/// `sys.unraisablehook` and clear it, naming `object` as what was being
/// operated on.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyErr_WriteUnraisable(object: *mut CPyObject) {
    write_unraisable(None, object);
}

/// The core `PyErr_FormatUnraisable` is built on, whose message states what
/// was going on rather than leaving the report to name the object.
///
/// `message` is a `str`, because the header composes it with the same format
/// engine every other variadic entry point uses.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _PyPyre_WriteUnraisable(message: *mut CPyObject, object: *mut CPyObject) {
    let text = unsafe { pyobject::from_ref(message) };
    let context = (!text.is_null() && unsafe { pyre_object::unicodeobject::is_str(text) })
        .then(|| unsafe { pyre_object::w_str_get_wtf8(text) }.to_string());
    write_unraisable(context, object);
}

/// End the process, reporting `message` the way a caller that cannot go on
/// does.
///
/// There is nothing to raise here: the caller's invariant is already broken,
/// and the exception machinery is one of the things it may have broken.
pub(super) fn fatal_error(function: Option<&str>, message: &str) -> ! {
    use std::io::Write as _;
    let mut stderr = std::io::stderr().lock();
    let _ = match function {
        Some(function) => writeln!(stderr, "Fatal Python error: {function}: {message}"),
        None => writeln!(stderr, "Fatal Python error: {message}"),
    };
    let _ = stderr.flush();
    std::process::abort()
}

/// `Py_FatalError(message)`, which the header spells as this so that the
/// report names the function the caller gave up in.
///
/// # Safety
/// `function` may be null; `message` must be null or NUL-terminated.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _Py_FatalErrorFunc(function: *const c_char, message: *const c_char) -> ! {
    let text = |pointer: *const c_char| {
        (!pointer.is_null()).then(|| unsafe { CStr::from_ptr(pointer) }.to_string_lossy())
    };
    fatal_error(
        text(function).as_deref(),
        text(message).as_deref().unwrap_or("<message unavailable>"),
    )
}

pub(super) fn ensure_linked() {
    std::hint::black_box(_PyErr_ChainExceptions1 as *const ());
    std::hint::black_box(_Py_FatalErrorFunc as *const ());
    ensure_mirrors_linked();
    std::hint::black_box(PyErr_SetObject as *const ());
    std::hint::black_box(PyErr_SetString as *const ());
    std::hint::black_box(PyErr_SetNone as *const ());
    std::hint::black_box(PyErr_Occurred as *const ());
    std::hint::black_box(PyErr_Clear as *const ());
    std::hint::black_box(PyErr_NoMemory as *const ());
    std::hint::black_box(PyErr_BadArgument as *const ());
    std::hint::black_box(PyErr_BadInternalCall as *const ());
    std::hint::black_box(_PyErr_BadInternalCall as *const ());
    std::hint::black_box(PyErr_ExceptionMatches as *const ());
    std::hint::black_box(PyErr_GivenExceptionMatches as *const ());
    std::hint::black_box(PyErr_Fetch as *const ());
    // ── the failed syscall ──────────────────────────────────────────────────

    /// `PyErr_CheckSignals` — run whatever a signal handler left pending.
    ///
    /// The handler is Python code, so it can raise; a raise is what the `-1`
    /// reports, and the exception it raised is left pending.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn PyErr_CheckSignals() -> c_int {
        match crate::module::signal::interp_signal::checksignals_now() {
            Ok(()) => 0,
            Err(error) => {
                set_pending_error(error);
                -1
            }
        }
    }

    /// `pyerrors.py:PyErr_SetFromErrnoWithFilename` — the family's body, over a
    /// code its callers read before anything that could overwrite it.
    ///
    /// The class is the caller's and it is *called*, so the exception is whatever
    /// calling it answers with rather than an `OSError` this layer built: a
    /// subclass declaring its own `__init__` sees the same arguments it would from
    /// Python.  Every spelling answers NULL, having raised.
    unsafe fn set_from_errno(
        code: c_int,
        exc: *mut CPyObject,
        filename: *mut CPyObject,
        second_filename: *mut CPyObject,
    ) -> *mut CPyObject {
        // A syscall interrupted by a signal is the handler's to report first, and
        // the handler raising is what leaves this with nothing of its own to say.
        if code == libc::EINTR && unsafe { PyErr_CheckSignals() } != 0 {
            return std::ptr::null_mut();
        }
        super::object::realize_all([exc, filename, second_filename]);
        let Some(class) = class_argument(exc) else {
            return std::ptr::null_mut();
        };
        // `errno` unset is a syscall that failed without recording which way.
        let message = match code {
            0 => "Error".to_owned(),
            code => crate::PyError::clean_strerror(code),
        };
        let roots = pyre_object::gc_roots::push_roots();
        let class_slot = pyre_object::gc_roots::shadow_stack_len();
        roots.pin_root(class);
        let filename_slot = pyre_object::gc_roots::shadow_stack_len();
        roots.pin_root(unsafe { pyobject::from_ref(filename) });
        let second_slot = pyre_object::gc_roots::shadow_stack_len();
        roots.pin_root(unsafe { pyobject::from_ref(second_filename) });

        let mut arguments = vec![
            pyre_object::w_int_new(code as i64),
            pyre_object::w_str_new(&message),
        ];
        let filename = pyre_object::gc_roots::shadow_stack_get(filename_slot);
        if !filename.is_null() {
            arguments.push(filename);
            let second = pyre_object::gc_roots::shadow_stack_get(second_slot);
            if !second.is_null() {
                // The `winerror` slot, 0 off Windows, which is what makes a second
                // filename the fifth argument rather than the fourth.
                arguments.push(pyre_object::w_int_new(0));
                arguments.push(second);
            }
        }
        let arguments = pyre_object::tupleobject::w_tuple_new(arguments);
        set_normalized(
            pyre_object::gc_roots::shadow_stack_get(class_slot),
            arguments,
        );
        std::ptr::null_mut()
    }

    /// The code a failed syscall left behind, read before anything else runs.
    fn current_errno() -> c_int {
        std::io::Error::last_os_error().raw_os_error().unwrap_or(0)
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn PyErr_SetFromErrno(exc: *mut CPyObject) -> *mut CPyObject {
        unsafe {
            set_from_errno(
                current_errno(),
                exc,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        }
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn PyErr_SetFromErrnoWithFilenameObject(
        exc: *mut CPyObject,
        filename: *mut CPyObject,
    ) -> *mut CPyObject {
        unsafe { set_from_errno(current_errno(), exc, filename, std::ptr::null_mut()) }
    }

    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn PyErr_SetFromErrnoWithFilenameObjects(
        exc: *mut CPyObject,
        filename: *mut CPyObject,
        second_filename: *mut CPyObject,
    ) -> *mut CPyObject {
        unsafe { set_from_errno(current_errno(), exc, filename, second_filename) }
    }

    /// The `const char *` spelling, whose filename is a path rather than text.
    ///
    /// Decoding it is a call of its own, so the code is read in front of it: the
    /// C body restores `errno` afterwards for the same reason.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn PyErr_SetFromErrnoWithFilename(
        exc: *mut CPyObject,
        filename: *const c_char,
    ) -> *mut CPyObject {
        let code = current_errno();
        if filename.is_null() {
            return unsafe {
                set_from_errno(code, exc, std::ptr::null_mut(), std::ptr::null_mut())
            };
        }
        let name = unsafe { super::unicodeobject::PyUnicode_DecodeFSDefault(filename) };
        if name.is_null() {
            return std::ptr::null_mut();
        }
        let answer = unsafe { set_from_errno(code, exc, name, std::ptr::null_mut()) };
        unsafe { pyobject::decref(name) };
        answer
    }

    std::hint::black_box(PyErr_Restore as *const ());
    std::hint::black_box(PyErr_NormalizeException as *const ());
}
