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
    PyExc_BufferError => "BufferError",
    PyExc_EOFError => "EOFError",
    PyExc_FileNotFoundError => "FileNotFoundError",
    PyExc_FloatingPointError => "FloatingPointError",
    PyExc_GeneratorExit => "GeneratorExit",
    PyExc_ImportError => "ImportError",
    PyExc_IndexError => "IndexError",
    PyExc_KeyError => "KeyError",
    PyExc_KeyboardInterrupt => "KeyboardInterrupt",
    PyExc_LookupError => "LookupError",
    PyExc_MemoryError => "MemoryError",
    PyExc_ModuleNotFoundError => "ModuleNotFoundError",
    PyExc_NameError => "NameError",
    PyExc_NotImplementedError => "NotImplementedError",
    PyExc_OSError => "OSError",
    PyExc_OverflowError => "OverflowError",
    PyExc_RecursionError => "RecursionError",
    PyExc_ReferenceError => "ReferenceError",
    PyExc_RuntimeError => "RuntimeError",
    PyExc_StopAsyncIteration => "StopAsyncIteration",
    PyExc_StopIteration => "StopIteration",
    PyExc_SyntaxError => "SyntaxError",
    PyExc_SystemError => "SystemError",
    PyExc_SystemExit => "SystemExit",
    PyExc_TypeError => "TypeError",
    PyExc_UnboundLocalError => "UnboundLocalError",
    PyExc_UnicodeDecodeError => "UnicodeDecodeError",
    PyExc_UnicodeEncodeError => "UnicodeEncodeError",
    PyExc_UnicodeError => "UnicodeError",
    PyExc_UnicodeTranslateError => "UnicodeTranslateError",
    PyExc_ValueError => "ValueError",
    PyExc_ZeroDivisionError => "ZeroDivisionError",
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

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyErr_SetObject(w_type: *mut CPyObject, value: *mut CPyObject) {
    super::object::realize_all([w_type, value]);
    let Some(class) = class_argument(w_type) else {
        return;
    };
    let value = unsafe { pyobject::from_ref(value) };
    if let Some(instance) = trap(normalized(class, value)) {
        set_pending_raw(pyobject::make_ref(instance));
    }
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
    if let Some(instance) = trap(normalized(class, text)) {
        set_pending_raw(pyobject::make_ref(instance));
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyErr_SetNone(w_type: *mut CPyObject) {
    let Some(class) = class_argument(w_type) else {
        return;
    };
    if let Some(instance) = trap(normalized(class, PY_NULL)) {
        set_pending_raw(pyobject::make_ref(instance));
    }
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
/// `PyErr_Fetch`'s contract.  The traceback slot is always NULL: pyre attaches
/// the traceback while the error propagates, so a C caller that restores the
/// triple unchanged loses nothing.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyErr_Fetch(
    ptype: *mut *mut CPyObject,
    pvalue: *mut *mut CPyObject,
    ptraceback: *mut *mut CPyObject,
) {
    let raw = pending_raw();
    let (class, value) = if raw.is_null() {
        (std::ptr::null_mut(), std::ptr::null_mut())
    } else {
        let class = unsafe { (*raw).ob_type as *mut CPyObject };
        unsafe { pyobject::incref(class) };
        // The indicator's own reference is transferred to `pvalue`.
        PENDING.with(|slot| slot.set(std::ptr::null_mut()));
        (class, raw)
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
            *ptraceback = std::ptr::null_mut();
        }
    }
}

/// The inverse of [`PyErr_Fetch`]; every argument's reference is stolen.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyErr_Restore(
    ptype: *mut CPyObject,
    pvalue: *mut CPyObject,
    ptraceback: *mut CPyObject,
) {
    super::object::realize_all([ptype, pvalue]);
    unsafe { pyobject::decref(ptraceback) };
    if pvalue.is_null() {
        unsafe { pyobject::decref(ptype) };
        set_pending_raw(std::ptr::null_mut());
        return;
    }
    let value = unsafe { pyobject::from_ref(pvalue) };
    if !value.is_null() && unsafe { pyre_object::is_exception(value) } {
        set_pending_raw(pvalue);
        unsafe { pyobject::decref(ptype) };
        return;
    }
    // A non-instance value has to go back through normalization, so the
    // restored pair is consumed rather than stored.
    let class = unsafe { pyobject::from_ref(ptype) };
    if !class.is_null()
        && let Some(instance) = trap(normalized(class, value))
    {
        set_pending_raw(pyobject::make_ref(instance));
    }
    unsafe {
        pyobject::decref(ptype);
        pyobject::decref(pvalue);
    }
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

/// The non-variadic half of `PyErr_Format`; the header formats the message.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _PyPyre_ErrFormatted(
    w_type: *mut CPyObject,
    message: *const c_char,
) -> *mut CPyObject {
    unsafe { PyErr_SetString(w_type, message) };
    std::ptr::null_mut()
}

pub(super) fn ensure_linked() {
    ensure_mirrors_linked();
    std::hint::black_box(PyErr_SetObject as *const ());
    std::hint::black_box(PyErr_SetString as *const ());
    std::hint::black_box(PyErr_SetNone as *const ());
    std::hint::black_box(PyErr_Occurred as *const ());
    std::hint::black_box(PyErr_Clear as *const ());
    std::hint::black_box(PyErr_NoMemory as *const ());
    std::hint::black_box(PyErr_BadArgument as *const ());
    std::hint::black_box(PyErr_BadInternalCall as *const ());
    std::hint::black_box(PyErr_ExceptionMatches as *const ());
    std::hint::black_box(PyErr_GivenExceptionMatches as *const ());
    std::hint::black_box(PyErr_Fetch as *const ());
    std::hint::black_box(PyErr_Restore as *const ());
    std::hint::black_box(PyErr_NormalizeException as *const ());
    std::hint::black_box(_PyPyre_ErrFormatted as *const ());
}
