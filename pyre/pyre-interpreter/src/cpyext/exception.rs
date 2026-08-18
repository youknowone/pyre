//! The exception instance's own slots -- PyPy `cpyext/exception.py`.
//!
//! Each getter answers the slot verbatim, so an empty one is NULL rather than
//! `None`; each setter takes the slot's documented ownership.  The four PEP
//! 3134 slots live on `W_BaseException` directly, so `__cause__` and
//! `__context__` are written through the typed accessors rather than through
//! `setattr`: the attribute setters refuse anything that is not `None` or a
//! `BaseException` instance, and these entry points perform no such check.

use super::pyobject::{self, CPyObject};
use pyre_object::PyObjectRef;
use std::ffi::{c_char, c_int};

/// The exception instance an accessor was handed, or `None` after recording
/// the `AttributeError` a foreign object deserves.
///
/// Upstream reaches every slot with `space.getattr`, which is what refuses a
/// non-exception; the message is that same lookup failure spelled once here,
/// since the slots below are read off the typed layout instead.
fn instance(object: *mut CPyObject, slot: &str) -> Option<PyObjectRef> {
    let value = unsafe { pyobject::from_ref(object) };
    if value.is_null() || !unsafe { pyre_object::is_exception(value) } {
        let named = match value.is_null() {
            true => "NoneType".to_owned(),
            false => crate::type_methods::arg_type_name(value),
        };
        super::pyerrors::set_pending_error(crate::PyError::attribute_error(format!(
            "'{named}' object has no attribute '{slot}'"
        )));
        return None;
    }
    Some(value)
}

/// A slot read handed out as a new reference, NULL standing for the empty slot.
fn slot_reference(stored: PyObjectRef) -> *mut CPyObject {
    match stored.is_null() {
        true => std::ptr::null_mut(),
        false => pyobject::make_ref(stored),
    }
}

/// Write one of the two stealing slots.
///
/// The order is the one `Py_XSETREF` has: the slot takes the object first and
/// only then is the caller's reference released, because releasing it while
/// nothing else names the object is what would let the store write a corpse.
/// Once stored, the collector reaches the object through the exception.
fn steal_into(
    exception: *mut CPyObject,
    value: *mut CPyObject,
    slot: &str,
    write: unsafe fn(PyObjectRef, PyObjectRef),
) {
    super::object::realize_all([exception, value]);
    let object = unsafe { pyobject::from_ref(value) };
    if let Some(exception) = instance(exception, slot) {
        unsafe { write(exception, object) };
    }
    unsafe { pyobject::decref(value) };
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyException_GetTraceback(exception: *mut CPyObject) -> *mut CPyObject {
    let Some(exception) = instance(exception, "__traceback__") else {
        return std::ptr::null_mut();
    };
    let stored = unsafe { pyre_object::interp_exceptions::w_exception_get_traceback(exception) };
    // `descr_gettraceback` is the slot read plus the escape mark, and a
    // traceback handed to C escapes for as long as the reference lives.
    unsafe { crate::pytraceback::mark_traceback_escaped(stored) };
    slot_reference(stored)
}

/// Write `__traceback__`, the one slot here that is type-checked:
/// `descr_settraceback` accepts a traceback or `None` and nothing else.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyException_SetTraceback(
    exception: *mut CPyObject,
    traceback: *mut CPyObject,
) -> c_int {
    super::object::realize_all([exception, traceback]);
    let Some(exception) = instance(exception, "__traceback__") else {
        return -1;
    };
    let traceback = unsafe { pyobject::from_ref(traceback) };
    let traceback = match traceback.is_null() {
        true => pyre_object::w_none(),
        false => traceback,
    };
    match super::pyerrors::trap(crate::baseobjspace::setattr_str(
        exception,
        "__traceback__",
        traceback,
    )) {
        Some(_) => 0,
        None => -1,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyException_GetCause(exception: *mut CPyObject) -> *mut CPyObject {
    let Some(exception) = instance(exception, "__cause__") else {
        return std::ptr::null_mut();
    };
    slot_reference(unsafe { pyre_object::interp_exceptions::w_exception_get_cause(exception) })
}

/// Write `__cause__` and raise `__suppress_context__`, stealing the reference.
///
/// The flag goes up whatever the cause is, NULL included, which is what
/// `raise ... from None` is spelled as.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyException_SetCause(exception: *mut CPyObject, cause: *mut CPyObject) {
    steal_into(exception, cause, "__cause__", |exception, cause| unsafe {
        pyre_object::interp_exceptions::w_exception_set_cause(exception, cause);
        pyre_object::interp_exceptions::w_exception_set_suppress_context(exception, true);
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyException_GetContext(exception: *mut CPyObject) -> *mut CPyObject {
    let Some(exception) = instance(exception, "__context__") else {
        return std::ptr::null_mut();
    };
    slot_reference(unsafe { pyre_object::interp_exceptions::w_exception_get_context(exception) })
}

/// Write `__context__`, stealing the reference and leaving
/// `__suppress_context__` alone.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyException_SetContext(
    exception: *mut CPyObject,
    context: *mut CPyObject,
) {
    steal_into(
        exception,
        context,
        "__context__",
        |exception, context| unsafe {
            pyre_object::interp_exceptions::w_exception_set_context(exception, context)
        },
    );
}

/// `args`, always a tuple and never NULL for an exception instance.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyException_GetArgs(exception: *mut CPyObject) -> *mut CPyObject {
    let Some(exception) = instance(exception, "args") else {
        return std::ptr::null_mut();
    };
    pyobject::make_ref(unsafe { pyre_object::interp_exceptions::w_exception_get_args(exception) })
}

/// Write `args`, keeping the caller's reference: this setter increfs rather
/// than stealing, which makes it the odd one out among the four.
///
/// The stored form is `descr_setargs`' — the items of the argument, flattened
/// as `space.fixedview` flattens them — so a non-tuple sequence reads back as
/// the tuple of its items rather than as itself.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyException_SetArgs(exception: *mut CPyObject, args: *mut CPyObject) {
    super::object::realize_all([exception, args]);
    let Some(exception) = instance(exception, "args") else {
        return;
    };
    let Some(args) = super::object::argument(args) else {
        return;
    };
    super::pyerrors::trap(crate::baseobjspace::setattr_str(exception, "args", args));
}

/// Whether `object` is a class deriving from `BaseException`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyExceptionClass_Check(object: *mut CPyObject) -> c_int {
    let object = unsafe { pyobject::from_ref(object) };
    (!object.is_null() && unsafe { crate::baseobjspace::exception_is_valid_obj_as_class_w(object) })
        as c_int
}

/// Whether `object` is an instance of a class deriving from `BaseException`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyExceptionInstance_Check(object: *mut CPyObject) -> c_int {
    let object = unsafe { pyobject::from_ref(object) };
    (!object.is_null() && unsafe { pyre_object::is_exception(object) }) as c_int
}

/// An exception class's `tp_name`, which is its mirror's.
///
/// The mirror owns the string for as long as it stands for the class, so the
/// pointer stays readable as long as the caller's reference to the class does.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyExceptionClass_Name(object: *mut CPyObject) -> *const c_char {
    if unsafe { PyExceptionClass_Check(object) } == 0 {
        super::pyerrors::set_pending_error(crate::PyError::type_error(
            "PyExceptionClass_Name(): expected an exception class",
        ));
        return std::ptr::null();
    }
    unsafe { (*(object as *mut super::typeobject::CPyTypeObject)).tp_name }
}

pub(super) fn ensure_linked() {
    std::hint::black_box(PyException_GetTraceback as *const ());
    std::hint::black_box(PyException_SetTraceback as *const ());
    std::hint::black_box(PyException_GetCause as *const ());
    std::hint::black_box(PyException_SetCause as *const ());
    std::hint::black_box(PyException_GetContext as *const ());
    std::hint::black_box(PyException_SetContext as *const ());
    std::hint::black_box(PyException_GetArgs as *const ());
    std::hint::black_box(PyException_SetArgs as *const ());
    std::hint::black_box(PyExceptionClass_Check as *const ());
    std::hint::black_box(PyExceptionInstance_Check as *const ());
    std::hint::black_box(PyExceptionClass_Name as *const ());
}
