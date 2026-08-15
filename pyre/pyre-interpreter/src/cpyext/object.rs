//! The object protocol -- PyPy `cpyext/object.py` and `cpyext/mapping.py`.

use super::pyerrors::trap;
use super::pyobject::{self, CPyObject};
use pyre_object::PyObjectRef;
use std::ffi::{CStr, c_char, c_int};

/// The interpreter object behind an argument, or a recorded `SystemError`.
///
/// Upstream's generated wrappers reject a NULL argument with
/// `PyErr_BadInternalCall`; this is that check in one place.
pub(super) fn argument(raw: *mut CPyObject) -> Option<PyObjectRef> {
    let value = unsafe { pyobject::from_ref(raw) };
    if value.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return None;
    }
    Some(value)
}

/// A new reference to the result of an interpreter operation, or NULL with the
/// error recorded.
pub(super) fn result(value: Result<PyObjectRef, crate::PyError>) -> *mut CPyObject {
    match trap(value) {
        Some(value) => pyobject::make_ref(value),
        None => std::ptr::null_mut(),
    }
}

fn name_of(pointer: *const c_char) -> Option<String> {
    if pointer.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return None;
    }
    Some(
        unsafe { CStr::from_ptr(pointer) }
            .to_string_lossy()
            .into_owned(),
    )
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_GetAttrString(
    object: *mut CPyObject,
    name: *const c_char,
) -> *mut CPyObject {
    let (Some(object), Some(name)) = (argument(object), name_of(name)) else {
        return std::ptr::null_mut();
    };
    result(crate::baseobjspace::getattr_str(object, &name))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_SetAttrString(
    object: *mut CPyObject,
    name: *const c_char,
    value: *mut CPyObject,
) -> c_int {
    let (Some(object), Some(name)) = (argument(object), name_of(name)) else {
        return -1;
    };
    let value = unsafe { pyobject::from_ref(value) };
    let outcome = if value.is_null() {
        crate::baseobjspace::delattr_str(object, &name).map(|_| pyre_object::w_none())
    } else {
        crate::baseobjspace::setattr_str(object, &name, value)
    };
    if trap(outcome).is_none() { -1 } else { 0 }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_HasAttrString(
    object: *mut CPyObject,
    name: *const c_char,
) -> c_int {
    let (Some(object), Some(name)) = (argument(object), name_of(name)) else {
        unsafe { super::pyerrors::PyErr_Clear() };
        return 0;
    };
    match crate::baseobjspace::getattr_str(object, &name) {
        Ok(_) => 1,
        Err(_) => 0,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_GetAttr(
    object: *mut CPyObject,
    name: *mut CPyObject,
) -> *mut CPyObject {
    let (Some(object), Some(name)) = (argument(object), argument(name)) else {
        return std::ptr::null_mut();
    };
    result(crate::baseobjspace::getattr(object, name))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_SetAttr(
    object: *mut CPyObject,
    name: *mut CPyObject,
    value: *mut CPyObject,
) -> c_int {
    let (Some(object), Some(name)) = (argument(object), argument(name)) else {
        return -1;
    };
    let Some(value) = argument(value) else {
        return -1;
    };
    if trap(crate::baseobjspace::setattr(object, name, value)).is_none() {
        return -1;
    }
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_Str(object: *mut CPyObject) -> *mut CPyObject {
    let Some(object) = argument(object) else {
        return std::ptr::null_mut();
    };
    result(unsafe { crate::display::py_str_wtf8(object) }.map(pyre_object::w_str_from_wtf8))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_Repr(object: *mut CPyObject) -> *mut CPyObject {
    let Some(object) = argument(object) else {
        return std::ptr::null_mut();
    };
    result(unsafe { crate::display::py_repr_wtf8(object) }.map(pyre_object::w_str_from_wtf8))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_IsTrue(object: *mut CPyObject) -> c_int {
    let Some(object) = argument(object) else {
        return -1;
    };
    match trap(crate::baseobjspace::is_true(object)) {
        Some(true) => 1,
        Some(false) => 0,
        None => -1,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_Not(object: *mut CPyObject) -> c_int {
    match unsafe { PyObject_IsTrue(object) } {
        -1 => -1,
        0 => 1,
        _ => 0,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_Size(object: *mut CPyObject) -> isize {
    let Some(object) = argument(object) else {
        return -1;
    };
    let length =
        trap(crate::baseobjspace::len(object).and_then(|value| {
            crate::baseobjspace::gateway_int_w(value).map(|length| length as isize)
        }));
    length.unwrap_or(-1)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_GetItem(
    object: *mut CPyObject,
    key: *mut CPyObject,
) -> *mut CPyObject {
    let (Some(object), Some(key)) = (argument(object), argument(key)) else {
        return std::ptr::null_mut();
    };
    result(crate::baseobjspace::getitem(object, key))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_SetItem(
    object: *mut CPyObject,
    key: *mut CPyObject,
    value: *mut CPyObject,
) -> c_int {
    let (Some(object), Some(key)) = (argument(object), argument(key)) else {
        return -1;
    };
    let Some(value) = argument(value) else {
        return -1;
    };
    if trap(crate::baseobjspace::setitem(object, key, value)).is_none() {
        return -1;
    }
    0
}

/// `PyObject_Call(callable, args, kwargs)` — `args` must be a tuple and
/// `kwargs` a dict or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_Call(
    callable: *mut CPyObject,
    args: *mut CPyObject,
    kwargs: *mut CPyObject,
) -> *mut CPyObject {
    let (Some(callable), Some(args)) = (argument(callable), argument(args)) else {
        return std::ptr::null_mut();
    };
    if !unsafe { pyre_object::is_tuple(args) } {
        super::pyerrors::set_pending_error(crate::PyError::type_error(
            "argument list must be a tuple",
        ));
        return std::ptr::null_mut();
    }
    let kwargs = unsafe { pyobject::from_ref(kwargs) };
    if !kwargs.is_null() && !unsafe { pyre_object::is_dict(kwargs) } {
        super::pyerrors::set_pending_error(crate::PyError::type_error(
            "keyword arguments must be a dict",
        ));
        return std::ptr::null_mut();
    }
    result(call_with_keywords(callable, args, kwargs))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_CallObject(
    callable: *mut CPyObject,
    args: *mut CPyObject,
) -> *mut CPyObject {
    let Some(callable) = argument(callable) else {
        return std::ptr::null_mut();
    };
    let args = unsafe { pyobject::from_ref(args) };
    if args.is_null() {
        return result(crate::call::call_function_impl_result(callable, &[]));
    }
    unsafe {
        PyObject_Call(
            pyobject::as_pyobj(callable),
            pyobject::as_pyobj(args),
            std::ptr::null_mut(),
        )
    }
}

fn call_with_keywords(
    callable: PyObjectRef,
    args: PyObjectRef,
    kwargs: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let callable_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(callable);
    let args_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(args);
    let kwargs_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(kwargs);
    let arguments = unsafe {
        pyre_object::tupleobject::w_tuple_items_copy_as_vec(
            pyre_object::gc_roots::shadow_stack_get(args_slot),
        )
    };
    let kwargs = pyre_object::gc_roots::shadow_stack_get(kwargs_slot);
    if kwargs.is_null() {
        return crate::call::call_function_impl_result(
            pyre_object::gc_roots::shadow_stack_get(callable_slot),
            &arguments,
        );
    }
    let entries = unsafe { pyre_object::w_dict_str_entries(kwargs) };
    if entries.is_empty() {
        return crate::call::call_function_impl_result(
            pyre_object::gc_roots::shadow_stack_get(callable_slot),
            &arguments,
        );
    }
    let names: Vec<(rustpython_wtf8::Wtf8Buf, PyObjectRef)> = entries
        .iter()
        .map(|(name, value)| (rustpython_wtf8::Wtf8Buf::from_string(name.clone()), *value))
        .collect();
    crate::eval::CURRENT_FRAME.with(|current| {
        let frame = current.get();
        if frame.is_null() {
            return Err(crate::PyError::runtime_error(
                "cpyext keyword call has no current frame",
            ));
        }
        crate::call::call_with_kwargs(
            unsafe { &mut *frame },
            pyre_object::gc_roots::shadow_stack_get(callable_slot),
            &arguments,
            &names,
        )
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyCallable_Check(object: *mut CPyObject) -> c_int {
    let object = unsafe { pyobject::from_ref(object) };
    if object.is_null() {
        return 0;
    }
    crate::baseobjspace::callable_w(object) as c_int
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_IsInstance(
    object: *mut CPyObject,
    class: *mut CPyObject,
) -> c_int {
    let (Some(object), Some(class)) = (argument(object), argument(class)) else {
        return -1;
    };
    match trap(crate::baseobjspace::isinstance(object, class)) {
        Some(true) => 1,
        Some(false) => 0,
        None => -1,
    }
}

/// `Py_TYPE(object)`, borrowed — the mirror a type mirror is.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_Type(object: *mut CPyObject) -> *mut CPyObject {
    let Some(object) = argument(object) else {
        return std::ptr::null_mut();
    };
    let Some(class) = crate::typedef::r#type(object) else {
        return std::ptr::null_mut();
    };
    pyobject::make_ref(class.as_ptr())
}

pub(super) fn ensure_linked() {
    std::hint::black_box(PyObject_GetAttrString as *const ());
    std::hint::black_box(PyObject_SetAttrString as *const ());
    std::hint::black_box(PyObject_HasAttrString as *const ());
    std::hint::black_box(PyObject_GetAttr as *const ());
    std::hint::black_box(PyObject_SetAttr as *const ());
    std::hint::black_box(PyObject_Str as *const ());
    std::hint::black_box(PyObject_Repr as *const ());
    std::hint::black_box(PyObject_IsTrue as *const ());
    std::hint::black_box(PyObject_Not as *const ());
    std::hint::black_box(PyObject_Size as *const ());
    std::hint::black_box(PyObject_GetItem as *const ());
    std::hint::black_box(PyObject_SetItem as *const ());
    std::hint::black_box(PyObject_Call as *const ());
    std::hint::black_box(PyObject_CallObject as *const ());
    std::hint::black_box(PyCallable_Check as *const ());
    std::hint::black_box(PyObject_IsInstance as *const ());
    std::hint::black_box(PyObject_Type as *const ());
}

// ── The vectorcall protocol ───────────────────────────────────────────────
//
// A vectorcall passes its arguments as a flat C array rather than a tuple and
// a dict: `args[..nargs]` are positional and `args[nargs..]` are the values
// for the names in `kwnames`, in order.  Pyre answers these by unpacking that
// array and going through its own call path -- the type's `tp_vectorcall` slot
// is an optimisation the caller may not assume was taken, which is why
// `PyVectorcall_Call` falls back to `tp_call` when a type declares no offset
// (`cpyext/src/call.c:120-129`).

/// The bit a caller sets in `nargsf` to say it left a spare slot before
/// `args[0]` that a callee may overwrite with its own receiver.
///
/// Pyre reads the array and never writes it, so the bit only has to be
/// stripped before the count is used.
const VECTORCALL_ARGUMENTS_OFFSET: usize = 1 << (usize::BITS - 1);

/// The positional count carried by an `nargsf`.
fn vectorcall_nargs(nargsf: usize) -> usize {
    nargsf & !VECTORCALL_ARGUMENTS_OFFSET
}

/// Call `callable` with a vectorcall argument vector.
///
/// # Safety
/// `args` must name `vectorcall_nargs(nargsf) + len(kwnames)` readable
/// pointers, and `kwnames` must be NULL or a tuple of `str`.
unsafe fn call_vector(
    callable: PyObjectRef,
    args: *const *mut CPyObject,
    nargsf: usize,
    kwnames: *mut CPyObject,
) -> Result<PyObjectRef, crate::PyError> {
    let nargs = vectorcall_nargs(nargsf);
    let kwnames = unsafe { pyobject::from_ref(kwnames) };
    if !kwnames.is_null() && !unsafe { pyre_object::is_tuple(kwnames) } {
        return Err(crate::PyError::type_error(
            "vectorcall keyword names must be a tuple",
        ));
    }

    // Every incoming value is pinned before anything below can collect, and
    // read back from the shadow stack afterwards.
    let roots = pyre_object::gc_roots::push_roots();
    let callable_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(callable);
    let kwnames_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(kwnames);
    let named = if kwnames.is_null() {
        0
    } else {
        unsafe { pyre_object::tupleobject::w_tuple_len(kwnames) }
    };
    let value_base = pyre_object::gc_roots::shadow_stack_len();
    for index in 0..nargs + named {
        let value = unsafe { pyobject::from_ref(*args.add(index)) };
        if value.is_null() {
            return Err(crate::PyError::new(
                crate::PyErrorKind::SystemError,
                "vectorcall argument vector holds a NULL",
            ));
        }
        roots.pin_root(value);
    }
    let value_at = |index: usize| pyre_object::gc_roots::shadow_stack_get(value_base + index);

    // The names come out before the values are read back, so that the
    // allocation each one makes cannot move an address already copied.
    let mut names: Vec<rustpython_wtf8::Wtf8Buf> = Vec::with_capacity(named);
    if named != 0 {
        let items = unsafe {
            pyre_object::tupleobject::w_tuple_items_copy_as_vec(
                pyre_object::gc_roots::shadow_stack_get(kwnames_slot),
            )
        };
        for name in items {
            if !unsafe { pyre_object::is_str(name) } {
                return Err(crate::PyError::type_error(
                    "vectorcall keyword name is not a string",
                ));
            }
            names.push(rustpython_wtf8::Wtf8Buf::from_string(
                unsafe { pyre_object::w_str_get_value(name) }.to_owned(),
            ));
        }
    }

    let positional: Vec<PyObjectRef> = (0..nargs).map(value_at).collect();
    if named == 0 {
        return crate::call::call_function_impl_result(
            pyre_object::gc_roots::shadow_stack_get(callable_slot),
            &positional,
        );
    }
    let keywords: Vec<(rustpython_wtf8::Wtf8Buf, PyObjectRef)> = names
        .into_iter()
        .enumerate()
        .map(|(index, name)| (name, value_at(nargs + index)))
        .collect();
    crate::eval::CURRENT_FRAME.with(|current| {
        let frame = current.get();
        if frame.is_null() {
            return Err(crate::PyError::runtime_error(
                "cpyext keyword call has no current frame",
            ));
        }
        crate::call::call_with_kwargs(
            unsafe { &mut *frame },
            pyre_object::gc_roots::shadow_stack_get(callable_slot),
            &positional,
            &keywords,
        )
    })
}

/// `PyObject_Vectorcall(callable, args, nargsf, kwnames)`.
///
/// # Safety
/// See [`call_vector`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_Vectorcall(
    callable: *mut CPyObject,
    args: *const *mut CPyObject,
    nargsf: usize,
    kwnames: *mut CPyObject,
) -> *mut CPyObject {
    let Some(callable) = argument(callable) else {
        return std::ptr::null_mut();
    };
    result(unsafe { call_vector(callable, args, nargsf, kwnames) })
}

/// `PyObject_VectorcallMethod(name, args, nargsf, kwnames)` — `args[0]` is the
/// object the method is looked up on, and stays the call's first argument.
///
/// # Safety
/// See [`call_vector`]; `args` must additionally hold at least one entry.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_VectorcallMethod(
    name: *mut CPyObject,
    args: *const *mut CPyObject,
    nargsf: usize,
    kwnames: *mut CPyObject,
) -> *mut CPyObject {
    let (Some(name), true) = (argument(name), vectorcall_nargs(nargsf) >= 1) else {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    };
    let Some(receiver) = argument(unsafe { *args }) else {
        return std::ptr::null_mut();
    };
    // The bound method carries the receiver, so the remaining vector is the
    // argument list and the offset bit no longer describes it.
    let Some(method) = trap(crate::baseobjspace::getattr(receiver, name)) else {
        return std::ptr::null_mut();
    };
    let nargs = vectorcall_nargs(nargsf) - 1;
    result(unsafe { call_vector(method, args.add(1), nargs, kwnames) })
}

/// `PyVectorcall_Call(callable, tuple, dict)` — the tuple/dict spelling of a
/// vectorcall, which is what a type's `tp_call` is set to when it declares one
/// (`cpyext/src/call.c:114-161`).
///
/// # Safety
/// `tuple` must be a tuple and `dict` NULL or a dict.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyVectorcall_Call(
    callable: *mut CPyObject,
    tuple: *mut CPyObject,
    dict: *mut CPyObject,
) -> *mut CPyObject {
    unsafe { PyObject_Call(callable, tuple, dict) }
}

/// `PyObject_CallNoArgs(callable)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_CallNoArgs(callable: *mut CPyObject) -> *mut CPyObject {
    let Some(callable) = argument(callable) else {
        return std::ptr::null_mut();
    };
    result(crate::call::call_function_impl_result(callable, &[]))
}

/// `PyObject_CallOneArg(callable, arg)`.
///
/// # Safety
/// `arg` must be a live reference.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_CallOneArg(
    callable: *mut CPyObject,
    arg: *mut CPyObject,
) -> *mut CPyObject {
    let Some(callable) = argument(callable) else {
        return std::ptr::null_mut();
    };
    let argument_vector = [arg];
    result(unsafe { call_vector(callable, argument_vector.as_ptr(), 1, std::ptr::null_mut()) })
}
