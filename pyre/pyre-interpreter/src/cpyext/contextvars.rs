//! Context variables -- PyPy `cpyext/contextvars.py`.
//!
//! Written over the `_contextvars` module, as upstream's is: the four entry
//! points are the calls a caller would make in Python, and none of the state
//! lives on this side.

use super::object::result;
use super::pyerrors::trap;
use super::pyobject::{self, CPyObject};
use pyre_object::PyObjectRef;
use std::ffi::{CStr, c_char, c_int};

/// `_contextvars.ContextVar`, or a recorded error.
fn context_var_class() -> Result<PyObjectRef, crate::PyError> {
    let module = super::import_::import_module("_contextvars")?;
    crate::baseobjspace::getattr_str(module, "ContextVar")
}

/// The variable behind `raw`, having checked that it is one — the
/// `isinstance(ovar, ContextVar)` every one of these entry points opens with.
fn variable(raw: *mut CPyObject) -> Result<PyObjectRef, crate::PyError> {
    let w_var = unsafe { pyobject::from_ref(raw) };
    if w_var.is_null() {
        return Err(crate::PyError::type_error(
            "an instance of ContextVar was expected",
        ));
    }
    let roots = pyre_object::gc_roots::push_roots();
    let slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(w_var);
    let class = context_var_class()?;
    let w_var = pyre_object::gc_roots::shadow_stack_get(slot);
    if !unsafe { crate::baseobjspace::isinstance_w(w_var, class) } {
        return Err(crate::PyError::type_error(
            "an instance of ContextVar was expected",
        ));
    }
    Ok(w_var)
}

/// Call `name` on the variable behind `raw` with `argument` where there is
/// one — the shape `set`, `reset` and `get` share.
fn call_method(
    raw: *mut CPyObject,
    name: &str,
    argument: *mut CPyObject,
) -> Result<PyObjectRef, crate::PyError> {
    let w_var = variable(raw)?;
    let roots = pyre_object::gc_roots::push_roots();
    let var_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(w_var);
    let w_argument = unsafe { pyobject::from_ref(argument) };
    let argument_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(w_argument);
    let method =
        crate::baseobjspace::getattr_str(pyre_object::gc_roots::shadow_stack_get(var_slot), name)?;
    let method_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(method);
    let arguments: Vec<PyObjectRef> = match argument.is_null() {
        true => Vec::new(),
        false => vec![pyre_object::gc_roots::shadow_stack_get(argument_slot)],
    };
    crate::call::call_function_impl_result(
        pyre_object::gc_roots::shadow_stack_get(method_slot),
        &arguments,
    )
}

/// `contextvars.py:11-28 PyContextVar_New`.
///
/// `default` is the keyword `ContextVar` declares it as; a null one is the
/// variable with no default rather than a default of `None`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyContextVar_New(
    name: *const c_char,
    default_value: *mut CPyObject,
) -> *mut CPyObject {
    let made = (|| -> Result<PyObjectRef, crate::PyError> {
        let text = match name.is_null() {
            true => String::new(),
            false => unsafe { CStr::from_ptr(name) }
                .to_string_lossy()
                .into_owned(),
        };
        let class = context_var_class()?;
        let roots = pyre_object::gc_roots::push_roots();
        let class_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(class);
        let default = unsafe { pyobject::from_ref(default_value) };
        let default_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(default);
        // The name is minted last, so nothing above it is a pre-move address.
        let name_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(pyre_object::w_str_new(&text));
        let class = pyre_object::gc_roots::shadow_stack_get(class_slot);
        let arguments = [pyre_object::gc_roots::shadow_stack_get(name_slot)];
        if default_value.is_null() {
            return crate::call::call_function_impl_result(class, &arguments);
        }
        super::object::call_keyword(
            class,
            &arguments,
            "default",
            pyre_object::gc_roots::shadow_stack_get(default_slot),
        )
    })();
    result(made)
}

/// `contextvars.py:30-37 PyContextVar_Set` — the token that undoes it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyContextVar_Set(
    var: *mut CPyObject,
    value: *mut CPyObject,
) -> *mut CPyObject {
    result(call_method(var, "set", value))
}

/// `contextvars.py:39-47 PyContextVar_Reset`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyContextVar_Reset(var: *mut CPyObject, token: *mut CPyObject) -> c_int {
    match trap(call_method(var, "reset", token)) {
        Some(_) => 0,
        None => -1,
    }
}

/// `contextvars.py:49-71 PyContextVar_Get`.
///
/// A variable with neither a value in this context nor a default is not an
/// error: `value` is left null and the call still reports success, which is
/// how a caller tells "unset" from "failed".
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyContextVar_Get(
    var: *mut CPyObject,
    default_value: *mut CPyObject,
    value: *mut *mut CPyObject,
) -> c_int {
    let w_value = match call_method(var, "get", default_value) {
        Ok(w_value) => w_value,
        Err(error) => {
            // Recorded first, so the match runs against the exception the
            // caller would have seen rather than against a tag: a class
            // derived from `LookupError` is one too.
            super::pyerrors::set_pending_error(error);
            let unset = unsafe {
                super::pyerrors::PyErr_ExceptionMatches(super::pyerrors::PyExc_LookupError)
            };
            if default_value.is_null() && unset != 0 {
                unsafe { super::pyerrors::PyErr_Clear() };
                unsafe { *value = std::ptr::null_mut() };
                return 0;
            }
            return -1;
        }
    };
    unsafe { *value = pyobject::make_ref(w_value) };
    0
}

pub(super) fn ensure_linked() {
    std::hint::black_box(PyContextVar_New as *const ());
    std::hint::black_box(PyContextVar_Set as *const ());
    std::hint::black_box(PyContextVar_Reset as *const ());
    std::hint::black_box(PyContextVar_Get as *const ());
}
