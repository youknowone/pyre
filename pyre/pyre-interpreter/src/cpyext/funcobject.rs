//! The bound-method entry points -- PyPy `cpyext/funcobject.py`.

use super::object::{argument, result};
use super::pyobject::{self, CPyObject};

/// `funcobject.py PyMethod_New(func, self)` — bind `receiver` to
/// `function`, the way an attribute lookup on an instance does.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyMethod_New(
    function: *mut CPyObject,
    receiver: *mut CPyObject,
) -> *mut CPyObject {
    super::object::realize_all([function, receiver]);
    let (Some(function), Some(receiver)) = (argument(function), argument(receiver)) else {
        return std::ptr::null_mut();
    };
    result(Ok(pyre_object::w_method_new(
        function,
        receiver,
        pyre_object::PY_NULL,
    )))
}

/// The member `reader` names on `method`, borrowed, or NULL with a
/// `SystemError` when `method` is not a bound method.
///
/// Borrowed is what both readers answer with: the member is reachable through
/// the method for as long as the caller holds it.
unsafe fn method_member(
    method: *mut CPyObject,
    reader: unsafe fn(pyre_object::PyObjectRef) -> pyre_object::PyObjectRef,
) -> *mut CPyObject {
    let Some(object) = argument(method) else {
        return std::ptr::null_mut();
    };
    if !unsafe { pyre_object::function::is_method(object) } {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    pyobject::borrow_from(method, unsafe { reader(object) })
}

/// `funcobject.py PyMethod_Function(method)` — the callable the binding
/// wraps, borrowed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyMethod_Function(method: *mut CPyObject) -> *mut CPyObject {
    unsafe { method_member(method, pyre_object::function::w_method_get_func) }
}

/// `funcobject.py PyMethod_Self(method)` — the receiver the binding
/// carries, borrowed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyMethod_Self(method: *mut CPyObject) -> *mut CPyObject {
    unsafe { method_member(method, pyre_object::function::w_method_get_self) }
}
