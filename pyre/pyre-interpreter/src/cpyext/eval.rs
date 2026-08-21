//! Running code and the namespaces it runs in -- PyPy `cpyext/eval.py`.

use super::pyobject::{self, CPyObject};
use pyre_object::{PY_NULL, PyObjectRef};

/// The namespace a name lookup running here would fall back to —
/// `eval.py:32-45 PyEval_GetBuiltins`.
///
/// Borrowed, as `result_borrowed=True` says: what keeps it alive is the frame
/// the namespace was read off, or the `builtins` module itself.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyEval_GetBuiltins() -> *mut CPyObject {
    pyobject::borrow_mirror(namespace())
}

/// The running frame's `__builtins__`, and the `builtins` module's own
/// namespace where there is no frame.
///
/// A frame whose globals name no `__builtins__` is answered for the same way
/// as no frame at all: this reports nothing, and every caller reads a null as
/// "the runtime is gone".
fn namespace() -> PyObjectRef {
    if let Some(w_builtins) = frame_builtins() {
        // A frame may name the module rather than the namespace.
        if unsafe { pyre_object::is_dict(w_builtins) } {
            return w_builtins;
        }
        return dict_of(w_builtins);
    }
    match crate::importing::get_sys_module("builtins") {
        Some(module) => dict_of(module),
        None => PY_NULL,
    }
}

fn frame_builtins() -> Option<PyObjectRef> {
    let ec = crate::call::getexecutioncontext() as *mut crate::PyExecutionContext;
    if ec.is_null() {
        return None;
    }
    let frame = unsafe { (*ec).gettopframe_nohidden() };
    if frame.is_null() {
        return None;
    }
    let w_globals = unsafe { (*frame).get_w_globals() };
    if w_globals.is_null() {
        return None;
    }
    unsafe { pyre_object::w_dict_getitem_str(w_globals, "__builtins__") }.filter(|w| !w.is_null())
}

/// `w_obj`'s own namespace, read back after the call because building one
/// allocates.
fn dict_of(w_obj: PyObjectRef) -> PyObjectRef {
    let roots = pyre_object::gc_roots::push_roots();
    let slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(w_obj);
    crate::baseobjspace::getdict_native(pyre_object::gc_roots::shadow_stack_get(slot))
}

pub(super) fn ensure_linked() {
    std::hint::black_box(PyEval_GetBuiltins as *const ());
}
