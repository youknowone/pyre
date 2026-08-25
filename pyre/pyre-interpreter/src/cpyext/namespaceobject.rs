//! `types.SimpleNamespace` seen from C -- `Objects/namespaceobject.c`.
//!
//! Only the constructor is exported. It is private API, so an extension that
//! calls it is built against the interpreter's own headers; the two test
//! modules `test_importlib` gates on are exactly that, and both use it to hand
//! back an attribute bag rather than a dict.

use super::pyobject::{self, CPyObject};

/// `_PyNamespace_New(kwds)` -- a fresh namespace whose dictionary starts as a
/// copy of `kwds`, or empty when `kwds` is NULL.
///
/// Answers a new reference, or NULL with the exception set.
///
/// # Safety
/// `kwds` must be null or a live mirror of a mapping.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _PyNamespace_New(kwds: *mut CPyObject) -> *mut CPyObject {
    let namespace = match crate::module::sys::vm::new_simple_namespace_instance() {
        Ok(namespace) => namespace,
        Err(error) => {
            super::pyerrors::set_pending_error(error);
            return std::ptr::null_mut();
        }
    };
    if kwds.is_null() {
        return pyobject::make_ref(namespace);
    }
    // Realizing `kwds` and reading the namespace's dictionary both allocate,
    // and the namespace is a moving object held only by this local.
    let _roots = pyre_object::gc_roots::push_roots();
    let namespace_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(namespace);
    let source = unsafe { pyobject::from_ref(kwds) };
    let _ = pyre_object::gc_roots::pin_root(source);
    let namespace = pyre_object::gc_roots::shadow_stack_get(namespace_slot);
    let dict = crate::baseobjspace::getdict_native(namespace);
    let _ = pyre_object::gc_roots::pin_root(dict);
    let source = pyre_object::gc_roots::shadow_stack_get(namespace_slot + 1);
    let dict = pyre_object::gc_roots::shadow_stack_get(namespace_slot + 2);
    if let Err(error) = crate::opcode_ops::dict_update_value(dict, source) {
        super::pyerrors::set_pending_error(error);
        return std::ptr::null_mut();
    }
    pyobject::make_ref(pyre_object::gc_roots::shadow_stack_get(namespace_slot))
}

pub(super) fn ensure_linked() {
    std::hint::black_box(_PyNamespace_New as *const ());
}
