//! Pointer hiding — PyPy: `pypy/module/_cffi_backend/hide_reveal.py`.
//!
//! Pyre's stable GC allocations are the non-moving address-space case of
//! `HideRevealCast`: the hidden pointer is the object address, and reveal
//! first asks the collector whether the address is still managed before
//! checking the payload type and flavor.

use pyre_object::PyObjectRef;

use super::cdataobj::{self, W_CData};

/// `HideRevealCast.hide_object` for a `W_CDataHandle`.
pub fn hide_object(w_handle: PyObjectRef) -> *mut u8 {
    w_handle.cast::<u8>()
}

/// `HideRevealCast.reveal_object` for a `W_CDataHandle`.
pub fn reveal_object(ptr: *mut u8) -> Option<PyObjectRef> {
    if ptr.is_null() || !pyre_object::gc_hook::try_gc_owns_object(ptr) {
        return None;
    }
    let obj = ptr.cast::<pyre_object::PyObject>();
    W_CData::from_obj(obj)
        .filter(|cdata| cdata.flavor == cdataobj::FLAVOR_HANDLE)
        .map(|_| obj)
}
