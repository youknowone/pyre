//! `types.GenericAlias` for C -- `Include/genericaliasobject.h`.

use super::pyobject::CPyObject;

/// `Py_GenericAlias(origin, args)` — what `origin[args]` builds.
///
/// A bare `args` is wrapped in a one-element tuple, the same way
/// `GenericAlias.__new__` does, so a C `__class_getitem__` written over this
/// answers what the Python one would.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn Py_GenericAlias(
    origin: *mut CPyObject,
    args: *mut CPyObject,
) -> *mut CPyObject {
    super::object::realize_all([origin, args]);
    let Some([origin, args]) = super::object::arguments([origin, args]) else {
        return std::ptr::null_mut();
    };
    super::object::result(crate::_pypy_generic_alias::make_generic_alias(origin, args))
}

pub(super) fn ensure_linked() {
    std::hint::black_box(Py_GenericAlias as *const ());
}
