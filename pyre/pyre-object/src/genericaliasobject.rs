//! W_GenericAlias — Python `types.GenericAlias` (PEP 585).
//!
//! PyPy equivalent: lib_pypy/_pypy_generic_alias.py → GenericAlias
//!
//! Represents `X[Y]` parameterized generics (e.g. `list[int]`).
//! Produced by the `__class_getitem__` classmethod on builtin containers.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::pyobject::*;
use pyre_macros::pyre_class;

/// Python parameterized-generic alias object (PEP 585).
///
/// Layout: `[ob_type | origin | args | parameters | unpacked]`
///
/// - `origin`: the parameterized class (`list` in `list[int]`) — `_origin`
/// - `args`: tuple of the type arguments (`(int,)`) — `_args`
/// - `parameters`: tuple of free type variables — `_parameters`
/// - `unpacked`: set by `_make_starred` for `*Ts` unpacking — `__unpacked__`
#[pyre_class("types.GenericAlias", static_name = "GENERIC_ALIAS")]
pub struct W_GenericAlias {
    pub origin: PyObjectRef,
    pub args: PyObjectRef,
    pub parameters: PyObjectRef,
    pub unpacked: bool,
}

/// Check if an object is a GenericAlias.
#[inline]
pub unsafe fn is_generic_alias(obj: PyObjectRef) -> bool {
    py_type_check(obj, &GENERIC_ALIAS_TYPE)
}

/// Allocate a new GenericAlias.
///
/// `args` and `parameters` are pre-built tuples; the interpreter wraps a
/// bare item into a 1-tuple and collects the parameters before calling
/// (GenericAlias.__new__ in `_pypy_generic_alias.py`).
pub fn w_generic_alias_new(
    origin: PyObjectRef,
    args: PyObjectRef,
    parameters: PyObjectRef,
) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`).
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(origin);
    crate::gc_roots::pin_root(args);
    crate::gc_roots::pin_root(parameters);
    W_GenericAlias::allocate(W_GenericAlias {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        origin,
        args,
        parameters,
        unpacked: false,
    })
}

/// `_origin` reader.
///
/// # Safety
/// `obj` must point to a valid `W_GenericAlias`.
pub unsafe fn w_generic_alias_get_origin(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const W_GenericAlias)).origin
}

/// `_args` reader (the type-argument tuple).
///
/// # Safety
/// `obj` must point to a valid `W_GenericAlias`.
pub unsafe fn w_generic_alias_get_args(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const W_GenericAlias)).args
}

/// `_parameters` reader (the free-type-variable tuple).
///
/// # Safety
/// `obj` must point to a valid `W_GenericAlias`.
pub unsafe fn w_generic_alias_get_parameters(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const W_GenericAlias)).parameters
}

/// `__unpacked__` reader.
///
/// # Safety
/// `obj` must point to a valid `W_GenericAlias`.
pub unsafe fn w_generic_alias_get_unpacked(obj: PyObjectRef) -> bool {
    (*(obj as *const W_GenericAlias)).unpacked
}

/// `__unpacked__` writer — `_make_starred` marks the alias unpacked.
///
/// # Safety
/// `obj` must point to a valid `W_GenericAlias`.
pub unsafe fn w_generic_alias_set_unpacked(obj: PyObjectRef, unpacked: bool) {
    (*(obj as *mut W_GenericAlias)).unpacked = unpacked;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intobject::w_int_new;
    use crate::tupleobject::w_tuple_new;

    #[test]
    fn test_generic_alias_create() {
        let origin = w_int_new(1); // stand-in for `list`
        let args = w_tuple_new(vec![w_int_new(2)]); // stand-in for `(int,)`
        let params = w_tuple_new(vec![]);
        let ga = w_generic_alias_new(origin, args, params);
        unsafe {
            assert!(is_generic_alias(ga));
            assert!(!is_int(ga));
            assert_eq!(crate::w_tuple_len(w_generic_alias_get_args(ga)), 1);
            assert!(!w_generic_alias_get_unpacked(ga));
        }
    }

    #[test]
    fn test_generic_alias_unpacked() {
        let origin = w_int_new(1);
        let args = w_tuple_new(vec![]);
        let params = w_tuple_new(vec![]);
        let ga = w_generic_alias_new(origin, args, params);
        unsafe {
            w_generic_alias_set_unpacked(ga, true);
            assert!(w_generic_alias_get_unpacked(ga));
        }
    }

    #[test]
    fn w_generic_alias_size_matches_descr() {
        assert_eq!(
            <W_GenericAlias as crate::lltype::GcType>::SIZE,
            W_GENERIC_ALIAS_OBJECT_SIZE
        );
    }
}
