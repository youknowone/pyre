//! Prebuilt `pypy/interpreter/gateway.py` gateway object layouts.
//!
//! `interp2app` is a W_Root holding a Code reference, not a Code subclass.
//! The concrete BuiltinCode and its typed Rust callable stay in the
//! interpreter crate, just as GetSetProperty holds its callables as W_Root
//! references without depending on the interpreter's OperationError.

#![allow(non_camel_case_types)]

use crate::pyobject::*;
use pyre_macros::pyre_class;

/// `gateway.py interp2app.__new__`'s declaration metadata. This internal
/// W_Root has no app-level TypeDef: wrapping it produces a Function instead.
#[pyre_class("gateway.interp2app", static_name = "INTERP2APP")]
pub struct interp2app {
    pub _code: PyObjectRef,
    pub __name__: &'static str,
    pub name: &'static str,
    pub as_classmethod: bool,
    pub self_type: *const PyType,
    pub _explicit_text_sig: Option<&'static str>,
    pub _is_type_method: bool,
}

impl interp2app {
    /// The BuiltinCode is constructed by gateway.py in the interpreter
    /// crate. It is a prebuilt immortal Code; no typed callable is erased or
    /// converted to an integer at this boundary. This constructor covers
    /// gateways without host defaults; _staticdefs/unwrap-spec construction
    /// must be connected before admitting gateways with defaults.
    ///
    /// # Safety
    /// code must be a live prebuilt BuiltinCode, not an app-level Function.
    pub unsafe fn new(code: PyObjectRef, name: &'static str) -> PyObjectRef {
        crate::lltype::malloc_typed(Self {
            ob: PyObject {
                ob_type: &INTERP2APP_TYPE,
                w_class: PY_NULL,
            },
            _code: code,
            __name__: name,
            name,
            as_classmethod: false,
            self_type: std::ptr::null(),
            _explicit_text_sig: None,
            _is_type_method: false,
        }) as PyObjectRef
    }
}

/// # Safety
/// obj must be a live PyObject pointer or null.
pub unsafe fn is_interp2app(obj: PyObjectRef) -> bool {
    !obj.is_null() && unsafe { py_type_check(obj, &INTERP2APP_TYPE) }
}
