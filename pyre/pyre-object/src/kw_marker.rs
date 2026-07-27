//! Sentinel value tagging the builtin-kwargs marker dict.
//!
//! `call_with_kwargs` smuggles keyword arguments to a flat-ABI builtin as a
//! trailing dict holding the `__pyre_kw__` key.  The value stored under that
//! key is this immortal, Python-invisible sentinel; detection compares the
//! value by identity, so a positional user dict that merely contains a
//! `__pyre_kw__` string key is never mistaken for the marker.

use crate::pyobject::*;

static KW_MARKER_TYPE: PyType = new_pytype("__pyre_kw_marker__");

#[repr(C)]
struct KwMarkerSentinel {
    ob_header: PyObject,
}

static KW_MARKER_SENTINEL: KwMarkerSentinel = KwMarkerSentinel {
    ob_header: PyObject {
        ob_type: &KW_MARKER_TYPE as *const PyType,
        w_class: std::ptr::null_mut(),
    },
};

/// The sentinel stored under the `__pyre_kw__` key of a builtin-kwargs marker
/// dict.  Its immortal static identity is unforgeable by Python code.
pub fn w_kw_marker_sentinel() -> PyObjectRef {
    &KW_MARKER_SENTINEL as *const KwMarkerSentinel as *mut PyObject
}

/// Cached immortal `"__pyre_kw__"` key `W_str`.  Builtin-kwargs packing stores
/// the sentinel under this key on every keyworded call; minting a fresh
/// `w_str_new("__pyre_kw__")` per call allocated one never-freed immortal string
/// per call, so the constant key is allocated once and reused.  Immortal by
/// design — it is shared across every (collectable) marker dict, which on
/// collection drops only the borrowed pointer.
pub fn w_kw_marker_key() -> PyObjectRef {
    static KW_MARKER_KEY: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *KW_MARKER_KEY.get_or_init(|| crate::unicodeobject::w_str_new("__pyre_kw__") as usize)
        as PyObjectRef
}

/// `true` when `value` is the marker sentinel (pointer identity).
#[inline]
pub fn is_kw_marker_sentinel(value: PyObjectRef) -> bool {
    std::ptr::eq(
        value as *const PyObject,
        w_kw_marker_sentinel() as *const PyObject,
    )
}
