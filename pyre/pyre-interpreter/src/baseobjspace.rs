//! ObjSpace — Python object operation dispatch.
#![allow(non_camel_case_types, non_snake_case)]
//!
//! pypy/interpreter/baseobjspace.py — the core ObjSpace interface.
//!
//! Binary/unary operation dispatch (add, sub, compare, etc.) lives in
//! `crate::objspace::descroperation`; printf-style formatting lives in
//! `crate::objspace::std::formatting`. Both are re-exported here so
//! existing `crate::baseobjspace::add` paths continue to resolve.

// Suppress unsafe-in-unsafe-fn warnings; our unsafe fns are inherently
// working with raw pointers throughout and wrapping every call in an
// additional unsafe block adds noise without safety benefit.
#![allow(unsafe_op_in_unsafe_fn)]

use malachite_bigint::BigInt;

use std::cell::{Cell, RefCell};
use std::collections::HashMap;

use crate::function::is_function;
pub use crate::{PyError, PyErrorKind, PyResult};
use pyre_object::unicodeobject::is_str;
use pyre_object::*;
use rustpython_wtf8::{CodePoint, Wtf8, Wtf8Buf};

// ── Re-exports from split-out modules ────────────────────────────────
pub use crate::objspace::descroperation::*;
pub(crate) use crate::objspace::std::formatting::{format_g_like, normalise_exponent};

// ── Pending dict-key-callback error slot ─────────────────────────────
// The `r_dict(eq_w, hash_w)` callbacks (`pyre_object_hash_w_trampoline`
// / `pyre_object_eq_w_trampoline` in pyre-jit) cannot return a `Result`
// across the pyre-object dict probe, so on a raising `__hash__` or
// `__eq__` they stash the concrete PyError here.  Dict entry gates call
// `take_pending_hash_error` after a checked dict op returns
// `DictKeyError` to recover whichever exception was raised.
thread_local! {
    static PENDING_HASH_ERROR: Cell<Option<PyError>> = const { Cell::new(None) };
}

pub fn set_pending_hash_error(e: PyError) {
    PENDING_HASH_ERROR.with(|cell| cell.set(Some(e)));
}

/// `dont_look_inside`: the `PENDING_HASH_ERROR` thread-local `.with`
/// read has no extractable graph; the call stays a residual.
#[majit_macros::dont_look_inside]
pub fn take_pending_hash_error() -> PyError {
    PENDING_HASH_ERROR.with(|cell| {
        cell.take()
            .unwrap_or_else(|| PyError::type_error("unhashable type"))
    })
}

/// Compatibility alias for PyPy's base-object type.
/// PyPy frequently models interpreter values as subclasses of `W_Root`.
pub type W_Root = PyObjectRef;

/// Compatibility marker for a type mismatch in descriptor lookup.
#[derive(Debug, Clone)]
pub struct DescrMismatch;

/// Compatibility marker for lock-sensitive APIs that are disabled under
/// this no-GIL runtime.
#[derive(Debug, Clone)]
pub struct CannotHaveLock;

/// Minimal compatibility placeholder for PyPy-style cache objects.
#[derive(Debug, Default)]
pub struct SpaceCache {
    space: PyObjectRef,
    _entries: RefCell<HashMap<usize, PyObjectRef>>,
}

impl SpaceCache {
    pub fn new(space: PyObjectRef) -> Self {
        Self {
            space,
            _entries: RefCell::new(HashMap::new()),
        }
    }

    #[inline]
    pub fn getorbuild(&self, _key: PyObjectRef) -> PyObjectRef {
        std::ptr::null_mut()
    }

    #[inline]
    pub fn ready(&self, _result: PyObjectRef) {}
}

/// Compatibility cache variant with `callable(self)` construction path.
#[derive(Debug, Default)]
pub struct InternalSpaceCache {
    base: SpaceCache,
}

impl InternalSpaceCache {
    pub fn new(space: PyObjectRef) -> Self {
        Self {
            base: SpaceCache::new(space),
        }
    }

    #[inline]
    pub fn getorbuild<F>(&self, f: F) -> PyObjectRef
    where
        F: FnOnce(PyObjectRef) -> PyObjectRef,
    {
        let _ = self.base.space;
        f(std::ptr::null_mut())
    }
}

/// Compatibility helper used by `ObjSpace` bootstrap in PyPy.
#[derive(Debug, Default)]
pub struct AppExecCache {
    base: SpaceCache,
}

impl AppExecCache {
    pub fn new(space: PyObjectRef) -> Self {
        Self {
            base: SpaceCache::new(space),
        }
    }

    pub fn build(&self, _source: PyObjectRef) -> PyObjectRef {
        let _ = self.base.space;
        std::ptr::null_mut()
    }
}

/// Very small compatibility object for PyPy's `ObjSpace` interface.
/// The full object-space API is implemented as free functions in this module.
#[derive(Debug, Default)]
pub struct ObjSpace {
    fromcache: Option<PyObjectRef>,
}

impl ObjSpace {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn fromcache<T, F>(&self, mut build: F, cache: &SpaceCache) -> T
    where
        T: Default,
        F: FnMut(&SpaceCache) -> T,
    {
        let _ = cache.getorbuild(std::ptr::null_mut());
        build(cache)
    }
}

// ── Cell unwrap ──────────────────────────────────────────────────────
// CPython 3.13 unified locals+cells means LoadFast can return cell
// objects. All operations must transparently unwrap cells.
// PyPy: each opcode implementation calls space.unwrap_cell() implicitly.

/// Unwrap a cell object to its contents. Non-cells pass through.
#[inline(always)]
pub fn unwrap_cell(obj: PyObjectRef) -> PyObjectRef {
    if obj.is_null() {
        return obj;
    }
    if unsafe { is_cell(obj) } {
        let inner = unsafe { w_cell_get(obj) };
        if !inner.is_null() {
            return inner;
        }
        // Cell with null content — return cell itself (caller will handle)
        return obj;
    }
    obj
}

/// pypy/interpreter/baseobjspace.py `issubtype_w` — `cls` is in
/// `w_type.mro_w`. Uses the cached MRO when present, otherwise
/// recomputes via `compute_default_mro`.
pub(crate) unsafe fn issubtype_w(w_type: PyObjectRef, cls: PyObjectRef) -> bool {
    if w_type.is_null() {
        return false;
    }
    // PyPy's issubtype_w is only valid for type objects.  Use the same
    // object-space test as abstractinst.py (`space.isinstance_w(...,
    // space.w_type)`) instead of peeking at Rust layout internals.
    if !is_type_like_w(w_type) {
        return false;
    }
    let mro_ptr = w_type_get_mro(w_type);
    if !mro_ptr.is_null() {
        return (*mro_ptr).iter().any(|&t| std::ptr::eq(t, cls));
    }
    compute_default_mro(w_type)
        .iter()
        .any(|&t| std::ptr::eq(t, cls))
}

/// pypy/interpreter/baseobjspace.py:1359 `exception_is_valid_obj_as_class_w`.
///
///   def exception_is_valid_obj_as_class_w(self, w_obj):
///       if not self.isinstance_w(w_obj, self.w_type):
///           return False
///       return self.issubtype_w(w_obj, self.w_BaseException)
///
/// Canonical `BaseException` comes from the EXC_CLASS_REGISTRY populated at
/// `make_exc_type` time — not from the mutable builtins dict — so a user
/// rebinding `builtins.BaseException` cannot redirect the gate.
pub unsafe fn exception_is_valid_obj_as_class_w(w_obj: PyObjectRef) -> bool {
    if !is_type_like_w(w_obj) {
        return false;
    }
    let Some(base_exc) = crate::builtins::lookup_exc_class("BaseException") else {
        return false;
    };
    issubtype_w(w_obj, base_exc)
}

/// pypy/interpreter/baseobjspace.py:1364-1365 `exception_is_valid_class_w`.
///
///   def exception_is_valid_class_w(self, w_cls):
///       return self.issubtype_w(w_cls, self.w_BaseException)
///
/// Like `exception_is_valid_obj_as_class_w` but skips the
/// `isinstance_w(w_cls, w_type)` precheck — the caller already knows
/// `w_cls` is a class object.
pub unsafe fn exception_is_valid_class_w(w_cls: PyObjectRef) -> bool {
    let Some(base_exc) = crate::builtins::lookup_exc_class("BaseException") else {
        return false;
    };
    issubtype_w(w_cls, base_exc)
}

/// pypy/interpreter/baseobjspace.py:1367-1368 `exception_getclass`.
///
///   def exception_getclass(self, w_obj):
///       return self.type(w_obj)
pub fn exception_getclass(w_obj: PyObjectRef) -> PyObjectRef {
    crate::typedef::r#type(w_obj).unwrap_or(pyre_object::PY_NULL)
}

/// True when `obj` is a `BlockingIOError` whose constructor took the numeric
/// third argument as `characters_written` — recognised by `args_w[2]` still
/// being an int (every other 2..=5-argument form trims `args_w` to two
/// elements).  Gates the `characters_written` reader and suppresses the
/// `filename` derivation for that argument (`interp_exceptions.py` `_init_error`).
fn exc_blocking_written(obj: PyObjectRef) -> bool {
    let args = unsafe { pyre_object::interp_exceptions::w_exception_get_args(obj) };
    let n = unsafe { pyre_object::w_tuple_len(args) };
    if n < 3 {
        return false;
    }
    let Some(v) = (unsafe { pyre_object::w_tuple_getitem(args, 2) }) else {
        return false;
    };
    if !unsafe { pyre_object::is_int(v) } {
        return false;
    }
    let Some(blocking) = crate::builtins::lookup_exc_class("BlockingIOError") else {
        return false;
    };
    unsafe { isinstance_w(obj, blocking) }
}

/// `interp_exceptions.py:1357-1424 W_SyntaxError.descr_init` parses the
/// constructor arguments into `msg` (`args_w[0]`) and, when a second
/// argument is supplied, a `(filename, lineno, offset, text[, end_lineno,
/// end_offset])` details tuple, exposing each piece as a
/// `readwrite_attrproperty_w` slot whose class default is `None`.  Pyre
/// keeps no dedicated SyntaxError slots, so an explicit `e.lineno = ...`
/// write lands in the hasdict instance dict: read it first so the write
/// wins, then derive the construct-time value from `args_w`, and finally
/// fall back to the `None` class default.
fn syntax_error_attr(obj: PyObjectRef, name: &str) -> PyObjectRef {
    let w_dict = getdict_backing(obj);
    if !w_dict.is_null() {
        if let Some(v) = unsafe { pyre_object::w_dict_getitem_str(w_dict, name) } {
            return v;
        }
    }
    let args = unsafe { pyre_object::interp_exceptions::w_exception_get_args(obj) };
    let n = unsafe { pyre_object::w_tuple_len(args) };
    if name == "msg" {
        if n >= 1 {
            if let Some(v) = unsafe { pyre_object::w_tuple_getitem(args, 0) } {
                return v;
            }
        }
        return w_none();
    }
    // `print_file_and_line` is a vestigial slot with no derivation.
    if name == "print_file_and_line" {
        return w_none();
    }
    // The location attributes derive from the `args_w[1]` details tuple.
    if n == 2 {
        if let Some(details) = unsafe { pyre_object::w_tuple_getitem(args, 1) } {
            if unsafe { pyre_object::is_tuple(details) } {
                let dn = unsafe { pyre_object::w_tuple_len(details) };
                let idx: usize = match name {
                    "filename" => 0,
                    "lineno" => 1,
                    "offset" => 2,
                    "text" => 3,
                    "end_lineno" => 4,
                    _ => 5, // "end_offset"
                };
                if idx < dn {
                    if let Some(v) = unsafe { pyre_object::w_tuple_getitem(details, idx as i64) } {
                        return v;
                    }
                }
            }
        }
    }
    w_none()
}

/// pypy/interpreter/baseobjspace.py:1370-1371 `exception_issubclass_w`.
///
///   def exception_issubclass_w(self, w_cls1, w_cls2):
///       return self.issubtype_w(w_cls1, w_cls2)
pub unsafe fn exception_issubclass_w(w_cls1: PyObjectRef, w_cls2: PyObjectRef) -> bool {
    unsafe { issubtype_w(w_cls1, w_cls2) }
}

/// abstractinst.py:18-31 `_get_bases(space, w_cls)`.
/// Returns `Some(bases_tuple)` when `getattr(w_cls, "__bases__")` exists
/// and is a tuple, `None` when the attribute is missing or not a tuple.
/// AttributeError is swallowed; other errors propagate.
fn _get_bases(w_cls: PyObjectRef) -> Result<Option<PyObjectRef>, PyError> {
    let w_bases = match getattr_str(w_cls, "__bases__") {
        Ok(b) => b,
        Err(e) if e.kind == PyErrorKind::AttributeError => return Ok(None),
        Err(e) => return Err(e),
    };
    if w_bases.is_null() {
        return Ok(None);
    }
    if unsafe { is_tuple(w_bases) } {
        Ok(Some(w_bases))
    } else {
        Ok(None)
    }
}

/// abstractinst.py:33-34 `abstract_isclass_w(space, w_obj)`.
fn abstract_isclass_w(w_obj: PyObjectRef) -> Result<bool, PyError> {
    Ok(_get_bases(w_obj)?.is_some())
}

/// abstractinst.py:36-38 `check_class(space, w_obj, msg)`. Raises
/// `TypeError(msg)` when `w_obj` lacks a tuple-valued `__bases__`.
fn check_class(w_obj: PyObjectRef, msg: &str) -> Result<(), PyError> {
    if !abstract_isclass_w(w_obj)? {
        return Err(PyError::type_error(msg.to_string()));
    }
    Ok(())
}

/// abstractinst.py:74-88 `p_recursive_isinstance_type_w`. Assumes
/// `w_type` is a real type object: tries the MRO walk via `isinstance_w`
/// first, then consults `w_inst.__class__` to honour any custom class
/// override.
unsafe fn p_recursive_isinstance_type_w(
    w_inst: PyObjectRef,
    w_type: PyObjectRef,
) -> Result<bool, PyError> {
    if isinstance_w(w_inst, w_type) {
        return Ok(true);
    }
    let w_abstractclass = match getattr_str(w_inst, "__class__") {
        Ok(cls) => cls,
        Err(e) if e.kind == PyErrorKind::AttributeError => return Ok(false),
        Err(e) => return Err(e),
    };
    let w_inst_type = crate::typedef::r#type(w_inst).unwrap_or(pyre_object::PY_NULL);
    if !std::ptr::eq(w_abstractclass, w_inst_type) && is_type_like_w(w_abstractclass) {
        return Ok(issubtype_w(w_abstractclass, w_type));
    }
    Ok(false)
}

/// abstractinst.py:53-72 `p_recursive_isinstance_w`. The Py3 port drops
/// the `W_ClassObject`/`W_ObjectObject` Py2 fast path. Validates
/// `w_cls` via `check_class()` before falling back to the abstract
/// `__class__` / `__bases__` walk.
unsafe fn p_recursive_isinstance_w(
    w_inst: PyObjectRef,
    w_cls: PyObjectRef,
) -> Result<bool, PyError> {
    if is_type_like_w(w_cls) {
        return p_recursive_isinstance_type_w(w_inst, w_cls);
    }
    check_class(
        w_cls,
        "isinstance() arg 2 must be a type, a tuple of types, or a union",
    )?;
    let w_abstractclass = match getattr_str(w_inst, "__class__") {
        Ok(cls) => cls,
        Err(e) if e.kind == PyErrorKind::AttributeError => return Ok(false),
        Err(e) => return Err(e),
    };
    p_abstract_issubclass_w(w_abstractclass, w_cls)
}

/// abstractinst.py:53-56 / 154-156:
/// `space.isinstance_w(obj, space.w_type)`.
///
/// This deliberately goes through pyre's object-space `isinstance_w`,
/// which consults the Python-level class (`w_class` / W_TypeObject MRO).
/// Do not replace it with `pyre_object::is_type_or_subtype()`: that helper
/// inspects the static Rust `PyType` tag and is not the RPython data path.
unsafe fn is_type_like_w(obj: PyObjectRef) -> bool {
    let w_type = crate::typedef::w_type();
    !w_type.is_null() && isinstance_w(obj, w_type)
}

/// `space.isinstance_w(w_obj, space.w_text)` — PyPy parity helper for
/// accepting `str` and any `str` subclass.  Used by `function.py:464`
/// `fset_func_name` and similar gateway-level type checks where the
/// upstream test is `isinstance_w(..., w_text)`, not exact-type
/// equality.  pyre's `pyre_object::is_str` only matches the exact
/// `STR_TYPE` tag and so rejects `class MyStr(str): pass` instances
/// — this helper fills in the MRO walk.
pub unsafe fn isinstance_str_w(obj: PyObjectRef) -> bool {
    if obj.is_null() {
        return false;
    }
    if pyre_object::is_str(obj) {
        return true;
    }
    if let Some(str_type) = crate::typedef::gettypefor(&pyre_object::STR_TYPE) {
        return isinstance_w(obj, str_type);
    }
    false
}

/// `space.isinstance_w(w_obj, space.w_int)` — PyPy parity helper for
/// `space.int_w` callers that should accept `int` and any `int`
/// subclass (e.g. `bool` and user-defined `class MyInt(int): pass`).
/// pyre's `pyre_object::is_int` matches `int` + `bool` only.
pub unsafe fn isinstance_int_w(obj: PyObjectRef) -> bool {
    if obj.is_null() {
        return false;
    }
    if pyre_object::is_int(obj) {
        return true;
    }
    if let Some(int_type) = crate::typedef::gettypefor(&pyre_object::INT_TYPE) {
        return isinstance_w(obj, int_type);
    }
    false
}

/// `space.isinstance_w(w_obj, space.w_bytes)` — accepts `bytes` and
/// any `bytes` subclass.
pub unsafe fn isinstance_bytes_w(obj: PyObjectRef) -> bool {
    if obj.is_null() {
        return false;
    }
    if pyre_object::is_bytes(obj) {
        return true;
    }
    if let Some(bytes_type) = crate::typedef::gettypefor(&pyre_object::BYTES_TYPE) {
        return isinstance_w(obj, bytes_type);
    }
    false
}

/// `space.charbuf_w` admits anything implementing the buffer protocol;
/// PyPy's `W_UnicodeDecodeError.descr_init` (`interp_exceptions.py:1043`)
/// uses it for `w_object` and then coerces to `bytes`.  In pyre the
/// concrete buffer producers are `bytes` and `bytearray` (incl.
/// subclasses); this helper accepts either.
pub unsafe fn isinstance_bytes_like_w(obj: PyObjectRef) -> bool {
    if obj.is_null() {
        return false;
    }
    if pyre_object::is_bytes_like(obj) {
        return true;
    }
    if let Some(bytes_type) = crate::typedef::gettypefor(&pyre_object::BYTES_TYPE) {
        if isinstance_w(obj, bytes_type) {
            return true;
        }
    }
    if let Some(bytearray_type) = crate::typedef::gettypefor(&pyre_object::BYTEARRAY_TYPE) {
        return isinstance_w(obj, bytearray_type);
    }
    false
}

/// `space.isinstance_w(w_obj, space.w_list)` — accepts `list` and any
/// `list` subclass.  pyre's `pyre_object::is_list` matches the exact
/// `LIST_TYPE` tag only.
pub unsafe fn isinstance_list_w(obj: PyObjectRef) -> bool {
    if obj.is_null() {
        return false;
    }
    if pyre_object::is_list(obj) {
        return true;
    }
    if let Some(list_type) = crate::typedef::gettypefor(&pyre_object::LIST_TYPE) {
        return isinstance_w(obj, list_type);
    }
    false
}

/// abstractinst.py:127-147 `p_abstract_issubclass_w`. Walks
/// `w_derived.__bases__` looking for an identity match with `w_cls`.
/// Recursion is bounded by avoiding the last entry of each `__bases__`
/// tuple — that one is followed by re-entering the loop.
pub(crate) fn p_abstract_issubclass_w(
    w_derived: PyObjectRef,
    w_cls: PyObjectRef,
) -> Result<bool, PyError> {
    let mut w_derived = w_derived;
    loop {
        if is_w(w_derived, w_cls) {
            return Ok(true);
        }
        let w_bases = match _get_bases(w_derived)? {
            Some(b) => b,
            None => return Ok(false),
        };
        let n = unsafe { w_tuple_len(w_bases) };
        if n == 0 {
            return Ok(false);
        }
        let last_index = n - 1;
        for i in 0..last_index {
            let base = match unsafe { w_tuple_getitem(w_bases, i as i64) } {
                Some(b) => b,
                None => return Ok(false),
            };
            if p_abstract_issubclass_w(base, w_cls)? {
                return Ok(true);
            }
        }
        w_derived = match unsafe { w_tuple_getitem(w_bases, last_index as i64) } {
            Some(b) => b,
            None => return Ok(false),
        };
    }
}

/// abstractinst.py:150-169 `p_recursive_issubclass_w`. The both-types
/// fast path is the common case; otherwise both arguments are validated
/// via `check_class()` before entering the abstract walk.
unsafe fn p_recursive_issubclass_w(
    w_derived: PyObjectRef,
    w_cls: PyObjectRef,
) -> Result<bool, PyError> {
    if is_type_like_w(w_cls) && is_type_like_w(w_derived) {
        return Ok(issubtype_w(w_derived, w_cls));
    }
    check_class(w_derived, "issubclass() arg 1 must be a class")?;
    check_class(
        w_cls,
        "issubclass() arg 2 must be a class or tuple of classes",
    )?;
    p_abstract_issubclass_w(w_derived, w_cls)
}

/// pypy/module/__builtin__/abstractinst.py:91-122
/// `abstract_isinstance_w(space, w_obj, w_klass_or_tuple, allow_override=True)`.
/// Handles tuple/union recursion, the `__instancecheck__` override
/// looked up via `space.lookup(w_klass_or_tuple, "__instancecheck__")`,
/// then the abstract `__class__`/`__bases__` walk.
pub fn isinstance(obj: PyObjectRef, classinfo: PyObjectRef) -> Result<bool, PyError> {
    let obj = unwrap_cell(obj);
    let classinfo = unwrap_cell(classinfo);
    unsafe {
        // abstractinst.py:104-106 — quick exact-type test.
        if let Some(t) = crate::typedef::r#type(obj) {
            if std::ptr::eq(t, classinfo) {
                return Ok(true);
            }
        }
        // abstractinst.py:108-114 — tuple recursion.
        if is_tuple(classinfo) {
            let n = w_tuple_len(classinfo);
            for i in 0..n {
                if let Some(c) = w_tuple_getitem(classinfo, i as i64) {
                    if isinstance(obj, c)? {
                        return Ok(true);
                    }
                }
            }
            return Ok(false);
        }
        // PEP 604 `X | Y` union recursion — lib_pypy/_pypy_generic_alias.py.
        if pyre_object::is_union(classinfo) {
            let union_args = pyre_object::w_union_get_args(classinfo);
            let n = w_tuple_len(union_args);
            for i in 0..n {
                if let Some(c) = w_tuple_getitem(union_args, i as i64) {
                    if isinstance(obj, c)? {
                        return Ok(true);
                    }
                }
            }
            return Ok(false);
        }
        // abstractinst.py:117-124 — `__instancecheck__` override
        // (`allow_override=True`). PyPy uses
        // `space.lookup(w_klass_or_tuple, "__instancecheck__")`, which
        // is a metaclass-MRO lookup (`lookup_in_type(type(cls), …)`),
        // not `getattr(cls, …)`. The distinction matters for the
        // weakproxy proxy_typedef_dict row: pyre's `getattr` runs the
        // `force()` fast path at entry and would dereference the proxy
        // before the typedef row ever gets a chance to fire. Going
        // through `lookup_in_type` on `type(classinfo)` keeps the
        // proxy's typedef wrapper installed via `proxy_typedef_dict`
        // visible. For real type objects pyre's `type` does not yet
        // install an `__instancecheck__` slot, so this falls through
        // to `p_recursive_isinstance_w` below — semantics-equivalent to
        // PyPy's `type.__instancecheck__` slot calling back into
        // `p_recursive_isinstance_type_w`.
        if let Some(cls_type) = crate::typedef::r#type(classinfo) {
            if let Some(check) = lookup_in_type(cls_type, "__instancecheck__") {
                // abstractinst.py:122 `space.get_and_call_function(w_check,
                // w_klass_or_tuple, w_obj)` — bind the descriptor to
                // `classinfo` before calling with `obj`.
                let result = get_and_call_function(check, classinfo, cls_type, &[obj])?;
                return Ok(is_true(result)?);
            }
        }
        p_recursive_isinstance_w(obj, classinfo)
    }
}

/// pypy/module/__builtin__/abstractinst.py:169-198
/// `abstract_issubclass_w(space, w_derived, w_klass_or_tuple, allow_override=True)`.
/// Tuple/union recursion, `__subclasscheck__` override looked up on
/// `type(classinfo)`, then the abstract `__bases__` walk.
pub fn issubclass(derived: PyObjectRef, classinfo: PyObjectRef) -> Result<bool, PyError> {
    let derived = unwrap_cell(derived);
    let classinfo = unwrap_cell(classinfo);
    unsafe {
        // abstractinst.py:181-187 — tuple recursion.
        if is_tuple(classinfo) {
            let n = w_tuple_len(classinfo);
            for i in 0..n {
                if let Some(c) = w_tuple_getitem(classinfo, i as i64) {
                    if issubclass(derived, c)? {
                        return Ok(true);
                    }
                }
            }
            return Ok(false);
        }
        if pyre_object::is_union(classinfo) {
            let union_args = pyre_object::w_union_get_args(classinfo);
            let n = w_tuple_len(union_args);
            for i in 0..n {
                if let Some(c) = w_tuple_getitem(union_args, i as i64) {
                    if issubclass(derived, c)? {
                        return Ok(true);
                    }
                }
            }
            return Ok(false);
        }
        // abstractinst.py:190-196 — `__subclasscheck__` override.
        // Same `lookup_in_type(type(classinfo), …)` rationale as
        // `isinstance` above.
        if let Some(cls_type) = crate::typedef::r#type(classinfo) {
            if let Some(check) = lookup_in_type(cls_type, "__subclasscheck__") {
                // abstractinst.py:195 `space.get_and_call_function(w_check,
                // w_klass_or_tuple, w_derived)` — bind the descriptor to
                // `classinfo` before calling with `derived`.
                let result = get_and_call_function(check, classinfo, cls_type, &[derived])?;
                return Ok(is_true(result)?);
            }
        }
        p_recursive_issubclass_w(derived, classinfo)
    }
}

/// Test if an object is truthy (for branch conditions).
///
/// Python truthiness rules:
/// - None → false
/// - bool → its value
/// - int → nonzero

/// `baseobjspace.py:1346-1353 isabstractmethod_w`:
///
/// ```python
/// def isabstractmethod_w(self, w_obj):
///     try:
///         w_result = self.getattr(w_obj, self.newtext("__isabstractmethod__"))
///     except OperationError as e:
///         if e.match(self, self.w_AttributeError):
///             return False
///         raise
///     return self.is_true(w_result)
/// ```
///
/// Catches the AttributeError arm of the upstream try/except and
/// reraises any other PyError so the caller (typedef descr_isabstract
/// for staticmethod / classmethod) can propagate it.
pub fn isabstractmethod_w(obj: PyObjectRef) -> Result<bool, crate::PyError> {
    match getattr_str(obj, "__isabstractmethod__") {
        Ok(w_result) => Ok(is_true(w_result)?),
        Err(e) if matches!(e.kind, crate::PyErrorKind::AttributeError) => Ok(false),
        Err(e) => Err(e),
    }
}

/// Resolve a special method `name` on `obj`'s type, returning it (with the
/// owning `w_type` for `get_and_call_function` binding) only when the object
/// is a builtin subclass that *overrides* it — i.e. the MRO resolution
/// differs from the implementation the builtin layout type itself registers.
///
/// The by-layout fast paths (`space.is_true` / `len` / `getitem` / `compare`
/// / …) are the inherited builtin behaviour; a subclass that does not
/// override the method must keep using them.  Crucially, the builtin
/// typedef slots delegate back into these very `space` entry points
/// (`list.__getitem__` → `space.getitem`, `int.__eq__` → `compare`), so
/// dispatching the *inherited* slot would recurse forever — this helper
/// returns `None` for the inherited case so the caller takes the fast path,
/// and only returns `Some` for a genuine user override (whose body is user
/// code, not a re-entry).
///
/// Returns `None` for exact builtins, for absent methods, and for inherited
/// builtin slots.
///
/// # Safety
/// `obj` must be a valid `PyObjectRef`.
pub(crate) unsafe fn subclass_special_override(
    obj: PyObjectRef,
    name: &str,
) -> Option<(PyObjectRef, PyObjectRef)> {
    if pyre_object::is_exact_builtin_instance(obj) {
        return None;
    }
    let w_type = crate::typedef::r#type(obj)?;
    let method = lookup_in_type_where(w_type, name)?;
    // The builtin layout type for `obj` — the canonical type object for its
    // `ob_type`.  When the MRO resolution matches that type's own slot the
    // method is inherited, not overridden.
    let base = pyre_object::get_instantiate(&*(*obj).ob_type);
    if !base.is_null() {
        if let Some(inherited) = lookup_in_type_where(base, name) {
            if std::ptr::eq(inherited, method) {
                return None;
            }
        }
    }
    Some((method, w_type))
}

/// descroperation.py:265-285 `is_true`.
///
/// ```python
/// def is_true(space, w_obj):
///     w_descr = space.lookup(w_obj, "__bool__")
///     if w_descr is None:
///         w_descr = space.lookup(w_obj, "__len__")
///         if w_descr is None:
///             return True
///         w_res = space.get_and_call_function(w_descr, w_obj)
///         return space._check_len_result(space.index(w_res)) != 0
///     w_res = space.get_and_call_function(w_descr, w_obj)
///     if space.is_w(w_res, space.w_False): return False
///     if space.is_w(w_res, space.w_True):  return True
///     raise oefmt(space.w_TypeError,
///                 "__bool__ should return bool, returned %T", w_obj)
/// ```
///
/// A builtin subclass overriding `__bool__` / `__len__` is detected by
/// [`subclass_special_override`] and dispatched first. The leading built-in
/// fast paths then short-circuit the `lookup` + call machinery for exact
/// builtins (and non-overriding subclasses, whose inherited truthiness is
/// the by-layout result). Only objects matching no fast path reach the
/// generic tail, which consults `__bool__` then `__len__`, where the call
/// exceptions — and the non-bool-`__bool__` TypeError — propagate.
pub fn is_true(obj: PyObjectRef) -> Result<bool, PyError> {
    let obj = unwrap_cell(obj);
    // descroperation.py:265 — `__bool__` (anywhere in the MRO) is consulted
    // before `__len__`.  An exact builtin's `__bool__` / `__len__` are the
    // inherited builtin slots, so its truthiness is computed by layout in
    // `is_true_slot`; any other object (builtin subclass or user instance)
    // takes the `lookup` path, where an inherited builtin `__bool__` is still
    // found and wins over an overridden `__len__`.
    if unsafe { is_exact_builtin_instance(obj) } {
        return is_true_slot(obj);
    }
    is_true_lookup(obj)
}

/// descroperation.py:265-285 — the `lookup` path of `is_true`: `__bool__`
/// first, then `__len__`, each bound via `get_and_call_function` so
/// descriptors are honored.  Used for builtin subclasses and user instances
/// (and by `is_true_slot` for an exact builtin that matched no by-layout fast
/// path).  An inherited builtin `__bool__` is found here and takes priority
/// over an overridden `__len__`.
fn is_true_lookup(obj: PyObjectRef) -> Result<bool, PyError> {
    if let Some(w_type) = crate::typedef::r#type(obj) {
        if let Some(w_descr) = unsafe { lookup_in_type(w_type, "__bool__") } {
            let w_res = unsafe { get_and_call_function(w_descr, obj, w_type, &[]) }?;
            // The only instances of bool are `w_False` / `w_True`, so a
            // non-bool result is a TypeError reporting the receiver's type
            // (upstream's `%T` on `w_obj`).
            if unsafe { is_bool(w_res) } {
                return Ok(unsafe { w_bool_get_value(w_res) });
            }
            return Err(PyError::type_error(format!(
                "__bool__ should return bool, returned {}",
                object_functionstr_type_name(obj),
            )));
        }
        if let Some(w_descr) = unsafe { lookup_in_type(w_type, "__len__") } {
            let w_res = unsafe { get_and_call_function(w_descr, obj, w_type, &[]) }?;
            let w_index = space_index(w_res)?;
            return Ok(_check_len_result(w_index)? != 0);
        }
    }
    Ok(true)
}

/// Direct truthiness body for `is_true`: the by-layout fast paths for exact
/// builtins, falling back to `is_true_lookup`.  The builtin `int` / `float`
/// `__bool__` base slots bind here so that the lookup path can invoke them
/// (for non-overriding subclasses) without recursing through `is_true`.
pub(crate) fn is_true_slot(obj: PyObjectRef) -> Result<bool, PyError> {
    let obj = unwrap_cell(obj);
    unsafe {
        if is_bool(obj) {
            return Ok(w_bool_get_value(obj));
        }
        if is_int(obj) {
            return Ok(w_int_get_value(obj) != 0);
        }
        if is_long(obj) {
            return Ok(w_long_get_value(obj).clone() != BigInt::from(0));
        }
        if is_float(obj) {
            return Ok(w_float_get_value(obj) != 0.0);
        }
        if pyre_object::is_complex(obj) {
            return Ok(pyre_object::w_complex_get_real(obj) != 0.0
                || pyre_object::w_complex_get_imag(obj) != 0.0);
        }
        if is_str(obj) {
            return Ok(w_str_len(obj) != 0);
        }
        if pyre_object::is_bytes(obj) {
            return Ok(pyre_object::w_bytes_len(obj) != 0);
        }
        if pyre_object::is_bytearray(obj) {
            return Ok(pyre_object::w_bytearray_len(obj) != 0);
        }
        if is_list(obj) {
            return Ok(w_list_len(obj) > 0);
        }
        if is_tuple(obj) {
            return Ok(w_tuple_len(obj) > 0);
        }
        if is_dict(obj) {
            return Ok(w_dict_len(obj) > 0);
        }
        if pyre_object::is_set_or_frozenset(obj) {
            return Ok(pyre_object::w_set_len(obj) > 0);
        }
        if pyre_object::is_w_range(obj) {
            return Ok(pyre_object::w_range_bool(obj));
        }
        if is_none(obj) {
            return Ok(false);
        }
    }
    // No by-layout fast path matched (an exact builtin without a layout
    // predicate): consult `__bool__` / `__len__` via the lookup path.
    is_true_lookup(obj)
}

// ── Subscript operations ─────────────────────────────────────────────

/// Normalize a slice to (start, stop, step) for a sequence of `length`.
///
/// PyPy: sliceobject.py descr_indices, mirroring CPython
/// `PySlice_Unpack` + `PySlice_AdjustIndices`. Handles negative `step`
/// (which CPython adjusts the start/stop bounds for separately from
/// positive `step`). Each bound is evaluated through `__index__`
/// (`_eval_slice_index`), so a `float` or other non-integer bound raises
/// `TypeError` rather than being read as a raw integer field.
pub(crate) unsafe fn normalize_slice(
    index: PyObjectRef,
    length: i64,
) -> Result<(i64, i64, i64), PyError> {
    let start_obj = w_slice_get_start(index);
    let stop_obj = w_slice_get_stop(index);
    let step_obj = w_slice_get_step(index);
    let step = if is_none(step_obj) {
        1
    } else {
        crate::sliceobject::eval_slice_index(step_obj)?
    };
    if step == 0 {
        return Err(PyError::new(
            PyErrorKind::ValueError,
            "slice step cannot be zero",
        ));
    }
    let (lower, upper) = if step > 0 {
        (0, length)
    } else {
        (-1, length - 1)
    };
    let start = if is_none(start_obj) {
        if step > 0 { 0 } else { length - 1 }
    } else {
        let v = crate::sliceobject::eval_slice_index(start_obj)?;
        let v = if v < 0 { v + length } else { v };
        v.max(lower).min(upper)
    };
    let stop = if is_none(stop_obj) {
        if step > 0 { length } else { -1 }
    } else {
        let v = crate::sliceobject::eval_slice_index(stop_obj)?;
        let v = if v < 0 { v + length } else { v };
        v.max(lower).min(upper)
    };
    Ok((start, stop, step))
}

/// `descroperation.py:169 get_and_call_function`.
///
/// ```python
/// def get_and_call_function(space, w_descr, w_obj, *args_w):
///     typ = type(w_descr)
///     if typ is Function or typ is FunctionWithFixedCode:
///         return w_descr.funccall(w_obj, *args_w)
///     else:
///         args = Arguments(space, list(args_w))
///         w_impl = space.get(w_descr, w_obj)
///         return space.call_args(w_impl, args)
/// ```
///
/// The `Function`/`FunctionWithFixedCode` fast path (both `FUNCTION_TYPE`
/// here, exact match) calls `funccall(w_obj, *args_w)` — `w_obj` leads the
/// positionals.  `BuiltinFunction` (`BUILTIN_FUNCTION_TYPE`) is excluded
/// because it binds differently.  Every other descriptor is bound through
/// `get` (`space.get`) first, then called with `args_w` alone, so
/// `@staticmethod` / `@classmethod` / custom-descriptor dunders receive the
/// arguments PyPy gives them.
pub(crate) unsafe fn get_and_call_function(
    w_descr: PyObjectRef,
    w_obj: PyObjectRef,
    w_type: PyObjectRef,
    args_w: &[PyObjectRef],
) -> PyResult {
    if !w_descr.is_null()
        && std::ptr::eq(
            unsafe { (*w_descr).ob_type },
            &crate::FUNCTION_TYPE as *const _,
        )
    {
        let mut full = Vec::with_capacity(args_w.len() + 1);
        full.push(w_obj);
        full.extend_from_slice(args_w);
        return crate::call::call_function_impl_result(w_descr, &full);
    }
    let w_impl = unsafe { get(w_descr, w_obj, w_type) }?.unwrap_or(w_descr);
    crate::call::call_function_impl_result(w_impl, args_w)
}

/// `isinstance(w_obj, Coroutine) or gen_is_coroutine(w_obj)` from
/// `generator.py:569`, collapsed onto pyre's single generator object: an
/// `async def` coroutine and a `@types.coroutine`-marked generator both carry
/// their marker on the suspended frame's code (`CO_COROUTINE` /
/// `CO_ITERABLE_COROUTINE`), so the distinct PyPy `Coroutine` class becomes a
/// code-flag test here.
fn is_coroutine(w_obj: PyObjectRef) -> bool {
    unsafe {
        if !pyre_object::generator::is_generator(w_obj) {
            return false;
        }
        let frame_ptr =
            pyre_object::generator::w_generator_get_frame(w_obj) as *const crate::pyframe::PyFrame;
        // An exhausted generator has a null frame and so no readable flags; an
        // awaited coroutine is never exhausted at this point.
        if frame_ptr.is_null() {
            return false;
        }
        (*frame_ptr)
            .code()
            .flags
            .intersects(crate::CodeFlags::COROUTINE | crate::CodeFlags::ITERABLE_COROUTINE)
    }
}

/// True only for a native `async def` coroutine (`CO_COROUTINE`), excluding
/// `@types.coroutine`-wrapped generators (`CO_ITERABLE_COROUTINE`).  Mirrors
/// `PyCoro_CheckExact`, which gates the GET_AWAITABLE already-awaited guard.
fn is_native_coroutine(w_obj: PyObjectRef) -> bool {
    unsafe {
        if !pyre_object::generator::is_generator(w_obj) {
            return false;
        }
        let frame_ptr =
            pyre_object::generator::w_generator_get_frame(w_obj) as *const crate::pyframe::PyFrame;
        if frame_ptr.is_null() {
            return false;
        }
        (*frame_ptr)
            .code()
            .flags
            .contains(crate::CodeFlags::COROUTINE)
    }
}

/// `generator.py:563 get_awaitable_iter` — return the iterator implementing the
/// awaitable protocol for `w_obj`:
///   - `w_obj` itself when it is a coroutine (or `@types.coroutine` generator);
///   - otherwise `w_obj.__await__()`, which must be an iterator.
///
/// `context`: 0 = plain `await`, 1 = `__aenter__`, 2 = `__aexit__` — only the
/// missing-`__await__` error message differs.
pub fn get_awaitable_iter(w_obj: PyObjectRef, context: u32) -> PyResult {
    if is_coroutine(w_obj) {
        // GET_AWAITABLE: re-awaiting a native coroutine that is already
        // suspended at an `await` raises (`_PyGen_yf(coro) != NULL`). A native
        // coroutine only ever suspends at an `await`, so "started, not
        // exhausted, not currently running" is exactly that delegating state.
        if is_native_coroutine(w_obj)
            && unsafe {
                pyre_object::generator::w_generator_is_started(w_obj)
                    && !pyre_object::generator::w_generator_is_exhausted(w_obj)
                    && !pyre_object::generator::w_generator_is_running(w_obj)
            }
        {
            return Err(PyError::runtime_error("coroutine is being awaited already"));
        }
        return Ok(w_obj);
    }
    let w_await = crate::typedef::r#type(w_obj)
        .and_then(|w_type| unsafe { lookup_in_type(w_type, "__await__") });
    let Some(w_await) = w_await else {
        let msg = match context {
            1 => format!(
                "'async with' received an object from __aenter__ that does not \
                 implement __await__: {}",
                object_functionstr_type_name(w_obj),
            ),
            2 => format!(
                "'async with' received an object from __aexit__ that does not \
                 implement __await__: {}",
                object_functionstr_type_name(w_obj),
            ),
            _ => format!(
                "object {} can't be used in 'await' expression",
                object_functionstr_type_name(w_obj),
            ),
        };
        return Err(PyError::type_error(msg));
    };
    let w_type = crate::typedef::r#type(w_obj).unwrap_or(w_obj);
    let w_res = unsafe { get_and_call_function(w_await, w_obj, w_type, &[]) }?;
    if is_coroutine(w_res) {
        return Err(PyError::type_error(
            "__await__() returned a coroutine (it must return an iterator \
             instead, see PEP 492)",
        ));
    }
    // `space.lookup(w_res, "__next__")` — w_res must be an iterator.  pyre's
    // generator object (the usual `__await__` return) carries `__next__` at the
    // instance level rather than on its type, so it is accepted directly; other
    // iterators expose `__next__` on their type.
    let has_next = unsafe { pyre_object::generator::is_generator(w_res) }
        || crate::typedef::r#type(w_res)
            .is_some_and(|w_type| unsafe { lookup_in_type(w_type, "__next__") }.is_some());
    if !has_next {
        return Err(PyError::type_error(format!(
            "__await__() returned non-iterator of type '{}'",
            object_functionstr_type_name(w_res),
        )));
    }
    Ok(w_res)
}

/// _set_names (typeobject.py:1006) — invoke `__set_name__(owner, name)` for one
/// class-body entry.  `__set_name__` is found by a type-only lookup on
/// `type(w_value)` (`space.lookup`, NOT the full attribute protocol, so a
/// user `__getattribute__`/`__getattr__` does not run), then bound and called.
/// A missing `__set_name__` is a no-op.  When the call raises, the original
/// exception is re-raised with an `"Error calling __set_name__ ..."` note
/// attached (PEP 678), mirroring `_PyErr_FormatNote`.
pub(crate) unsafe fn set_name(
    w_owner: PyObjectRef,
    w_name: PyObjectRef,
    w_value: PyObjectRef,
) -> Result<(), PyError> {
    let w_valtype = match crate::typedef::r#type(w_value) {
        Some(t) => t,
        None => return Ok(()),
    };
    let set_name_meth = match unsafe { lookup_in_type_where(w_valtype, "__set_name__") } {
        Some(m) => m,
        None => return Ok(()),
    };
    // `space.get_and_call_function(w_meth, w_value, w_type, key)` — `w_value`
    // is the descriptor instance (bound as the receiver), and the call args
    // are `(owner, name)`: `__set_name__(self, owner, name)`.
    match unsafe { get_and_call_function(set_name_meth, w_value, w_valtype, &[w_owner, w_name]) } {
        Ok(_) => Ok(()),
        Err(e) => {
            if !e.exc_object.is_null() {
                let name_repr = if unsafe { is_str(w_name) } {
                    crate::display::format_wtf8_repr(unsafe { w_str_get_wtf8(w_name) })
                } else {
                    String::new()
                };
                let val_type_name = unsafe { pyre_object::w_type_get_name(w_valtype) }.to_string();
                let owner_name = unsafe { pyre_object::w_type_get_name(w_owner) }.to_string();
                let note = w_str_new(&format!(
                    "Error calling __set_name__ on '{val_type_name}' instance {name_repr} in '{owner_name}'"
                ));
                if let Ok(add) = getattr_str(e.exc_object, "add_note") {
                    // add_note is best-effort: a failure to attach the note must
                    // not mask the original __set_name__ exception `e`, which is
                    // re-raised below.
                    if let Err(_e) = crate::call::call_function_impl_result(add, &[note]) {}
                }
            }
            Err(e)
        }
    }
}

#[majit_macros::dont_look_inside]
pub(crate) fn dict_missing_or_key_error(obj: PyObjectRef, index: PyObjectRef) -> PyResult {
    if let Some(w_type_obj) = crate::typedef::r#type(obj) {
        let dict_type = crate::typedef::gettypeobject(&pyre_object::DICT_TYPE);
        if dict_type.is_null() == false && std::ptr::eq(w_type_obj, dict_type) == false {
            if let Some(w_missing) = unsafe { lookup_in_type(w_type_obj, "__missing__") } {
                // dictmultiobject.py:166 space.get_and_call_function(
                //     w_missing, self, w_key)
                return unsafe { get_and_call_function(w_missing, obj, w_type_obj, &[index]) };
            }
        }
    }
    Err(PyError::key_error_with_key(index))
}

/// Get item by index: `obj[index]`.
///
/// Dispatches based on the type of `obj`.

pub fn getitem(obj: PyObjectRef, index: PyObjectRef) -> PyResult {
    let obj = unwrap_cell(obj);
    let index = unwrap_cell(index);
    // `pypy/objspace/std/dictproxyobject.py:35 descr_getitem` →
    // `space.getitem(self.w_mapping, w_key)` — forward through the
    // proxy to its wrapped mapping.  The unwrap happens at the
    // entrance so all downstream dict arms (and dict-subclass
    // overrides via the wrapped W_DictObject) see the underlying
    // mapping unchanged.
    unsafe {
        let obj = if pyre_object::is_dict_proxy(obj) {
            pyre_object::w_dict_proxy_get_mapping(obj)
        } else {
            obj
        };
        // A builtin sequence subclass (`is_list`/`is_tuple`/… stays true on
        // the retagged layout) overriding `__getitem__` must dispatch the
        // override; the by-layout slot below gives the inherited builtin
        // subscript for exact instances and non-overriding subclasses.
        if is_list(obj)
            || is_tuple(obj)
            || is_str(obj)
            || pyre_object::bytesobject::is_bytes_like(obj)
            || pyre_object::is_w_range(obj)
        {
            if let Some((method, w_type)) = subclass_special_override(obj, "__getitem__") {
                return get_and_call_function(method, obj, w_type, &[index]);
            }
        }
        getitem_slot(obj, index)
    }
}

/// The builtin `__getitem__` slot body: subscript dispatch by concrete
/// layout.  Reached from the operator [`getitem`] for exact instances and
/// non-overriding subclasses, and bound directly as the `list`/`str`/`tuple`
/// `__getitem__` slot so a subclass override's `super().__getitem__` resolves
/// to the inherited builtin subscript instead of re-entering override
/// dispatch (which would recurse).
pub(crate) fn getitem_slot(obj: PyObjectRef, index: PyObjectRef) -> PyResult {
    let obj = unwrap_cell(obj);
    let index = unwrap_cell(index);
    unsafe {
        if is_list(obj) {
            return getitem_list(obj, index);
        }
        if is_tuple(obj) {
            return getitem_tuple(obj, index);
        }
        if is_dict(obj) {
            // `pypy/objspace/std/dictmultiobject.py:137-141 W_DictMultiObject
            // .descr_getitem` → `space.getitem(self, w_key)` → strategy
            return match pyre_object::dictmultiobject::w_dict_lookup_checked(obj, index) {
                Ok(Some(val)) => Ok(val),
                Ok(None) => {
                    // dictmultiobject.py:166-170 — dict subclass
                    // __missing__ dispatch before KeyError
                    dict_missing_or_key_error(obj, index)
                }
                Err(_) => Err(take_pending_hash_error()),
            };
        }
        if is_str(obj) {
            return getitem_str(obj, index);
        }
        if pyre_object::bytesobject::is_bytes_like(obj) {
            return getitem_bytes_like(obj, index);
        }
        if is_type(obj) {
            return getitem_type(obj, index);
        }
        if is_instance(obj) {
            return getitem_instance(obj, index);
        }
        if pyre_object::is_w_range(obj) {
            return getitem_range(obj, index);
        }
        if is_range_iter(obj) {
            return getitem_range_iter(obj, index);
        }
        // descroperation.py:356-381 DescrOperation.getitem — any object
        // whose type defines `__getitem__` on its MRO is subscriptable
        // (the arms above are fast paths for builtin sequence/mapping
        // types).  Covers W_Root types like `re.Match` whose typedef
        // registers `__getitem__`.
        if let Some(w_type) = crate::typedef::r#type(obj) {
            if let Some(method) = lookup_in_type_where(w_type, "__getitem__") {
                return get_and_call_function(method, obj, w_type, &[index]);
            }
        }
        Err(PyError::type_error(format!(
            "'{}' object is not subscriptable",
            (*(*obj).ob_type).name,
        )))
    }
}

/// `pypy/interpreter/baseobjspace.py:1574 getindex_w` — the `TypeError`
/// raised when a sequence subscript is neither an integer nor a slice:
/// `"<descr> indices must be integers or slices, not <type>"`.
fn index_type_error(descr: &str, index: PyObjectRef) -> PyError {
    let tp = unsafe {
        if index.is_null() {
            "NULL"
        } else {
            (*(*index).ob_type).name
        }
    };
    PyError::type_error(format!(
        "{descr} indices must be integers or slices, not {tp}"
    ))
}

#[inline(never)]
unsafe fn getitem_list(obj: PyObjectRef, index: PyObjectRef) -> PyResult {
    if is_slice(index) {
        let len = w_list_len(obj) as i64;
        let (start, stop, step) = normalize_slice(index, len)?;
        let mut items = Vec::new();
        let mut i = start;
        while (step > 0 && i < stop) || (step < 0 && i > stop) {
            if let Some(v) = w_list_getitem(obj, i) {
                items.push(v);
            }
            i += step;
        }
        return Ok(w_list_new(items));
    }
    if !is_int(index) {
        return Err(index_type_error("list", index));
    }
    let idx = w_int_get_value(index);
    match w_list_getitem(obj, idx) {
        Some(val) => Ok(val),
        None => Err(PyError::new(
            PyErrorKind::IndexError,
            "list index out of range",
        )),
    }
}

#[inline(never)]
unsafe fn getitem_tuple(obj: PyObjectRef, index: PyObjectRef) -> PyResult {
    if is_slice(index) {
        // tupleobject.py descr_getslice → slice.indices.
        let len = w_tuple_len(obj) as i64;
        let (start, stop, step) = normalize_slice(index, len)?;
        let mut items = Vec::new();
        let mut i = start;
        while (step > 0 && i < stop) || (step < 0 && i > stop) {
            if let Some(v) = w_tuple_getitem(obj, i) {
                items.push(v);
            }
            i += step;
        }
        return Ok(w_tuple_new(items));
    }
    if !is_int(index) {
        return Err(index_type_error("tuple", index));
    }
    let idx = w_int_get_value(index);
    match w_tuple_getitem(obj, idx) {
        Some(val) => Ok(val),
        None => Err(PyError::new(
            PyErrorKind::IndexError,
            "tuple index out of range",
        )),
    }
}

#[inline(never)]
unsafe fn getitem_str(obj: PyObjectRef, index: PyObjectRef) -> PyResult {
    // Index code points through the surrogate-aware WTF-8 view so a
    // surrogateescape / surrogatepass-decoded string can be sliced and
    // indexed without going through `w_str_get_value`.
    let cps: Vec<CodePoint> = w_str_get_wtf8(obj).code_points().collect();
    if is_slice(index) {
        // `pypy/objspace/std/unicodeobject.py W_UnicodeObject._getitem_slice`
        // → `slice.indices(len)` (`pypy/objspace/std/sliceobject.py`).
        // Reuse the shared `normalize_slice` helper so negative-step
        // defaults (`s[::-1]`, `s[5::-1]`) match list/tuple semantics.
        let len = cps.len() as i64;
        let (start, stop, step) = normalize_slice(index, len)?;
        let mut result = Wtf8Buf::new();
        let mut i = start;
        while (step > 0 && i < stop) || (step < 0 && i > stop) {
            if i >= 0 && (i as usize) < cps.len() {
                result.push(cps[i as usize]);
            }
            i += step;
        }
        return Ok(w_str_from_wtf8(result));
    }
    if is_int(index) {
        let idx = w_int_get_value(index);
        let actual_idx = if idx < 0 { cps.len() as i64 + idx } else { idx } as usize;
        if actual_idx < cps.len() {
            let mut one = Wtf8Buf::new();
            one.push(cps[actual_idx]);
            return Ok(w_str_from_wtf8(one));
        }
        return Err(PyError::new(
            PyErrorKind::IndexError,
            "string index out of range",
        ));
    }
    Err(index_type_error("string", index))
}

#[inline(never)]
unsafe fn getitem_bytes_like(obj: PyObjectRef, index: PyObjectRef) -> PyResult {
    let is_bytes = pyre_object::bytesobject::is_bytes(obj);
    if is_int(index) {
        let idx = w_int_get_value(index);
        let len = pyre_object::bytesobject::bytes_like_len(obj) as i64;
        let actual = if idx < 0 { len + idx } else { idx };
        if actual >= 0 && actual < len {
            return Ok(w_int_new(
                pyre_object::bytesobject::bytes_like_getitem(obj, actual as usize) as i64,
            ));
        }
        let name = if is_bytes { "bytes" } else { "bytearray" };
        return Err(PyError::new(
            PyErrorKind::IndexError,
            format!("{name} index out of range"),
        ));
    }
    if is_slice(index) {
        let len = pyre_object::bytesobject::bytes_like_len(obj) as i64;
        let (start, stop, step) = normalize_slice(index, len)?;
        let mut result = Vec::new();
        let mut i = start;
        if step > 0 {
            while i < stop {
                result.push(pyre_object::bytesobject::bytes_like_getitem(
                    obj, i as usize,
                ));
                i += step;
            }
        } else {
            while i > stop {
                result.push(pyre_object::bytesobject::bytes_like_getitem(
                    obj, i as usize,
                ));
                i += step;
            }
        }
        return Ok(if is_bytes {
            pyre_object::bytesobject::w_bytes_from_bytes(&result)
        } else {
            pyre_object::bytearrayobject::w_bytearray_from_bytes(&result)
        });
    }
    let descr = if is_bytes { "byte" } else { "bytearray" };
    Err(index_type_error(descr, index))
}

#[inline(never)]
unsafe fn getitem_type(obj: PyObjectRef, index: PyObjectRef) -> PyResult {
    // A metaclass `__getitem__` (resolved on `type(cls)`'s MRO, e.g.
    // `EnumMeta.__getitem__`) applies before the PEP 560
    // `__class_getitem__` fallback: `Color['RED']` is `type(Color)
    // .__getitem__(Color, 'RED')`.  `type` itself defines no `__getitem__`,
    // so an ordinary class still takes the `__class_getitem__` path below.
    if let Some(w_meta) = crate::typedef::r#type(obj) {
        if let Some(method) = lookup_in_type_where(w_meta, "__getitem__") {
            return get_and_call_function(method, obj, w_meta, &[index]);
        }
    }
    // descroperation.py:362 — `type[X]` (the operand is exactly `type`) builds
    // a GenericAlias even though `type` defines no `__class_getitem__`.
    if std::ptr::eq(obj, crate::typedef::w_type()) {
        return crate::_pypy_generic_alias::generic_alias_class_getitem(&[obj, index]);
    }
    // Python 3.9+ generic subscript: cls[X] → cls.__class_getitem__(X)
    // (`descroperation.py:366` getattr lookup).
    if let Some(method) = lookup_in_type_where(obj, "__class_getitem__") {
        return get_and_call_function(method, obj, obj, &[index]);
    }
    // abstract.py descr_getitem — a class that defines neither a metaclass
    // __getitem__ nor __class_getitem__ is not subscriptable.
    Err(PyError::type_error(format!(
        "type '{}' is not subscriptable",
        w_type_get_name(obj),
    )))
}

#[inline(never)]
unsafe fn getitem_instance(obj: PyObjectRef, index: PyObjectRef) -> PyResult {
    // descroperation.py __getitem__
    let w_type = w_instance_get_type(obj);
    if let Some(method) = lookup_in_type_where(w_type, "__getitem__") {
        return get_and_call_function(method, obj, w_type, &[index]);
    }
    Err(PyError::type_error(format!(
        "'{}' object is not subscriptable",
        w_type_get_name(w_instance_get_type(obj)),
    )))
}

#[inline(never)]
/// `functional.py W_Range.descr_getitem` — integer index returns
/// the member `start + i*step` (negative folded, bounds-checked); a slice
/// returns a NEW `range` object, not a list.  Indices and members are
/// kept in bignum so a range past a machine word is still subscriptable.
unsafe fn getitem_range(obj: PyObjectRef, index: PyObjectRef) -> PyResult {
    if is_slice(index) {
        return range_compute_slice(obj, index);
    }
    // `_compute_item` — `space.index(w_index)` then bounds-check.
    let w_index = space_index(index)?;
    let idx = pyre_object::range_obj_to_bigint(w_index);
    match pyre_object::w_range_compute_item(obj, &idx) {
        Some(v) => Ok(v),
        None => Err(PyError::new(
            PyErrorKind::IndexError,
            "range object index out of range",
        )),
    }
}

/// `functional.py compute_slice_indices3` — resolve a slice's
/// (start, stop, step) against a wrapped `length`, in bignum.
unsafe fn compute_slice_indices3_big(
    slice: PyObjectRef,
    length: &BigInt,
) -> Result<(BigInt, BigInt, BigInt), PyError> {
    use num_traits::{One, Zero};
    let zero = BigInt::zero();
    let one = BigInt::one();
    let w_step = w_slice_get_step(slice);
    let step = if is_none(w_step) {
        one.clone()
    } else {
        let s = pyre_object::range_obj_to_bigint(space_index(w_step)?);
        if s.is_zero() {
            return Err(PyError::new(
                PyErrorKind::ValueError,
                "slice step cannot be zero",
            ));
        }
        s
    };
    let negative_step = step < zero;
    let w_start = w_slice_get_start(slice);
    let start = if is_none(w_start) {
        if negative_step {
            length - &one
        } else {
            zero.clone()
        }
    } else {
        let st = pyre_object::range_obj_to_bigint(space_index(w_start)?);
        if st < zero {
            let st = st + length;
            if st < zero {
                if negative_step {
                    -one.clone()
                } else {
                    zero.clone()
                }
            } else {
                st
            }
        } else if st >= *length {
            if negative_step {
                length - &one
            } else {
                length.clone()
            }
        } else {
            st
        }
    };
    let w_stop = w_slice_get_stop(slice);
    let stop = if is_none(w_stop) {
        if negative_step {
            -one.clone()
        } else {
            length.clone()
        }
    } else {
        let sp = pyre_object::range_obj_to_bigint(space_index(w_stop)?);
        if sp < zero {
            let sp = sp + length;
            if sp < zero {
                if negative_step {
                    -one.clone()
                } else {
                    zero.clone()
                }
            } else {
                sp
            }
        } else if sp >= *length {
            if negative_step {
                length - &one
            } else {
                length.clone()
            }
        } else {
            sp
        }
    };
    Ok((start, stop, step))
}

/// `functional.py W_Range._compute_slice` — build the NEW `range`
/// a slice of `obj` denotes.
unsafe fn range_compute_slice(obj: PyObjectRef, slice: PyObjectRef) -> PyResult {
    use num_traits::Zero;
    let len_b = pyre_object::range_obj_to_bigint(pyre_object::w_range_length(obj));
    let (sl_start, sl_stop, sl_step) = compute_slice_indices3_big(slice, &len_b)?;
    let (rstart, _rstop, rstep) = pyre_object::w_range_fields(obj);
    let rstart_b = pyre_object::range_obj_to_bigint(rstart);
    let rstep_b = pyre_object::range_obj_to_bigint(rstep);
    let stop_is_zero = sl_stop.is_zero();
    let substart = &rstart_b + &sl_start * &rstep_b;
    let substep = &rstep_b * &sl_step;
    let _roots = pyre_object::gc_roots::push_roots();
    let w_substart = pyre_object::range_bigint_to_obj(substart);
    pyre_object::gc_roots::pin_root(w_substart);
    let w_substep = pyre_object::range_bigint_to_obj(substep);
    pyre_object::gc_roots::pin_root(w_substep);
    let w_substop = if stop_is_zero {
        w_substart
    } else {
        let substop = &rstart_b + &sl_stop * &rstep_b;
        let o = pyre_object::range_bigint_to_obj(substop);
        pyre_object::gc_roots::pin_root(o);
        o
    };
    Ok(pyre_object::w_range_new(w_substart, w_substop, w_substep))
}

/// `range.count(value)` — `functional.py W_Range.descr_count`.
fn range_count_method(args: &[PyObjectRef]) -> PyResult {
    let obj = args[0];
    let needle = args.get(1).copied().unwrap_or(PY_NULL);
    Ok(w_int_new(if contains(obj, needle)? { 1 } else { 0 }))
}

/// `range.index(value)` — `functional.py W_Range.descr_index`.
fn range_index_method(args: &[PyObjectRef]) -> PyResult {
    let obj = args[0];
    let needle = args.get(1).copied().unwrap_or(PY_NULL);
    unsafe {
        // int / bool / long needle → O(1) `(value - start) // step`.
        if is_int(needle) || is_long(needle) {
            let item = pyre_object::range_obj_to_bigint(needle);
            if pyre_object::w_range_contains_bigint(obj, &item) {
                return Ok(pyre_object::w_range_index_of(obj, &item));
            }
            return Err(PyError::value_error(format!(
                "{} is not in range",
                crate::display::py_repr(needle)?
            )));
        }
        // `space.sequence_index` — elementwise scan.
        let it = iter(obj)?;
        let mut i: i64 = 0;
        loop {
            match next(it) {
                Ok(item) => {
                    if is_true(compare(item, needle, CompareOp::Eq)?)? {
                        return Ok(w_int_new(i));
                    }
                    i += 1;
                }
                Err(e) if e.kind == PyErrorKind::StopIteration => break,
                Err(e) => return Err(e),
            }
        }
    }
    // `space.sequence_index` miss (`descroperation.py` `sequence_index`).
    Err(PyError::value_error(
        "sequence.index(x): x not in sequence".to_string(),
    ))
}

/// `range.__iter__()` — fresh `range_iterator` (word-fit or bignum cursor).
fn range_iter_method(args: &[PyObjectRef]) -> PyResult {
    iter(args[0])
}

/// `range.__reversed__()` — `functional.py W_Range.descr_reversed`.
fn range_reversed_method(args: &[PyObjectRef]) -> PyResult {
    unsafe { Ok(pyre_object::w_range_reversed(args[0])) }
}

/// `range.__reduce__()` — `functional.py W_Range.descr_reduce`:
/// `(type(self), (start, stop, step))`.
fn range_reduce_method(args: &[PyObjectRef]) -> PyResult {
    let (start, stop, step) = unsafe { pyre_object::w_range_fields(args[0]) };
    // `range` is bound in builtins as a constructor function, not as the
    // registry type object, so the reconstructor must be that name-bound
    // callable: `pickle.save_global` matches it to `builtins.range`, and
    // `range(start, stop, step)` rebuilds the instance.
    let range_ctor = builtin_callable("range");
    let state = w_tuple_new(vec![start, stop, step]);
    Ok(w_tuple_new(vec![range_ctor, state]))
}

/// `range.__hash__()` — `functional.py W_Range.descr_hash`: hashes the
/// `(length, start, step)` tuple, collapsing trailing fields to `None`
/// for empty (`(len, None, None)`) and single-element (`(len, start,
/// None)`) ranges so equal ranges hash equal.
fn range_hash_method(args: &[PyObjectRef]) -> PyResult {
    use num_traits::Zero;
    let (start, _stop, step) = unsafe { pyre_object::w_range_fields(args[0]) };
    let len_obj = unsafe { pyre_object::w_range_length(args[0]) };
    let len = unsafe { pyre_object::range_obj_to_bigint(len_obj) };
    let items = if len.is_zero() {
        vec![len_obj, w_none(), w_none()]
    } else if len == BigInt::from(1) {
        vec![len_obj, start, w_none()]
    } else {
        vec![len_obj, start, step]
    };
    Ok(w_int_new(hash_w_strict(w_tuple_new(items))?))
}

/// The builtin function `name` (`iter` / `enumerate`) as the
/// reconstructor for an iterator's `__reduce__`.  Mirrors PyPy's
/// `space.getbuiltin(name)` — the reduce tuple's first element must be
/// the live builtin so `pickle` recreates the iterator via `iter(seq)` /
/// `enumerate(iterable)`.
fn builtin_callable(name: &str) -> PyObjectRef {
    let ctx = crate::call::getexecutioncontext();
    if ctx.is_null() {
        return PY_NULL;
    }
    unsafe { (*ctx).lookup_builtin(name).unwrap_or(PY_NULL) }
}

/// `sequenceiterator.__reduce__()` — `iterobject.py
/// W_AbstractSeqIterObject.descr_reduce`: `(iter, (seq,), index)` for a live
/// sequence; an exhausted iterator (`w_seq is None`) pickles to `_empty_iterable`
/// (`iterobject.py:251-253`) = `(iter, ((),))` so it restores empty.
fn seq_iter_reduce_method(args: &[PyObjectRef]) -> PyResult {
    unsafe {
        let seq = pyre_object::w_seq_iter_seq(args[0]);
        if seq.is_null() {
            let empty_state = w_tuple_new(vec![w_tuple_new(vec![])]);
            return Ok(w_tuple_new(vec![builtin_callable("iter"), empty_state]));
        }
        let index = pyre_object::w_seq_iter_index(args[0]);
        let state = w_tuple_new(vec![seq]);
        Ok(w_tuple_new(vec![
            builtin_callable("iter"),
            state,
            w_int_new(index),
        ]))
    }
}

/// `sequenceiterator.__setstate__(index)` — `iterobject.py:40-45
/// W_AbstractSeqIterObject.descr_setstate`: restore the cursor only while the
/// sequence is live, clamping a negative index to 0.  There is no upper clamp —
/// an out-of-range cursor is absorbed by `next` raising StopIteration on the
/// IndexError.
fn seq_iter_setstate_method(args: &[PyObjectRef]) -> PyResult {
    let mut index = int_w(args[1])?;
    unsafe {
        if pyre_object::w_seq_iter_seq(args[0]).is_null() {
            return Ok(w_none());
        }
        if index < 0 {
            index = 0;
        }
        pyre_object::w_seq_iter_set_index(args[0], index);
    }
    Ok(w_none())
}

/// `sequenceiterator.__length_hint__()` — `W_AbstractSeqIterObject.getlength`
/// (iterobject.py:16-24): `len(seq) - index` recomputed from the LIVE sequence
/// — `space.len(w_seq)`, so a subclass `__len__` override or a mutation made
/// mid-iteration is reflected — clamped to 0.  An exhausted (cleared) sequence
/// reports 0.  A missing or raising `__len__` propagates as a real error;
/// `operator.length_hint` then maps a TypeError to its default, exactly as a
/// direct `space.len` would.
fn seq_iter_length_hint_method(args: &[PyObjectRef]) -> PyResult {
    unsafe {
        let seq = pyre_object::w_seq_iter_seq(args[0]);
        if seq.is_null() {
            return Ok(w_int_new(0));
        }
        let length = len_w(seq)?;
        let remaining = length - pyre_object::w_seq_iter_index(args[0]);
        Ok(w_int_new(remaining.max(0)))
    }
}

/// `range_iterator.__reduce__()` — `functional.py
/// W_IntRangeIterator.descr_reduce`: pyre rebuilds a `range(current, stop,
/// step)` covering the remaining span, then returns `(iter, (range,), None)`.
fn range_iter_reduce_method(args: &[PyObjectRef]) -> PyResult {
    unsafe {
        let (current, remaining, step) = pyre_object::w_range_iter_fields(args[0]);
        let stop = BigInt::from(current) + BigInt::from(remaining) * step;
        let w_range = pyre_object::w_range_new(
            w_int_new(current),
            pyre_object::range_bigint_to_obj(stop),
            w_int_new(step),
        );
        let state = w_tuple_new(vec![w_range]);
        Ok(w_tuple_new(vec![builtin_callable("iter"), state, w_none()]))
    }
}

/// `range_iterator.__length_hint__()` — remaining element count.
fn range_iter_length_hint_method(args: &[PyObjectRef]) -> PyResult {
    unsafe { Ok(w_int_new(pyre_object::w_range_iter_remaining(args[0]))) }
}

/// `longrange_iterator.__reduce__()` — rebuild a `range` covering the
/// remaining span (`current = start + index*step`, `stop = start +
/// len*step`), `(iter, (range,), None)`.
fn long_range_iter_reduce_method(args: &[PyObjectRef]) -> PyResult {
    unsafe {
        let (start, step, len, index) = pyre_object::w_long_range_iter_fields(args[0]);
        let start_b = pyre_object::range_obj_to_bigint(start);
        let step_b = pyre_object::range_obj_to_bigint(step);
        let len_b = pyre_object::range_obj_to_bigint(len);
        let index_b = pyre_object::range_obj_to_bigint(index);
        let current = &start_b + &index_b * &step_b;
        let stop = &start_b + &len_b * &step_b;
        let w_range = pyre_object::w_range_new(
            pyre_object::range_bigint_to_obj(current),
            pyre_object::range_bigint_to_obj(stop),
            pyre_object::range_bigint_to_obj(step_b),
        );
        let state = w_tuple_new(vec![w_range]);
        Ok(w_tuple_new(vec![builtin_callable("iter"), state, w_none()]))
    }
}

/// `longrange_iterator.__length_hint__()` — remaining element count.
fn long_range_iter_length_hint_method(args: &[PyObjectRef]) -> PyResult {
    unsafe {
        Ok(pyre_object::range_bigint_to_obj(
            pyre_object::w_long_range_iter_len(args[0]),
        ))
    }
}

/// `dict_keyiterator.__reduce__()` (and value/item siblings) —
/// `dictmultiobject.py W_BaseDictMultiIterObject.descr_reduce`: the remaining
/// entries as a list, wrapped `(iter, (list,))`.  No third element.
fn dict_view_iter_reduce_method(args: &[PyObjectRef]) -> PyResult {
    unsafe {
        let w_dict = pyre_object::dictmultiobject::w_dict_view_iterator_get_dict(args[0]);
        let kind = pyre_object::dictmultiobject::w_dict_view_iterator_get_kind(args[0]);
        let index = pyre_object::dictmultiobject::w_dict_view_iterator_get_index(args[0]);
        let entries = pyre_object::w_dict_items(w_dict);
        let mut items = Vec::new();
        for (k, v) in entries.into_iter().skip(index) {
            let item = match kind {
                pyre_object::dictmultiobject::DictViewKind::Keys => k,
                pyre_object::dictmultiobject::DictViewKind::Values => v,
                pyre_object::dictmultiobject::DictViewKind::Items => w_tuple_new(vec![k, v]),
            };
            items.push(item);
        }
        let state = w_tuple_new(vec![w_list_new(items)]);
        Ok(w_tuple_new(vec![builtin_callable("iter"), state]))
    }
}

/// `dict_keyiterator.__length_hint__()` — remaining entries.
fn dict_view_iter_length_hint_method(args: &[PyObjectRef]) -> PyResult {
    unsafe {
        let w_dict = pyre_object::dictmultiobject::w_dict_view_iterator_get_dict(args[0]);
        let index = pyre_object::dictmultiobject::w_dict_view_iterator_get_index(args[0]);
        let remaining = (pyre_object::w_dict_len(w_dict) as i64) - (index as i64);
        Ok(w_int_new(remaining.max(0)))
    }
}

/// `enumerate.__reduce__()` — `functional.py W_Enumerate.descr_reduce`:
/// `(enumerate, (source_iter, index))`.
fn enumerate_reduce_method(args: &[PyObjectRef]) -> PyResult {
    unsafe {
        let i64_index = pyre_object::functional::w_enumerate_get_index(args[0]);
        let raw = pyre_object::functional::w_enumerate_get_iter_or_list(args[0]);
        let w_iter = if raw.is_null() {
            // Exhausted enumerate (`:294-295` set `w_iter_or_list` to
            // null); substitute an empty seq-iter so the reduce stays
            // round-trippable.
            pyre_object::w_seq_iter_new(w_list_new(vec![]), 0)
        } else if pyre_object::is_list(raw) {
            // List fast path (`:289-294`): `w_iter_or_list` is the source
            // list itself and `index` is the cursor into it.  Materialise
            // a seq-iterator positioned at the cursor so the reconstructed
            // enumerate resumes from the right element rather than the
            // list head.
            let len = pyre_object::w_list_len(raw);
            let it = pyre_object::w_seq_iter_new(raw, len);
            let pos = i64_index.clamp(0, len as i64);
            pyre_object::w_seq_iter_set_index(it, pos);
            it
        } else {
            raw
        };
        let w_index_slot = pyre_object::functional::w_enumerate_get_w_index(args[0]);
        let index = if w_index_slot.is_null() {
            w_int_new(i64_index)
        } else {
            w_index_slot
        };
        let state = w_tuple_new(vec![w_iter, index]);
        Ok(w_tuple_new(vec![builtin_callable("enumerate"), state]))
    }
}

/// `reversed.__reduce__()` — `functional.py:407-417
/// W_ReversedIterator.descr___reduce__`: `(reversed, (sequence,),
/// remaining)` while live; `(reversed, ((),))` once exhausted (the slot
/// is cleared to `PY_NULL`).  The reconstructor is the `reversed`
/// builtin so `pickle` recreates the iterator via `reversed(sequence)`.
fn reversed_reduce_method(args: &[PyObjectRef]) -> PyResult {
    unsafe {
        let seq = pyre_object::functional::w_reversed_get_sequence(args[0]);
        if !seq.is_null() {
            let remaining = pyre_object::functional::w_reversed_get_remaining(args[0]);
            let state = w_tuple_new(vec![seq]);
            Ok(w_tuple_new(vec![
                builtin_callable("reversed"),
                state,
                w_int_new(remaining),
            ]))
        } else {
            let state = w_tuple_new(vec![w_tuple_new(vec![])]);
            Ok(w_tuple_new(vec![builtin_callable("reversed"), state]))
        }
    }
}

/// `reversed.__setstate__(index)` — `functional.py:419-429
/// descr___setstate__`: set `remaining` then clamp into `[-1, n-1]`
/// (`n == len(sequence)`, or 0 once exhausted).
fn reversed_setstate_method(args: &[PyObjectRef]) -> PyResult {
    let mut remaining = int_w(args[1])?;
    unsafe {
        let seq = pyre_object::functional::w_reversed_get_sequence(args[0]);
        let n = if !seq.is_null() { len_w(seq)? } else { 0 };
        if remaining < -1 {
            remaining = -1;
        } else if remaining > n - 1 {
            remaining = n - 1;
        }
        pyre_object::functional::w_reversed_set_remaining(args[0], remaining);
    }
    Ok(w_none())
}

/// `reversed.__length_hint__()` — `functional.py:374-383
/// descr_length_hint`: elements not yet produced, `0` once exhausted.
fn reversed_length_hint_method(args: &[PyObjectRef]) -> PyResult {
    unsafe {
        let remaining = pyre_object::functional::w_reversed_get_remaining(args[0]);
        let mut res = 0i64;
        if remaining >= 0 {
            let seq = pyre_object::functional::w_reversed_get_sequence(args[0]);
            let total = if !seq.is_null() { len_w(seq)? } else { 0 };
            let rem_length = remaining + 1;
            if rem_length <= total {
                res = rem_length;
            }
        }
        Ok(w_int_new(res))
    }
}

/// `filter.__reduce__()` — `functional.py:944-949 W_Filter.descr_reduce`:
/// `(filter, (predicate, iterable))`, where `predicate` is `None` when the
/// stored predicate is `PY_NULL`.  Pickle recreates the iterator via
/// `filter(predicate, iterable)`; the captured iterator carries its
/// position.
fn filter_reduce_method(args: &[PyObjectRef]) -> PyResult {
    unsafe {
        let w_predicate = pyre_object::functional::w_filter_get_predicate(args[0]);
        let w_predicate = if w_predicate.is_null() {
            w_none()
        } else {
            w_predicate
        };
        let w_iterable = pyre_object::functional::w_filter_get_iterable(args[0]);
        let state = w_tuple_new(vec![w_predicate, w_iterable]);
        Ok(w_tuple_new(vec![builtin_callable("filter"), state]))
    }
}

/// `functional.py:1065-1067 _raise_strict_error` — the strict zip/map
/// length-mismatch `ValueError`.  `index` is the 0-based position whose
/// argument is short/long; the message numbers are 1-based.
fn strict_zip_error(func_name: &str, index: usize, adjective: &str) -> PyError {
    let plural = if index == 1 { " " } else { "s 1-" };
    PyError::new(
        PyErrorKind::ValueError,
        format!(
            "{}() argument {} is {} than argument{}{}",
            func_name,
            index + 1,
            adjective,
            plural,
            index
        ),
    )
}

/// Pull one item from each iterator in the `list` `w_iterators`, returning
/// them.  `Ok(None)` is a normal stop (the shortest is exhausted, non-strict).
/// In `strict` mode a length mismatch raises `ValueError` naming `func_name`
/// ("zip" / "map").  Shared by `map` and `zip`
/// (`functional.py:1022-1079 W_Zip.next_w`).
///
/// # Safety
/// `w_iterators` must be a valid `list` of iterator objects.
unsafe fn pull_iterator_tuple(
    w_iterators: PyObjectRef,
    strict: bool,
    func_name: &str,
) -> Result<Option<Vec<PyObjectRef>>, PyError> {
    let n = pyre_object::w_list_len(w_iterators) as usize;
    if n == 0 {
        return Ok(None);
    }
    // Each pulled value must survive the `next()` calls on the later iterators:
    // those allocate and can relocate the young objects already pulled, leaving
    // a stale pointer in a plain `Vec`. Pin each value into the shadow stack as
    // it is produced, then re-read the set at its (possibly relocated) address
    // before returning.
    let _roots = pyre_object::gc_roots::push_roots();
    // Pin the iterator list itself: `next(it)` runs Python and can move it, and
    // later iterations dereference it again — a raw local would go stale.
    pyre_object::gc_roots::pin_root(w_iterators);
    let iters_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let base = pyre_object::gc_roots::shadow_stack_len();
    for i in 0..n {
        let it = pyre_object::w_list_getitem(
            pyre_object::gc_roots::shadow_stack_get(iters_slot),
            i as i64,
        )
        .unwrap();
        match next(it) {
            Ok(v) => pyre_object::gc_roots::pin_root(v),
            Err(e) if e.kind == PyErrorKind::StopIteration => {
                if !strict {
                    return Ok(None);
                }
                // A StopIteration in strict mode is a length mismatch.
                // `i` iterators yielded before this one ran dry.
                if i > 0 {
                    return Err(strict_zip_error(func_name, i, "shorter"));
                }
                if n == 1 {
                    // A single iterable can never mismatch.
                    return Ok(None);
                }
                if n == 2 {
                    // `functional.py:1047-1054` — the first ran dry; if the
                    // second still yields it is the longer one.
                    let it1 = pyre_object::w_list_getitem(
                        pyre_object::gc_roots::shadow_stack_get(iters_slot),
                        1,
                    )
                    .unwrap();
                    return match next(it1) {
                        Ok(_) => Err(strict_zip_error(func_name, 1, "longer")),
                        Err(e2) if e2.kind == PyErrorKind::StopIteration => Ok(None),
                        Err(e2) => Err(e2),
                    };
                }
                // `functional.py:1069-1079 _validate_strict` — the first ran
                // dry; any later iterator that still yields is the longer one.
                // Start at 1: iterator 0 is the one already known exhausted, so
                // re-`next`ing it is a wasted (and on a side-effectful iterator,
                // observable) call.
                for j in 1..n {
                    let itj = pyre_object::w_list_getitem(
                        pyre_object::gc_roots::shadow_stack_get(iters_slot),
                        j as i64,
                    )
                    .unwrap();
                    match next(itj) {
                        Ok(_) => return Err(strict_zip_error(func_name, j, "longer")),
                        Err(e2) if e2.kind == PyErrorKind::StopIteration => {}
                        Err(e2) => return Err(e2),
                    }
                }
                return Ok(None);
            }
            Err(e) => return Err(e),
        }
    }
    let items: Vec<PyObjectRef> = (0..n)
        .map(|k| pyre_object::gc_roots::shadow_stack_get(base + k))
        .collect();
    Ok(Some(items))
}

/// `map.__reduce__()` — `functional.py:869-873 W_Map.descr_reduce`:
/// `(map, (func, *iterators))`, with a trailing `True` when `strict`
/// (CPython 3.14).  The captured iterators carry their positions.
fn map_reduce_method(args: &[PyObjectRef]) -> PyResult {
    unsafe {
        let w_fun = pyre_object::functional::w_map_get_fun(args[0]);
        let w_iterators = pyre_object::functional::w_map_get_iterators(args[0]);
        let n = pyre_object::w_list_len(w_iterators);
        let mut state_items = Vec::with_capacity(n as usize + 1);
        state_items.push(w_fun);
        for i in 0..n {
            state_items.push(pyre_object::w_list_getitem(w_iterators, i as i64).unwrap());
        }
        let state = w_tuple_new(state_items);
        let map_fn = builtin_callable("map");
        if pyre_object::functional::w_map_get_strict(args[0]) {
            Ok(w_tuple_new(vec![map_fn, state, w_bool_from(true)]))
        } else {
            Ok(w_tuple_new(vec![map_fn, state]))
        }
    }
}

/// `map.__setstate__(strict)` — CPython 3.14: set the `strict` flag from the
/// unpickled state.
fn map_setstate_method(args: &[PyObjectRef]) -> PyResult {
    let strict = is_true(args[1])?;
    unsafe {
        pyre_object::functional::w_map_set_strict(args[0], strict);
    }
    Ok(w_none())
}

/// `zip.__reduce__()` — `functional.py:1081-1087 W_Zip.descr_reduce`:
/// `(zip, (*iterators))`, with a trailing `True` when `strict`.
fn zip_reduce_method(args: &[PyObjectRef]) -> PyResult {
    unsafe {
        let w_iterators = pyre_object::functional::w_zip_get_iterators(args[0]);
        let n = pyre_object::w_list_len(w_iterators);
        let mut state_items = Vec::with_capacity(n as usize);
        for i in 0..n {
            state_items.push(pyre_object::w_list_getitem(w_iterators, i as i64).unwrap());
        }
        let state = w_tuple_new(state_items);
        let zip_fn = builtin_callable("zip");
        if pyre_object::functional::w_zip_get_strict(args[0]) {
            Ok(w_tuple_new(vec![zip_fn, state, w_bool_from(true)]))
        } else {
            Ok(w_tuple_new(vec![zip_fn, state]))
        }
    }
}

/// `zip.__setstate__(strict)` — `functional.py:1089-1091
/// W_Zip.descr_setstate`: `self.strict = bool(state)`.
fn zip_setstate_method(args: &[PyObjectRef]) -> PyResult {
    let strict = is_true(args[1])?;
    unsafe {
        pyre_object::functional::w_zip_set_strict(args[0], strict);
    }
    Ok(w_none())
}

unsafe fn getitem_range_iter(obj: PyObjectRef, index: PyObjectRef) -> PyResult {
    let r = &*(obj as *const pyre_object::functional::W_IntRangeIterator);
    let len = r.remaining;
    if is_int(index) {
        // range[i]
        let i = w_int_get_value(index);
        let idx = if i < 0 { len + i } else { i };
        if idx < 0 || idx >= len {
            return Err(PyError::new(
                PyErrorKind::IndexError,
                "range object index out of range",
            ));
        }
        return Ok(w_int_new(r.current + idx * r.step));
    }
    if is_slice(index) {
        // range[start:stop:step] → returns a list
        let (start, stop, step) = normalize_slice(index, len)?;
        let mut items = Vec::new();
        let mut i = start;
        while (step > 0 && i < stop) || (step < 0 && i > stop) {
            items.push(w_int_new(r.current + i * r.step));
            i += step;
        }
        return Ok(w_list_new(items));
    }
    Err(index_type_error("range", index))
}

/// `pypy/interpreter/baseobjspace.py:870 finditem` — return the value
/// for `key` in `obj`, or `None` if absent.  PyPy catches only the
/// `KeyError` arm and re-raises any other `OperationError`; in Rust
/// the re-raise surfaces as `Result::Err`, the absent case as
/// `Ok(None)`, and a hit as `Ok(Some(value))`.
pub fn finditem(obj: PyObjectRef, index: PyObjectRef) -> Result<Option<PyObjectRef>, PyError> {
    match getitem(obj, index) {
        Ok(value) => Ok(Some(value)),
        Err(err) if err.kind == crate::PyErrorKind::KeyError => Ok(None),
        Err(err) => Err(err),
    }
}

/// Set item by index: `obj[index] = value`.

// `STORE_SUBSCR` discards the result (`pypy/interpreter/pyopcode.py:702`
// calls `space.setitem` as a bare statement), so the traced residual
// call is a void `CALL_N` (`rpython/jit/codewriter/jtransform.py
// handle_residual_call` keys `result_kind` off the discarded result's
// Void concretetype). descroperation.py:389 setitem returns
// `space.get_and_call_function`'s result: builtin containers'
// `__setitem__` yield None; an instance's `__setitem__` yields its own
// return value.  STORE_SUBSCR and the `jit_setitem` residual drop this
// result, so the void-ness lives at the opcode boundary, not in this
// method's `PyResult` type.
pub fn setitem(obj: PyObjectRef, index: PyObjectRef, value: PyObjectRef) -> PyResult {
    let obj = unwrap_cell(obj);
    let index = unwrap_cell(index);
    let value = unwrap_cell(value);
    unsafe {
        // `pypy/objspace/std/dictproxyobject.py` exposes neither
        // `__setitem__` nor `__delitem__`, so `space.setitem` on a
        // mappingproxy raises `TypeError: 'mappingproxy' object does
        // not support item assignment`.  Detect proxy before any
        // dict-like assignment fallthrough.
        if pyre_object::is_dict_proxy(obj) {
            return Err(PyError::type_error(
                "'mappingproxy' object does not support item assignment",
            ));
        }
        // A builtin sequence subclass overriding `__setitem__` dispatches the
        // override; exact instances and non-overriding subclasses fall through
        // to the by-layout assignment slot below.
        if is_list(obj) || pyre_object::bytearrayobject::is_bytearray(obj) {
            if let Some((method, w_type)) = subclass_special_override(obj, "__setitem__") {
                return get_and_call_function(method, obj, w_type, &[index, value]);
            }
        }
    }
    setitem_slot(obj, index, value)
}

/// The builtin `__setitem__` slot body: item-assignment dispatch by concrete
/// layout.  Reached from the operator [`setitem`] for exact instances and
/// non-overriding subclasses, and bound directly as the `list` `__setitem__`
/// slot so a subclass override's `super().__setitem__` resolves to the
/// inherited builtin assignment instead of re-entering override dispatch
/// (which would recurse).
pub(crate) fn setitem_slot(obj: PyObjectRef, index: PyObjectRef, value: PyObjectRef) -> PyResult {
    let obj = unwrap_cell(obj);
    let index = unwrap_cell(index);
    let value = unwrap_cell(value);
    unsafe {
        if is_list(obj) {
            return setitem_list(obj, index, value);
        }
        if is_dict(obj) {
            return match pyre_object::dictmultiobject::w_dict_store_checked(obj, index, value) {
                Ok(()) => Ok(w_none()),
                Err(_) => Err(take_pending_hash_error()),
            };
        }
        if pyre_object::bytearrayobject::is_bytearray(obj) {
            return setitem_bytearray(obj, index, value);
        }
        if is_instance(obj) {
            return setitem_instance(obj, index, value);
        }
        // descroperation.py:382-392 DescrOperation.setitem — any object
        // whose type defines `__setitem__` on its MRO supports item
        // assignment (the arms above are fast paths for builtin mutable
        // containers).  Mirrors the generic `__getitem__` fallback in
        // `getitem_slot`; covers native W_Root types like `memoryview`
        // whose typedef registers `__setitem__`.
        if let Some(w_type) = crate::typedef::r#type(obj) {
            if let Some(method) = lookup_in_type_where(w_type, "__setitem__") {
                return get_and_call_function(method, obj, w_type, &[index, value]);
            }
        }
        Err(PyError::type_error(format!(
            "'{}' object does not support item assignment",
            (*(*obj).ob_type).name,
        )))
    }
}

#[inline(never)]
unsafe fn setitem_list(obj: PyObjectRef, index: PyObjectRef, value: PyObjectRef) -> PyResult {
    if is_slice(index) {
        return setitem_list_slice(obj, index, value);
    }
    if !is_int(index) {
        return Err(index_type_error("list", index));
    }
    let idx = w_int_get_value(index);
    if w_list_setitem(obj, idx, value) {
        Ok(w_none())
    } else {
        Err(PyError::new(
            PyErrorKind::IndexError,
            "list assignment index out of range",
        ))
    }
}

#[inline(never)]
unsafe fn setitem_list_slice(obj: PyObjectRef, index: PyObjectRef, value: PyObjectRef) -> PyResult {
    let len = w_list_len(obj) as i64;
    let (start, stop, step) = normalize_slice(index, len)?;
    // listobject.py:709-714 wraps non-list iterables into a
    // temporary W_ListObject so the strategy-aware setslice
    // (`listobject.py:1746-1758`) and extended-slice
    // (`listobject.py:descr_setitem` step != 1 branch) paths
    // see a list operand.
    let w_other = if pyre_object::is_list(value) {
        value
    } else {
        let items = crate::builtins::collect_iterable(value)?;
        pyre_object::listobject::w_list_new(items)
    };
    if step == 1 {
        let s_lo = start.max(0) as usize;
        let s_hi = stop.max(0) as usize;
        pyre_object::listobject::w_list_setslice(obj, s_lo, s_hi, w_other)
            .expect("w_other is always a valid list");
        return Ok(w_none());
    }
    // Extended slice: `pypy/objspace/std/listobject.py
    // W_ListObject.descr_setitem` enforces equal length
    // ("attempt to assign sequence of size %d to extended
    // slice of size %d") and writes positions in order.
    let mut indices = Vec::new();
    let mut i = start;
    while (step > 0 && i < stop) || (step < 0 && i > stop) {
        if i >= 0 && i < len {
            indices.push(i);
        }
        i += step;
    }
    let other_len = pyre_object::w_list_len(w_other);
    if other_len != indices.len() {
        return Err(PyError::new(
            PyErrorKind::ValueError,
            format!(
                "attempt to assign sequence of size {} to extended slice of size {}",
                other_len,
                indices.len()
            ),
        ));
    }
    for (k, &idx) in indices.iter().enumerate() {
        let item =
            pyre_object::w_list_getitem(w_other, k as i64).expect("k < other_len by construction");
        if !pyre_object::w_list_setitem(obj, idx, item) {
            return Err(PyError::new(
                PyErrorKind::IndexError,
                "list assignment index out of range",
            ));
        }
    }
    Ok(w_none())
}

/// Resolve a `bytearray` subscript index (`bytearray_ass_subscript`): honor
/// `__index__`, raise the "indices must be integers or slices" TypeError for a
/// non-index, non-slice key, and raise IndexError for a value too large to fit
/// an index (`PyNumber_AsSsize_t(index, IndexError)`).
unsafe fn bytearray_index(index: PyObjectRef) -> Result<i64, PyError> {
    if !pyre_object::pyobject::is_int_or_long(index) && lookup(index, "__index__").is_none() {
        return Err(index_type_error("bytearray", index));
    }
    match int_w(space_index(index)?) {
        Ok(i) => Ok(i),
        // `baseobjspace.py getindex_w` — an index that overflows a machine
        // word reports the *source* object's type, `oefmt("cannot fit '%T'
        // into an index-sized integer", w_obj)`, not the coerced `int`.
        Err(e) if e.kind == PyErrorKind::OverflowError => Err(PyError::new(
            PyErrorKind::IndexError,
            format!(
                "cannot fit '{}' into an index-sized integer",
                object_functionstr_type_name(index)
            ),
        )),
        Err(e) => Err(e),
    }
}

#[inline(never)]
unsafe fn setitem_bytearray(obj: PyObjectRef, index: PyObjectRef, value: PyObjectRef) -> PyResult {
    if is_slice(index) {
        return setitem_bytearray_slice(obj, index, value);
    }
    // `descr_setitem`: getindex_w(index) → _fixindex(idx) → coerce value.  The
    // index gate and byte coercion are inlined (not routed through the shared
    // `bytearray_index`/`byte_w`) and the coercion is kept inside the in-bounds
    // block: a shared-function result fails to bind a concrete Int repr in the
    // rtyper (concretetype None), which the codewriter then mis-colors Ref
    // against its Int provenance.  `__index__` index support and coercing before
    // the bounds check (so `ba[oob] = bad` reports the value error ahead of
    // IndexError) are deferred on that same kind-provenance gap.
    if !is_int(index) {
        return Err(PyError::type_error("bytearray indices must be integers"));
    }
    let idx = w_int_get_value(index);
    let len = pyre_object::bytearrayobject::w_bytearray_len(obj) as i64;
    let actual = if idx < 0 { len + idx } else { idx };
    if actual >= 0 && actual < len {
        let v = if is_int(value) {
            w_int_get_value(value)
        } else {
            let indexed = space_index(value)?;
            if is_int(indexed) {
                w_int_get_value(indexed)
            } else {
                match i64::try_from(w_long_get_value(indexed)) {
                    Ok(v) => v,
                    Err(_) => {
                        return Err(PyError::value_error("byte must be in range(0, 256)"));
                    }
                }
            }
        };
        if !(0..=255).contains(&v) {
            return Err(PyError::value_error("byte must be in range(0, 256)"));
        }
        pyre_object::bytearrayobject::w_bytearray_setitem(obj, actual as usize, v as u8);
        return Ok(w_none());
    }
    Err(PyError::new(
        PyErrorKind::IndexError,
        "bytearray index out of range",
    ))
}

/// `space.byte_w` (`bytearrayobject.py _getbytevalue` / `bytesobject.py
/// _from_byte_sequence_loop`) — coerce a value to a single byte: honor
/// `__index__`, then enforce `0 <= v < 256`.  `noun` selects the divergent
/// range-error text — "byte" for bytearray, "bytes" for the bytes constructor.
pub(crate) unsafe fn byte_w(value: PyObjectRef, noun: &str) -> Result<u8, PyError> {
    let v = if is_int(value) {
        w_int_get_value(value)
    } else {
        let indexed = space_index(value)?;
        if is_int(indexed) {
            w_int_get_value(indexed)
        } else {
            // `space.index` may yield a long; one that overflows i64 is
            // necessarily outside 0..256 → the ValueError below.
            match i64::try_from(w_long_get_value(indexed)) {
                Ok(v) => v,
                Err(_) => {
                    return Err(PyError::value_error(format!(
                        "{noun} must be in range(0, 256)"
                    )));
                }
            }
        }
    };
    if !(0..=255).contains(&v) {
        return Err(PyError::value_error(format!(
            "{noun} must be in range(0, 256)"
        )));
    }
    Ok(v as u8)
}

/// `bytesobject.py makebytesdata_w` — coerce a slice-assignment source to
/// raw bytes: a buffer (bytes/bytearray/array/memoryview) yields its bytes,
/// otherwise an iterable of ints is range-checked element-wise.  A `str` or
/// non-iterable source is rejected.
unsafe fn bytearray_assign_source(value: PyObjectRef) -> Result<Vec<u8>, PyError> {
    if let Some(src) = crate::typedef::buffer_as_bytes_like(value)? {
        return Ok(pyre_object::bytesobject::bytes_like_data(src).to_vec());
    }
    // A `str` or index operand (`= "x"` / `= 5`) is the common mis-assignment
    // → the "can assign only ..." hint; any other non-iterable is "cannot convert".
    if is_str(value)
        || pyre_object::pyobject::is_int_or_long(value)
        || lookup(value, "__index__").is_some()
    {
        return Err(PyError::type_error(
            "can assign only bytes, buffers, or iterables of ints in range(0, 256)",
        ));
    }
    // `_from_byte_sequence` / `_from_byte_sequence_loop` — iterate the source,
    // converting and range-checking each item as it is pulled
    // (`builder.append(space.byte_w(w_item))` per element), so a bad byte or a
    // raising `__next__` surfaces at once without draining the rest (an
    // infinite iterator that yields a bad byte still fails immediately).  A
    // source with no `__iter__` is the non-iterable "cannot convert" case; an
    // error raised *by* `__iter__`/`__next__` propagates unchanged.
    let it = match crate::baseobjspace::iter(value) {
        Ok(it) => it,
        Err(e) => {
            if lookup(value, "__iter__").is_none() {
                return Err(PyError::type_error(format!(
                    "cannot convert '{}' object to bytearray",
                    object_functionstr_type_name(value),
                )));
            }
            return Err(e);
        }
    };
    let mut out = Vec::new();
    loop {
        match crate::baseobjspace::next(it) {
            Ok(w_item) => out.push(byte_w(w_item, "byte")?),
            Err(e) if e.kind == PyErrorKind::StopIteration => break,
            Err(e) => return Err(e),
        }
    }
    Ok(out)
}

/// `bytearrayobject.py descr_setitem` slice branch + `_setitem_slice_helper`
/// — replace a (possibly extended) slice with the source bytes, resizing for
/// step-1 slices and requiring equal length for extended slices.
#[inline(never)]
unsafe fn setitem_bytearray_slice(
    obj: PyObjectRef,
    index: PyObjectRef,
    value: PyObjectRef,
) -> PyResult {
    // `descr_setitem`: materialize the source (`makebytesdata_w`) first, then
    // `_unpack_slice` — evaluate the slice's `__index__` before reading the
    // length, since both the source's `__iter__`/`__next__` and the slice
    // components' `__index__` may mutate the bytearray, and the bounds must be
    // clamped against the post-mutation length. (`x[:] = x` stays safe — the
    // source is copied into `sequence2`.)
    let sequence2 = bytearray_assign_source(value)?;
    let (rs, rp, st) = crate::sliceobject::slice_unpack(
        w_slice_get_start(index),
        w_slice_get_stop(index),
        w_slice_get_step(index),
    )?;
    let len = pyre_object::bytearrayobject::w_bytearray_len(obj) as i64;
    let (start, stop, step, _) = crate::sliceobject::slice_adjust_indices(rs, rp, st, len);
    let vec = pyre_object::bytearrayobject::w_bytearray_vec_mut(obj);
    if step == 1 {
        let cur = vec.len();
        let s = (start.max(0) as usize).min(cur);
        let e = (stop.max(start) as usize).min(cur).max(s);
        vec.splice(s..e, sequence2.iter().copied());
        return Ok(w_none());
    }
    // Extended slice: `descr_setitem` forbids resizing — the source length
    // must equal the slice length; positions are written in order.
    let mut indices = Vec::new();
    let mut i = start;
    while (step > 0 && i < stop) || (step < 0 && i > stop) {
        if i >= 0 && i < len {
            indices.push(i as usize);
        }
        i += step;
    }
    if sequence2.len() != indices.len() {
        return Err(PyError::new(
            PyErrorKind::ValueError,
            format!(
                "attempt to assign bytes of size {} to extended slice of size {}",
                sequence2.len(),
                indices.len()
            ),
        ));
    }
    for (k, &idx) in indices.iter().enumerate() {
        if let Some(slot) = vec.get_mut(idx) {
            *slot = sequence2[k];
        }
    }
    Ok(w_none())
}

#[inline(never)]
unsafe fn setitem_instance(obj: PyObjectRef, index: PyObjectRef, value: PyObjectRef) -> PyResult {
    // descroperation.py:389 `space.get_and_call_function(w_descr, w_obj,
    // w_key, w_value)` — bind the `__setitem__` descriptor to the receiver
    // before calling with `(index, value)`.
    let w_type = w_instance_get_type(obj);
    if let Some(method) = lookup_in_type_where(w_type, "__setitem__") {
        return get_and_call_function(method, obj, w_type, &[index, value]);
    }
    Err(PyError::type_error(format!(
        "'{}' object does not support item assignment",
        w_type_get_name(w_instance_get_type(obj)),
    )))
}

/// String-keyed `finditem` shorthand: `space.finditem_str(w_obj, key)`.
pub fn finditem_str(obj: PyObjectRef, key: &str) -> Result<Option<PyObjectRef>, PyError> {
    finditem(obj, w_str_new(key))
}

/// PyPy-compatible identity check returning a raw boolean value.
pub fn is_w(w_one: PyObjectRef, w_two: PyObjectRef) -> bool {
    if std::ptr::eq(w_one, w_two) {
        return true;
    }
    // `W_AbstractIntObject.is_w` (intobject.py:44-53): two plain `int`s
    // — `W_IntObject` or the BigInt-backed `W_LongObject` — are
    // identical when their values are equal.  `bool`
    // (`W_BoolObject.is_w` is pure pointer identity, boolobject.py:25)
    // and `int` subclasses (`user_overridden_class`) keep pointer
    // identity — the exact-type gate excludes both (a `bool`'s
    // `w_class` is `bool`, a subclass instance's is the subclass), so
    // they fall through to the `ptr::eq` above.
    unsafe {
        if pyre_object::pyobject::is_exact_type(w_one, &pyre_object::pyobject::INT_TYPE)
            && pyre_object::pyobject::is_exact_type(w_two, &pyre_object::pyobject::INT_TYPE)
        {
            // `space.bigint_w(self).eq(space.bigint_w(w_other))`
            // (intobject.py:51-53). A `W_LongObject` stores a `BigInt`
            // pointer, so it must be read as a bigint, not as an i64.
            return pyre_object::functional::range_obj_to_bigint(w_one)
                == pyre_object::functional::range_obj_to_bigint(w_two);
        }
        // `W_FloatObject.is_w` (floatobject.py:196-204): two plain
        // `float`s are identical when their bit patterns are equal
        // (`float2longlong`), so `0.0 is -0.0` is false and a NaN is its
        // own identity. `float` subclasses (`user_overridden_class`) keep
        // pointer identity — the exact-type gate excludes them.
        if pyre_object::pyobject::is_exact_type(w_one, &pyre_object::pyobject::FLOAT_TYPE)
            && pyre_object::pyobject::is_exact_type(w_two, &pyre_object::pyobject::FLOAT_TYPE)
        {
            return pyre_object::floatobject::w_float_get_value(w_one).to_bits()
                == pyre_object::floatobject::w_float_get_value(w_two).to_bits();
        }
        // `W_ComplexObject.is_w` (complexobject.py:287-301): two plain
        // `complex`es are identical when both component bit patterns are
        // equal (`float2longlong`). `complex` subclasses
        // (`user_overridden_class`) keep pointer identity.
        if pyre_object::pyobject::is_exact_type(w_one, &pyre_object::pyobject::COMPLEX_TYPE)
            && pyre_object::pyobject::is_exact_type(w_two, &pyre_object::pyobject::COMPLEX_TYPE)
        {
            return pyre_object::complexobject::w_complex_get_real(w_one).to_bits()
                == pyre_object::complexobject::w_complex_get_real(w_two).to_bits()
                && pyre_object::complexobject::w_complex_get_imag(w_one).to_bits()
                    == pyre_object::complexobject::w_complex_get_imag(w_two).to_bits();
        }
        // `W_AbstractTupleObject.is_w` (tupleobject.py:47-55): a `tuple` is
        // identical to another only when both are the empty tuple — "empty
        // tuples are unique-ified". Non-empty tuples keep pointer identity
        // (the `ptr::eq` above handled `self is w_other`); `tuple`
        // subclasses keep pointer identity through the exact-type gate. The
        // specialised arity-2 tuples carry the canonical `tuple` w_class, so
        // they pass the gate but are never empty (length 2).
        if pyre_object::pyobject::is_exact_type(w_one, &pyre_object::pyobject::TUPLE_TYPE)
            && pyre_object::pyobject::is_exact_type(w_two, &pyre_object::pyobject::TUPLE_TYPE)
        {
            return pyre_object::tupleobject::w_tuple_len(w_one) == 0
                && pyre_object::tupleobject::w_tuple_len(w_two) == 0;
        }
        // `W_AbstractBytesObject.is_w` (bytesobject.py:24-38): for distinct
        // exact-`bytes` operands, `len(s2) > 1` returns `s1 is s2` (storage
        // identity) — distinct `bytes` never share their backing buffer, so
        // `false`; `len(s2) == 0` returns `len(s1) == 0`; `len(s2) == 1`
        // (unique-ified) returns `len(s1) == 1 && s1[0] == s2[0]`.
        if pyre_object::pyobject::is_exact_type(w_one, &pyre_object::bytesobject::BYTES_TYPE)
            && pyre_object::pyobject::is_exact_type(w_two, &pyre_object::bytesobject::BYTES_TYPE)
        {
            let len1 = pyre_object::bytesobject::w_bytes_len(w_one);
            let len2 = pyre_object::bytesobject::w_bytes_len(w_two);
            if len2 > 1 {
                return false;
            }
            if len2 == 0 {
                return len1 == 0;
            }
            return len1 == 1
                && pyre_object::bytesobject::w_bytes_getitem(w_one, 0)
                    == pyre_object::bytesobject::w_bytes_getitem(w_two, 0);
        }
        // `W_UnicodeObject.is_w` (unicodeobject.py:101-113): `_len()` is the
        // codepoint count. When it is > 1, upstream returns `s1 is s2`
        // (utf8 storage identity) — distinct `str`s never share storage, so
        // `false`; when it is <= 1 (unique-ified) it returns `s1 == s2`,
        // i.e. WTF-8 byte equality. `str` subclasses keep pointer identity
        // through the exact-type gate.
        if pyre_object::pyobject::is_exact_type(w_one, &pyre_object::pyobject::STR_TYPE)
            && pyre_object::pyobject::is_exact_type(w_two, &pyre_object::pyobject::STR_TYPE)
        {
            if pyre_object::unicodeobject::w_str_len(w_one) > 1 {
                return false;
            }
            return pyre_object::unicodeobject::w_str_get_wtf8(w_one)
                == pyre_object::unicodeobject::w_str_get_wtf8(w_two);
        }
        // `W_FrozensetObject.is_w` (setobject.py:592-600): two `frozenset`s
        // are identical only when both are empty — "empty frozensets are
        // unique-ified". The mutable `set` carries a distinct type tag and
        // does not override `is_w`, so the `FROZENSET_TYPE` gate excludes
        // it; `frozenset` subclasses are excluded too.
        if pyre_object::pyobject::is_exact_type(w_one, &pyre_object::setobject::FROZENSET_TYPE)
            && pyre_object::pyobject::is_exact_type(w_two, &pyre_object::setobject::FROZENSET_TYPE)
        {
            return pyre_object::setobject::w_set_len(w_one) == 0
                && pyre_object::setobject::w_set_len(w_two) == 0;
        }
    }
    false
}

/// PyPy-compatible identity check returning a Python bool object.
pub fn is_(w_one: PyObjectRef, w_two: PyObjectRef) -> PyObjectRef {
    w_bool_from(is_w(w_one, w_two))
}

/// `W_TypeObject.flag_sequence_bug_compat` — set on exactly the builtin
/// sequence types (list/tuple/bytes/bytearray/str); subclasses do not
/// inherit it, so this is an exact type-object identity check.  Used by
/// the in-place `+=` / `*=` bug-to-bug compatibility branch in
/// descroperation.
pub fn flag_sequence_bug_compat(w_type: PyObjectRef) -> bool {
    use pyre_object::pyobject;
    is_w(w_type, crate::typedef::gettypeobject(&pyobject::LIST_TYPE))
        || is_w(w_type, crate::typedef::gettypeobject(&pyobject::TUPLE_TYPE))
        || is_w(w_type, crate::typedef::gettypeobject(&pyobject::STR_TYPE))
        || is_w(
            w_type,
            crate::typedef::gettypeobject(&pyre_object::bytesobject::BYTES_TYPE),
        )
        || is_w(
            w_type,
            crate::typedef::gettypeobject(&pyre_object::bytearrayobject::BYTEARRAY_TYPE),
        )
}

/// Python-level `not` operation. descroperation.py:289-290
/// `not_ = space.newbool(not space.is_true(w_obj))`; the `is_true` call
/// may raise, so the result is fallible.
pub fn not_(obj: PyObjectRef) -> Result<PyObjectRef, PyError> {
    Ok(w_bool_from(!is_true(obj)?))
}

/// PyPy-compatible attribute lookup returning `None` when not found.
pub fn findattr(obj: PyObjectRef, name: &str) -> Option<PyObjectRef> {
    if unsafe { is_none(obj) } {
        return None;
    }
    match getattr_str(obj, name) {
        Ok(value) => Some(value),
        Err(err) => {
            if err.kind == crate::PyErrorKind::AttributeError
                || err.kind == crate::PyErrorKind::NameError
            {
                None
            } else {
                panic!("space.findattr: unexpected {err:?}");
            }
        }
    }
}

/// Like [`findattr`] but propagates a non-`AttributeError`/`NameError` error
/// (e.g. a descriptor or `__getattr__` raising) instead of panicking. `Ok(None)`
/// means the attribute is absent.
pub fn findattr_result(obj: PyObjectRef, name: &str) -> Result<Option<PyObjectRef>, PyError> {
    if unsafe { is_none(obj) } {
        return Ok(None);
    }
    match getattr_str(obj, name) {
        Ok(value) => Ok(Some(value)),
        Err(err) => {
            if err.kind == crate::PyErrorKind::AttributeError
                || err.kind == crate::PyErrorKind::NameError
            {
                Ok(None)
            } else {
                Err(err)
            }
        }
    }
}

/// Check whether `exc_type` matches `check_class`, including tuple/list class inputs.
pub fn exception_match(exc_type: PyObjectRef, check_class: PyObjectRef) -> bool {
    let (exc_type, check_class) = (exc_type, check_class);
    if unsafe { is_none(check_class) || is_none(exc_type) } {
        return false;
    }

    let is_tuple_check = unsafe { is_tuple(check_class) };
    if is_tuple_check {
        let len = unsafe { w_tuple_len(check_class) };
        for i in 0..len {
            let candidate = unsafe { w_tuple_getitem(check_class, i as i64) };
            if let Some(candidate) = candidate {
                if exception_match(exc_type, candidate) {
                    return true;
                }
            }
        }
        return false;
    }

    // Python 3: except clause only accepts tuple, not list.
    if !unsafe { is_type(check_class) } {
        return false;
    }

    if is_w(exc_type, check_class) {
        return true;
    }

    let mro_ptr = unsafe { w_type_get_mro(exc_type) };
    if mro_ptr.is_null() {
        return false;
    }

    let mro = unsafe { &*mro_ptr };
    mro.iter().any(|&klass| is_w(klass, check_class))
}

/// Get the length of a container: `len(obj)`.
pub fn len(obj: PyObjectRef) -> PyResult {
    // descroperation.py:294-298 `_len` — a builtin subclass overriding
    // `__len__` dispatches the override (bound via `get_and_call_function`);
    // exact builtins and non-overriding subclasses fall through to the
    // by-layout slot body, which gives the inherited builtin length without
    // re-entering override dispatch.
    unsafe {
        if let Some((method, w_type)) = subclass_special_override(obj, "__len__") {
            return get_and_call_function(method, obj, w_type, &[]);
        }
    }
    len_slot(obj)
}

/// The builtin `__len__` slot body: length dispatch by concrete layout.
/// Reached from the operator [`len`] for exact instances and non-overriding
/// subclasses, and bound directly as the `list`/`str` `__len__` slot so a
/// subclass override's `super().__len__` resolves to the inherited builtin
/// length instead of re-entering override dispatch (which would recurse).
pub(crate) fn len_slot(obj: PyObjectRef) -> PyResult {
    // `pypy/objspace/std/dictproxyobject.py:32 descr_len` →
    // `space.len(self.w_mapping)`.
    let obj = unsafe {
        if pyre_object::is_dict_proxy(obj) {
            pyre_object::w_dict_proxy_get_mapping(obj)
        } else {
            obj
        }
    };
    // `pypy/objspace/std/dictmultiobject.py`
    // `W_DictViewKeysObject.descr_len` returns `space.len(self.w_dict)`
    // for all three view
    // kinds.  Forward to the source dict so the view's len reflects
    // live mutations on the dict, matching PyPy's view semantics.
    unsafe {
        if pyre_object::dictmultiobject::is_dict_view(obj) {
            let dict = pyre_object::dictmultiobject::w_dict_view_get_dict(obj);
            if dict.is_null() {
                return Ok(w_int_new(0));
            }
            return Ok(w_int_new(pyre_object::w_dict_len(dict) as i64));
        }
        if is_list(obj) {
            return Ok(w_int_new(w_list_len(obj) as i64));
        }
        if is_tuple(obj) {
            return Ok(w_int_new(w_tuple_len(obj) as i64));
        }
        if is_dict(obj) {
            return Ok(w_int_new(w_dict_len(obj) as i64));
        }
        if pyre_object::is_set_or_frozenset(obj) {
            return Ok(w_int_new(pyre_object::w_set_len(obj) as i64));
        }
        if is_str(obj) {
            return Ok(w_int_new(w_str_len(obj) as i64));
        }
        if pyre_object::bytesobject::is_bytes_like(obj) {
            return Ok(w_int_new(
                pyre_object::bytesobject::bytes_like_len(obj) as i64
            ));
        }
        if pyre_object::is_w_range(obj) {
            // `descr_len → self.w_length`, then `_check_len_result` →
            // `getindex_w(w_int, w_OverflowError)`: `len()` must fit a
            // machine word, so a bignum-length range raises OverflowError.
            return match pyre_object::w_range_length_i64(obj) {
                Some(n) => Ok(w_int_new(n)),
                None => Err(PyError::overflow_error(
                    "cannot fit 'int' into an index-sized integer",
                )),
            };
        }
        if pyre_object::is_long_range_iter(obj) {
            // `functional.py W_LongRangeIterator.descr_len → w_len - w_index`.
            return Ok(pyre_object::range_bigint_to_obj(
                pyre_object::w_long_range_iter_len(obj),
            ));
        }
        if is_range_iter(obj) {
            // `functional.py W_IntRangeIterator.descr_len` reports the
            // stored `remaining` count directly.
            let r = &*(obj as *const pyre_object::functional::W_IntRangeIterator);
            return Ok(w_int_new(r.remaining.max(0)));
        }
        // descroperation.py:294-298 `_len` — `space.lookup(w_obj, '__len__')`
        // then `space.get_and_call_function(w_descr, w_obj)`.  Routed through
        // `r#type` so a true user instance, a W_Root type (e.g. `deque`), and
        // a class whose metaclass defines `__len__` (e.g. `EnumMeta.__len__`)
        // all dispatch correctly.
        if let Some(w_type) = crate::typedef::r#type(obj) {
            if let Some(method) = lookup_in_type_where(w_type, "__len__") {
                return get_and_call_function(method, obj, w_type, &[]);
            }
        }
        // Per-instance __len__ via the unified getattr path (live dict).
        if let Ok(method) = getattr_str(obj, "__len__") {
            return crate::builtins::call_and_check(method, &[obj]);
        }
        Err(PyError::type_error(format!(
            "object of type '{}' has no len()",
            object_functionstr_type_name(obj),
        )))
    }
}

// ── Attribute operations ──────────────────────────────────────────────

// `INSTANCE_DICT` and `WEAKREF_TABLE` live in `objspace/std/mapdict.rs`,
// mirroring PyPy's `MapdictDictSupport` and `MapdictWeakrefSupport`.

/// interpreter/baseobjspace.py:43-44 W_Root.getdict(space).
///
/// ```python
/// def getdict(self, space):
///     return None
/// ```
///
/// objspace/std/mapdict.py:817-818 MapdictDictSupport.getdict overrides
/// it to call `_obj_getdict`. pyre dispatches at runtime via the type's
/// hasdict flag because Rust has no per-class virtual table.
pub fn getdict(obj: PyObjectRef) -> PyObjectRef {
    // exceptions/interp_exceptions.py:222-225 W_BaseException.getdict
    // override — lazily allocates the instance dict on the typed slot.
    if unsafe { pyre_object::is_exception(obj) } {
        return unsafe { pyre_object::interp_exceptions::w_exception_getdict(obj) };
    }
    let w_type = match crate::typedef::r#type(obj) {
        Some(tp) => tp,
        None => return pyre_object::PY_NULL,
    };
    if unsafe { pyre_object::w_type_get_hasdict(w_type) } {
        crate::objspace::std::mapdict::_obj_getdict(obj)
    } else {
        // W_Root.getdict default — return None
        pyre_object::PY_NULL
    }
}

/// `__slots__` storage fallback for a native-layout subclass instance.
///
/// A `W_Member` slot normally reads/writes the receiver's mapdict slot
/// storage (`MapdictSlotsSupport`), which only a `W_ObjectObject` carries.
/// A subclass of a builtin type with a fixed Rust payload (e.g. a subclass
/// of `array.array`) keeps that native layout and has no mapdict, so the
/// slot is instead backed by the instance `__dict__` — the same side table
/// (`mapdict::INSTANCE_DICT`) that already holds the subclass's regular
/// attributes. `None`/`false` means the receiver has no writable dict.
pub(crate) fn native_slot_get(obj: PyObjectRef, name: &str) -> Option<PyObjectRef> {
    let w_dict = getdict(obj);
    if w_dict.is_null() {
        return None;
    }
    unsafe { pyre_object::dictmultiobject::w_dict_getitem_str(w_dict, name) }
}

pub(crate) fn native_slot_set(obj: PyObjectRef, name: &str, value: PyObjectRef) -> bool {
    let w_dict = getdict(obj);
    if w_dict.is_null() {
        return false;
    }
    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str(w_dict, name, value) };
    true
}

pub(crate) fn native_slot_del(obj: PyObjectRef, name: &str) -> bool {
    let w_dict = getdict(obj);
    if w_dict.is_null() {
        return false;
    }
    unsafe { pyre_object::dictmultiobject::w_dict_delitem_str(w_dict, name) }
}

/// interpreter/baseobjspace.py:70-73 W_Root.setdict(space, w_dict).
///
/// ```python
/// def setdict(self, space, w_dict):
///     raise oefmt(space.w_TypeError,
///                  "attribute '__dict__' of %T objects is not writable",
///                  self)
/// ```
///
/// objspace/std/mapdict.py:820-821 MapdictDictSupport.setdict overrides
/// it to call `_obj_setdict`.
pub fn setdict(obj: PyObjectRef, w_dict: PyObjectRef) -> Result<(), PyError> {
    // exceptions/interp_exceptions.py:227-231 W_BaseException.setdict
    // override — validates the value is a dict, then writes the slot.
    // `space.isinstance_w(w_dict, space.w_dict)` accepts dict subclasses.
    if unsafe { pyre_object::is_exception(obj) } {
        let w_dict_type = crate::typedef::gettypeobject(&pyre_object::pyobject::DICT_TYPE);
        if !unsafe { isinstance_w(w_dict, w_dict_type) } {
            return Err(PyError::type_error(
                "setting exceptions's dictionary to a non-dict".to_string(),
            ));
        }
        unsafe { pyre_object::interp_exceptions::w_exception_setdict(obj, w_dict) };
        return Ok(());
    }
    let w_type = match crate::typedef::r#type(obj) {
        Some(tp) => tp,
        None => {
            return Err(PyError::type_error(
                "attribute '__dict__' of object is not writable".to_string(),
            ));
        }
    };
    if unsafe { pyre_object::w_type_get_hasdict(w_type) } {
        crate::objspace::std::mapdict::_obj_setdict(obj, w_dict)
    } else {
        let tp_name = unsafe { pyre_object::w_type_get_name(w_type) };
        Err(PyError::type_error(format!(
            "attribute '__dict__' of '{}' objects is not writable",
            tp_name,
        )))
    }
}

/// `space.finditem_str`-shaped resolution for raw dict ops on a
/// `getdict` result.  Upstream dict-subclass instances are dict-layout
/// `W_DictMultiObject`, so `finditem_str`/`setitem_str` strategy
/// dispatch works on them directly; pyre dict-subclass instances are
/// `__dict_data__`-composed W_ObjectObject (typedef.rs
/// dict_descr_new), so the `w_dict_*` layout accessors must target the
/// backing dict.  Plain dicts, module dicts and mapdict views pass
/// through unchanged.  The `__dict__` getter keeps returning the
/// stored object itself (identity), so it reads `getdict` directly.
fn getdict_backing(obj: PyObjectRef) -> PyObjectRef {
    let w_dict = getdict(obj);
    if w_dict.is_null() {
        return w_dict;
    }
    crate::type_methods::resolve_dict_backing(w_dict)
}

/// interpreter/baseobjspace.py:142-143 W_Root.getweakref().
///
/// ```python
/// def getweakref(self):
///     return None
/// ```
///
/// MapdictWeakrefSupport.getweakref overrides it.
pub fn getweakref(obj: PyObjectRef) -> Option<PyObjectRef> {
    let w_type = crate::typedef::r#type(obj)?;
    if unsafe { pyre_object::w_type_get_weakrefable(w_type) } {
        crate::objspace::std::mapdict::getweakref(obj)
    } else {
        None
    }
}

/// interpreter/baseobjspace.py:145-147 W_Root.setweakref(space, weakreflifeline).
///
/// ```python
/// def setweakref(self, space, weakreflifeline):
///     raise oefmt(space.w_TypeError,
///                  "cannot create weak reference to '%T' object", self)
/// ```
///
/// MapdictWeakrefSupport.setweakref overrides it.
pub fn setweakref(obj: PyObjectRef, weakreflifeline: PyObjectRef) -> Result<(), PyError> {
    let w_type = match crate::typedef::r#type(obj) {
        Some(tp) => tp,
        None => {
            return Err(PyError::type_error(
                "cannot create weak reference to object".to_string(),
            ));
        }
    };
    if unsafe { pyre_object::w_type_get_weakrefable(w_type) } {
        crate::objspace::std::mapdict::setweakref(obj, weakreflifeline);
        Ok(())
    } else {
        let tp_name = unsafe { pyre_object::w_type_get_name(w_type) };
        Err(PyError::type_error(format!(
            "cannot create weak reference to '{}' object",
            tp_name,
        )))
    }
}

/// interpreter/baseobjspace.py:149-150 W_Root.delweakref().
///
/// ```python
/// def delweakref(self):
///     pass
/// ```
pub fn delweakref(obj: PyObjectRef) {
    let w_type = match crate::typedef::r#type(obj) {
        Some(tp) => tp,
        None => return,
    };
    if unsafe { pyre_object::w_type_get_weakrefable(w_type) } {
        crate::objspace::std::mapdict::delweakref(obj);
    }
}

/// `pypy/interpreter/module.py:77 Module.getdict()` parity: return
/// the **canonical** `W_DictObject` already paired with this storage,
/// not a fresh snapshot.  When the storage was first allocated
/// (`w_module_new`, exec/eval anonymous path, etc.) it was bound to
/// a sibling `W_DictObject` via `set_mirror_target` so that
/// storage-side writes back-mirror into that one dict's entries Vec.
/// This lookup retrieves that canonical dict so
/// `function.__globals__`, `globals()`, and the module's own
/// `__dict__` all share **one** identity (`f.__globals__ is
/// m.__dict__` invariant) and the iterating surfaces (`keys`,
/// `values`, `items`, `update`, `copy`, `iter`, `repr`) line up with
/// `lookup` / `len` on the same logical state.
///
/// `type.__dict__` is **not** routed through this helper: PyPy
/// `pypy/objspace/std/typeobject.py:1277 descr_get_dict` returns
/// `W_DictProxyObject(w_dict)` (a read-only live view), not the
/// type's underlying `W_DictObject`.  The dictproxy keeps its own
/// identity per call and forwards reads/iterations to the type's
/// `w_dict`; pyre's type.__dict__ readers stay on that path.
///
/// Lazy-canonical fallback: a storage that has not yet been paired
/// (the `set_mirror_target` call has not happened) gets one allocated
/// here and registered as the `mirror_target`, so subsequent calls
/// return the same object.
pub fn dict_storage_to_dict(ns_ptr: *const crate::DictStorage) -> PyObjectRef {
    dict_storage_to_dict_kind(ns_ptr, DictWrapKind::Module)
}

/// `pypy/objspace/std/dictmultiobject.py:57-89 allocate_and_init_instance`
/// distinguishes `module=True` (W_ModuleDictObject backed by
/// ModuleDictStrategy with version-tag caches), `instance=True`
/// (mapdict.make_instance_dict), and the default branch (regular
/// W_DictObject on EmptyDictStrategy).  Pyre exposes the choice to
/// callers so module globals get the strategy-cache machinery while
/// function locals / type namespaces / generic dicts land on the
/// regular path.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum DictWrapKind {
    /// `dictmultiobject.py:60-69` — Module.__init__ globals path.
    /// Wraps into W_ModuleDictObject with ModuleDictStrategy +
    /// GlobalCache slot map.  Used by `PyFrame.w_globals`,
    /// `function.w_func_globals`, REPL globals, module sys.
    Module,
    /// `dictmultiobject.py:70-89` — instance / default path.  PyPy's
    /// `instance=True` goes through `mapdict.make_instance_dict`
    /// which pyre has not ported; pyre's default (no `module=True`,
    /// no mapdict) lands on a regular W_DictObject with
    /// EmptyDictStrategy.  Used by `type.__dict__`, `frame.f_locals`,
    /// and exec/eval-only locals stores.
    Instance,
}

/// Wrap a `DictStorage` as a Python dict object, classifying the
/// shape per `DictWrapKind`.  Maintains the `mirror_target` invariant
/// — the same storage always returns the same wrapper.
pub fn dict_storage_to_dict_kind(
    ns_ptr: *const crate::DictStorage,
    kind: DictWrapKind,
) -> PyObjectRef {
    if ns_ptr.is_null() {
        return pyre_object::w_dict_new();
    }
    let storage = unsafe { &mut *(ns_ptr as *mut crate::DictStorage) };
    let target = storage.mirror_target();
    if !target.is_null() {
        return target;
    }
    // Lazy canonical: snapshot-populate a fresh wrapper of the
    // requested flavor and register it as the storage's permanent
    // back-mirror target.  The wrapper's `dict_storage_proxy = ns_ptr`
    // keeps forward writes (module.__dict__ / cls.__dict__ /
    // f_locals[k] = ...) in step with the legacy storage that
    // `PyFrame.w_globals` and friends still read through.
    let dict = match kind {
        DictWrapKind::Module => {
            // `pypy/interpreter/module.py:18 Module.__init__` uses
            // `space.newdict(module=True)`; the resulting W_ModuleDictObject
            // carries ModuleDictStrategy + GlobalCache slot map.
            pyre_object::dictmultiobject::w_module_dict_new_with_storage_proxy(ns_ptr as *mut u8)
        }
        DictWrapKind::Instance => {
            // `dictmultiobject.py:81-89` default branch — EmptyDictStrategy
            // regular W_DictObject (PyPy `instance=True`'s mapdict path
            // is a TODO: pyre stops at the regular
            // W_DictObject shape until mapdict is ported).
            pyre_object::dictmultiobject::w_dict_new_with_storage_proxy(ns_ptr as *mut u8)
        }
    };
    unsafe {
        for (key, &value) in storage.entries_wtf8() {
            if value.is_null() {
                continue;
            }
            // Valid names take the str no-proxy setter (Unicode strategy);
            // a lone-surrogate name routes through the WTF-8 setter, which
            // forces the dict onto the surrogate-safe ObjectKey strategy.
            match key.as_str() {
                Ok(s) => pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(dict, s, value),
                Err(_) => {
                    pyre_object::dictmultiobject::w_dict_setitem_wtf8_no_proxy(dict, key, value)
                }
            }
        }
    }
    storage.set_mirror_target(dict);
    // Fresh proxy link: the immortal storage's slots became reachable
    // through a new path; rescan on the next minor collection.
    pyre_object::gc_roots::mark_prebuilt_roots_dirty();
    dict
}

/// Get an attribute from an object: `obj.name`.
///
/// For module objects, looks up the name in the module's namespace dict
/// (PyPy: Module.getdict → w_dict lookup).
/// For other objects, looks up the attribute in the per-object side table.

pub fn getattr_str(obj: PyObjectRef, name: &str) -> PyResult {
    // `space.getattr` — the full path, including the `__getattr__` fallback.
    getattr_str_impl(obj, name, true)
}

/// Shared body of `space.getattr` and the bare `object.__getattribute__` slot.
/// `call_getattr` selects between them: `space.getattr` (`true`) consults
/// `__getattr__` on miss (objspace.py:707 / descroperation.py:242), while the
/// `object.__getattribute__` slot (`false`) raises straight away
/// (descroperation.py:88) — the `__getattr__` fallback is space.getattr's job.
fn getattr_str_impl(obj: PyObjectRef, name: &str, call_getattr: bool) -> PyResult {
    // `pypy/interpreter/baseobjspace.py:1146-1162 getattr`:
    //
    //     def getattr(self, w_obj, w_name):
    //         ...
    //         w_descr = space.lookup(w_obj, '__getattribute__')
    //         try:
    //             return space.get_and_call_function(w_descr, w_obj, w_name)
    //         except ...
    //
    // PyPy never auto-unwraps cells before `getattr`; the user sees the
    // cell type's descriptor namespace (e.g. `cell_contents` from
    // `nestedscope.py:Cell.typedef`).  Pyre previously prepended an
    // `unwrap_cell` here to keep `LOAD_FAST` on a cellvar slot
    // transparent, but the only valid escape of a cell to user-visible
    // code is through `function.__closure__` indexing — where the cell
    // is what the user wants.
    //
    // pypy/module/_weakref/interp__weakref.py:356-394 — proxy_typedef_dict
    // wraps every space op in `force(space, w_obj)`. PyPy then dispatches
    // through the type's `__getattribute__` slot at the C level, so the
    // proxy's wrapper runs before any inline path. pyre's `getattr` does
    // not consult the type's `__getattribute__`, so we apply the same
    // effect by forcing the receiver here. `force()` is a no-op for any
    // non-proxy operand, costing only one ptr-equality check on the hot
    // path.
    let obj = crate::module::_weakref::interp__weakref::force(obj)?;

    // GenericAlias.__getattribute__ (`_pypy_generic_alias.py:52`) — every
    // attribute outside `_ATTR_EXCEPTIONS` delegates to `__origin__`.
    // pyre's `getattr` does not dispatch through a typedef
    // `__getattribute__` for builtin W_Roots, so the delegation is wired
    // here; the exception names fall through to the normal lookup that
    // serves the `__origin__`/`__args__`/`__parameters__` getsets.
    if unsafe { pyre_object::is_generic_alias(obj) }
        && !crate::_pypy_generic_alias::is_attr_exception(name)
    {
        let origin = unsafe { pyre_object::w_generic_alias_get_origin(obj) };
        return getattr_str(origin, name);
    }

    // super proxy — PyPy: pypy/module/__builtin__/descriptor.py W_Super.getattribute
    // Looks up `name` in cls's MRO starting AFTER super_type.
    unsafe {
        if pyre_object::descriptor::is_super(obj) {
            let super_type = pyre_object::descriptor::w_super_get_type(obj);
            let bound_obj = pyre_object::descriptor::w_super_get_obj(obj);

            // Walk obj's type MRO, skip until we pass super_type.
            // Fall back to `crate::typedef::r#type(obj)` so non-INSTANCE
            // built-in subclasses (W_BaseException, etc.) resolve their
            // class through the same path that powers `type(obj)` —
            // `pypy/objspace/std/typeobject.py:1083 type_get_mro`.
            // descriptor.py:127-149 _super_check: `su_obj` is itself a
            // subtype of `su_type` only in the classmethod / class-level
            // case (return `su_obj`).  A class whose metaclass is
            // `su_type` is an *instance* of `su_type`, not a subtype, so
            // it must resolve through `type(su_obj)` — otherwise its MRO
            // never reaches `su_type` and the lookup fails.
            let w_obj_type = if is_type(bound_obj) && issubtype_w(bound_obj, super_type) {
                bound_obj
            } else if is_instance(bound_obj) {
                w_instance_get_type(bound_obj)
            } else if let Some(cls) = crate::typedef::r#type(bound_obj) {
                cls
            } else {
                return Err(PyError::type_error("super: bad obj type"));
            };
            let mro_ptr = w_type_get_mro(w_obj_type);
            if !mro_ptr.is_null() {
                let mro = &*mro_ptr;
                let mut past_super = false;
                for &t in mro {
                    if std::ptr::eq(t, super_type) {
                        past_super = true;
                        continue;
                    }
                    if !past_super {
                        continue;
                    }
                    if is_type(t) {
                        // Look in this class's own dict only (not its MRO),
                        // since we are already iterating the full MRO ourselves.
                        let ns_ptr = w_type_get_dict_ptr(t) as *mut crate::DictStorage;
                        let found = if !ns_ptr.is_null() {
                            (*ns_ptr).get(name).copied()
                        } else {
                            None
                        };
                        if let Some(attr) = found {
                            // descriptor.py W_Super.getattribute:
                            // Invoke descriptor __get__ protocol.
                            // classmethod.__get__(obj, type) binds the class
                            // (`w_obj_type`); staticmethod.__get__ unwraps to
                            // the plain function; a plain function binds `obj`.
                            // `__new__` is implicitly static — never bind.
                            if pyre_object::is_classmethod(attr) {
                                let func = pyre_object::w_classmethod_get_func(attr);
                                return Ok(pyre_object::w_method_new(func, w_obj_type, w_obj_type));
                            }
                            if pyre_object::is_staticmethod(attr) {
                                return Ok(pyre_object::w_staticmethod_get_func(attr));
                            }
                            if name != "__new__" && crate::is_function(attr) {
                                return Ok(pyre_object::w_method_new(attr, bound_obj, w_obj_type));
                            }
                            return Ok(attr);
                        }
                    }
                }
            }
            return Err(PyError::new(
                PyErrorKind::AttributeError,
                format!("'super' object has no attribute '{name}'"),
            ));
        }
    }

    // Generator/coroutine methods — PyPy: generator.py GeneratorIterator
    //
    // Return W_Method(func, gen) so the generator is passed as args[0].
    unsafe {
        if pyre_object::generator::is_generator(obj) {
            let (sname, func, arity): (&str, fn(&[PyObjectRef]) -> PyResult, Option<u16>) =
                match name {
                    "send" => ("send", generator_send_method, Some(2)),
                    "throw" => ("throw", generator_throw_method, None),
                    "close" => ("close", generator_close_method, Some(1)),
                    "__next__" => ("__next__", generator_next_method, Some(1)),
                    "__iter__" => ("__iter__", iter_self_method, Some(1)),
                    _ => ("", generator_next_method, None), // sentinel — won't match
                };
            if !sname.is_empty() {
                let func_obj = if let Some(a) = arity {
                    crate::make_builtin_function_with_arity(sname, func, a)
                } else {
                    crate::make_builtin_function(sname, func)
                };
                return Ok(pyre_object::w_method_new(
                    func_obj,
                    obj,
                    pyre_object::PY_NULL,
                ));
            }
        }
    }

    // itertools.count / itertools.repeat methods — PyPy interp_itertools.py
    // Expose `__next__` and `__iter__` so `_count(1).__next__` and
    // `iter(counter)` work.
    unsafe {
        if pyre_object::interp_itertools::is_count(obj)
            || pyre_object::interp_itertools::is_repeat(obj)
            || pyre_object::interp_itertools::is_takewhile(obj)
            || pyre_object::interp_itertools::is_dropwhile(obj)
            || pyre_object::interp_itertools::is_filterfalse(obj)
            || pyre_object::interp_itertools::is_pairwise(obj)
            || pyre_object::interp_itertools::is_cycle(obj)
        {
            let entry: Option<(fn(&[PyObjectRef]) -> PyResult, &str, u16)> = match name {
                "__next__" => Some((iter_next_method, "__next__", 1)),
                "__iter__" => Some((iter_self_method, "__iter__", 1)),
                // takewhile/dropwhile expose `__reduce__` + `__setstate__`,
                // filterfalse `__reduce__` only (interp_itertools.py
                // W_TakeWhile/W_DropWhile/W_FilterFalse typedefs); pairwise
                // exposes neither.  cycle exposes both (W_Cycle typedef).
                // count/repeat expose `__reduce__` only (W_Count.reduce_w /
                // W_Repeat.descr_reduce — no `__setstate__`).
                "__reduce__" if pyre_object::interp_itertools::is_count(obj) => {
                    Some((count_reduce_method, "__reduce__", 1))
                }
                "__reduce__" if pyre_object::interp_itertools::is_repeat(obj) => {
                    Some((repeat_reduce_method, "__reduce__", 1))
                }
                "__reduce__" if pyre_object::interp_itertools::is_cycle(obj) => {
                    Some((cycle_reduce_method, "__reduce__", 1))
                }
                "__setstate__" if pyre_object::interp_itertools::is_cycle(obj) => {
                    Some((cycle_setstate_method, "__setstate__", 2))
                }
                "__reduce__" if pyre_object::interp_itertools::is_takewhile(obj) => {
                    Some((takewhile_reduce_method, "__reduce__", 1))
                }
                "__setstate__" if pyre_object::interp_itertools::is_takewhile(obj) => {
                    Some((takewhile_setstate_method, "__setstate__", 2))
                }
                "__reduce__" if pyre_object::interp_itertools::is_dropwhile(obj) => {
                    Some((dropwhile_reduce_method, "__reduce__", 1))
                }
                "__setstate__" if pyre_object::interp_itertools::is_dropwhile(obj) => {
                    Some((dropwhile_setstate_method, "__setstate__", 2))
                }
                "__reduce__" if pyre_object::interp_itertools::is_filterfalse(obj) => {
                    Some((filterfalse_reduce_method, "__reduce__", 1))
                }
                _ => None,
            };
            if let Some((func, sname, arity)) = entry {
                let func_obj = crate::make_builtin_function_with_arity(sname, func, arity);
                return Ok(pyre_object::w_method_new(
                    func_obj,
                    obj,
                    pyre_object::PY_NULL,
                ));
            }
        }
    }

    // range attributes/methods — functional.py W_Range.
    // `.start`/`.stop`/`.step` read-only ints; count/index/__iter__/
    // __reversed__ exposed as bound methods.
    unsafe {
        if pyre_object::is_w_range(obj) {
            match name {
                "start" | "stop" | "step" => {
                    let (start, stop, step) = pyre_object::w_range_fields(obj);
                    return Ok(match name {
                        "start" => start,
                        "stop" => stop,
                        _ => step,
                    });
                }
                _ => {}
            }
            let entry: Option<(fn(&[PyObjectRef]) -> PyResult, &str, u16)> = match name {
                "count" => Some((range_count_method, "count", 2)),
                "index" => Some((range_index_method, "index", 2)),
                "__iter__" => Some((range_iter_method, "__iter__", 1)),
                "__reversed__" => Some((range_reversed_method, "__reversed__", 1)),
                "__reduce__" => Some((range_reduce_method, "__reduce__", 1)),
                "__hash__" => Some((range_hash_method, "__hash__", 1)),
                _ => None,
            };
            if let Some((func, sname, arity)) = entry {
                let func_obj = crate::make_builtin_function_with_arity(sname, func, arity);
                return Ok(pyre_object::w_method_new(
                    func_obj,
                    obj,
                    pyre_object::PY_NULL,
                ));
            }
        }
    }

    // Native iterator methods — `iter(x)` products: list/tuple/str/set/
    // bytes/zip/map/reversed share the seq-iter type; range, the dict
    // views and enumerate are distinct.  `next(it)` and `for` already
    // drive these through the iternext slot; expose `__next__` and
    // `__iter__` so explicit `it.__next__()` / `it.__iter__()` work too.
    unsafe {
        if is_seq_iter(obj)
            || is_range_iter(obj)
            || pyre_object::is_long_range_iter(obj)
            || pyre_object::dictmultiobject::is_dict_view_iterator(obj)
            || pyre_object::functional::is_enumerate(obj)
            || pyre_object::functional::is_reversed(obj)
            || pyre_object::functional::is_filter(obj)
            || pyre_object::functional::is_map(obj)
            || pyre_object::functional::is_zip(obj)
            || pyre_object::operation::is_callable_iterator(obj)
        {
            let entry: Option<(fn(&[PyObjectRef]) -> PyResult, &str)> = match name {
                "__next__" => Some((iter_next_method, "__next__")),
                "__iter__" => Some((iter_self_method, "__iter__")),
                _ => None,
            };
            if let Some((func, sname)) = entry {
                let func_obj = crate::make_builtin_function_with_arity(sname, func, 1);
                return Ok(pyre_object::w_method_new(
                    func_obj,
                    obj,
                    pyre_object::PY_NULL,
                ));
            }
            // Per-iterator-type pickle protocol: `__reduce__` /
            // `__setstate__` / `__length_hint__` recreate the iterator's
            // CPython 3.14 pickle shape.  `arity` includes `self`.
            let entry: Option<(fn(&[PyObjectRef]) -> PyResult, &str, u16)> = if is_seq_iter(obj) {
                match name {
                    "__reduce__" => Some((seq_iter_reduce_method, "__reduce__", 1)),
                    "__setstate__" => Some((seq_iter_setstate_method, "__setstate__", 2)),
                    "__length_hint__" => Some((seq_iter_length_hint_method, "__length_hint__", 1)),
                    _ => None,
                }
            } else if is_range_iter(obj) {
                match name {
                    "__reduce__" => Some((range_iter_reduce_method, "__reduce__", 1)),
                    "__length_hint__" => {
                        Some((range_iter_length_hint_method, "__length_hint__", 1))
                    }
                    _ => None,
                }
            } else if pyre_object::is_long_range_iter(obj) {
                match name {
                    "__reduce__" => Some((long_range_iter_reduce_method, "__reduce__", 1)),
                    "__length_hint__" => {
                        Some((long_range_iter_length_hint_method, "__length_hint__", 1))
                    }
                    _ => None,
                }
            } else if pyre_object::dictmultiobject::is_dict_view_iterator(obj) {
                match name {
                    "__reduce__" => Some((dict_view_iter_reduce_method, "__reduce__", 1)),
                    "__length_hint__" => {
                        Some((dict_view_iter_length_hint_method, "__length_hint__", 1))
                    }
                    _ => None,
                }
            } else if pyre_object::functional::is_enumerate(obj) {
                match name {
                    "__reduce__" => Some((enumerate_reduce_method, "__reduce__", 1)),
                    _ => None,
                }
            } else if pyre_object::functional::is_reversed(obj) {
                match name {
                    "__reduce__" => Some((reversed_reduce_method, "__reduce__", 1)),
                    "__setstate__" => Some((reversed_setstate_method, "__setstate__", 2)),
                    "__length_hint__" => Some((reversed_length_hint_method, "__length_hint__", 1)),
                    _ => None,
                }
            } else if pyre_object::functional::is_filter(obj) {
                match name {
                    "__reduce__" => Some((filter_reduce_method, "__reduce__", 1)),
                    _ => None,
                }
            } else if pyre_object::functional::is_map(obj) {
                match name {
                    "__reduce__" => Some((map_reduce_method, "__reduce__", 1)),
                    "__setstate__" => Some((map_setstate_method, "__setstate__", 2)),
                    _ => None,
                }
            } else if pyre_object::functional::is_zip(obj) {
                match name {
                    "__reduce__" => Some((zip_reduce_method, "__reduce__", 1)),
                    "__setstate__" => Some((zip_setstate_method, "__setstate__", 2)),
                    _ => None,
                }
            } else {
                None
            };
            if let Some((func, sname, arity)) = entry {
                let func_obj = crate::make_builtin_function_with_arity(sname, func, arity);
                return Ok(pyre_object::w_method_new(
                    func_obj,
                    obj,
                    pyre_object::PY_NULL,
                ));
            }
        }
    }

    // Property descriptor methods — PyPy: descriptor.py W_Property.setter / getter / deleter
    // Returns a bound method (W_Method) that captures the property via w_self,
    // so the static handler can extract the property from args[0].
    unsafe {
        if is_property(obj) {
            let static_name: Option<(
                &'static str,
                fn(&[PyObjectRef]) -> Result<PyObjectRef, crate::PyError>,
                u16,
            )> = match name {
                "setter" => Some(("setter", property_setter_impl, 2)),
                "getter" => Some(("getter", property_getter_impl, 2)),
                "deleter" => Some(("deleter", property_deleter_impl, 2)),
                "__set_name__" => Some(("__set_name__", property_set_name_impl, 3)),
                _ => None,
            };
            if let Some((sname, func, arity)) = static_name {
                let builtin = crate::make_builtin_function_with_arity(sname, func, arity);
                return Ok(pyre_object::function::w_method_new(
                    builtin,
                    obj,
                    pyre_object::PY_NULL,
                ));
            }
            match name {
                "fget" => return Ok(w_property_get_fget(obj)),
                "fset" => return Ok(w_property_get_fset(obj)),
                "fdel" => return Ok(w_property_get_fdel(obj)),
                "__name__" => {
                    // descriptor.py exposes the name set by `__set_name__`;
                    // an unset name falls through to the normal
                    // `'property' object has no attribute '__name__'`.
                    let w_name = pyre_object::descriptor::w_property_get_name(obj);
                    if !w_name.is_null() {
                        return Ok(w_name);
                    }
                }
                "__doc__" => {
                    // descriptor.py:316-318 `__doc__ = GetSetProperty(
                    // W_Property.get_doc, W_Property.set_doc)` → :249-250
                    // get_doc returns the `w_doc` slot.
                    let stored = pyre_object::descriptor::w_property_get_doc(obj);
                    return Ok(if stored.is_null() { w_none() } else { stored });
                }
                _ => {}
            }
        }
    }

    // Member descriptor attributes — typedef.py:443 Member.__name__, __objclass__
    unsafe {
        if pyre_object::typedef::is_member(obj) {
            match name {
                "__name__" => {
                    return Ok(pyre_object::w_str_new(pyre_object::w_member_get_name(obj)));
                }
                "__objclass__" => return Ok(pyre_object::w_member_get_cls(obj)),
                _ => {}
            }
        }
    }

    // Module objects: look up in module namespace.
    // PyPy `space.getattr(w_module, w_name) → Module.getdictvalue(space,
    // name)` (`pypy/interpreter/module.py:Module.getdictvalue`
    // inherited from `baseobjspace.py:45-48 W_Root.getdictvalue`):
    //
    //     w_dict = self.getdict(space)        # module.py:77 → self.w_dict
    //     if w_dict is not None:
    //         return space.finditem_str(w_dict, attr)
    //     return None
    //
    // Routing through `space.finditem_str` (rather than reading the
    // backing storage directly) gives dict subclass `__getitem__`
    // overrides their PyPy chance to fire on the user-supplied
    // `__builtins__` aliasing case (`moduledef.py:102-103
    // Module(space, None, w_builtin)`), and routes through the
    // storage-authoritative read path so transient W_DictObject
    // snapshots can't shadow the live storage state.  The Result-
    // bearing variant propagates non-KeyError errors from subclass
    // overrides (`baseobjspace.py:870 finditem` re-raise).
    unsafe {
        if is_module(obj) {
            if name == "__dict__" {
                // module.py:20 — `Module.getdict(space)` returns
                // `self.w_dict`.  Always non-null after construction.
                return Ok(pyre_object::w_module_get_w_dict(obj));
            }
            let w_dict = pyre_object::w_module_get_w_dict(obj);
            if !w_dict.is_null() {
                if let Some(value) = finditem_str(w_dict, name)? {
                    if !value.is_null() {
                        return Ok(value);
                    }
                }
            }
        }
    }

    // Instance objects — PyPy: descroperation.py descr__getattribute__
    //
    // Full descriptor protocol (PEP 252):
    //   1. Look up name in type MRO → w_descr
    //   2. If w_descr is a data descriptor (__get__ + __set__/__delete__):
    //      → call w_descr.__get__(obj, type)
    //   3. Check instance dict
    //   4. If w_descr is a non-data descriptor (__get__ only):
    //      → call w_descr.__get__(obj, type)
    //   5. Return w_descr as-is
    unsafe {
        // `pypy/interpreter/typedef.py:825-826 Method.typedef` exposes
        // `__func__` / `__self__` as `interp_attrproperty_w` getset
        // descriptors that resolve to the wrapped function / instance
        // directly on attribute access.  Pyre's method typedef
        // registers them as regular `make_builtin_function` entries
        // which the descriptor protocol below would surface as bound
        // methods (binding the `__func__` helper to the method
        // instance), breaking `m.__func__ is C.m` and `m.__self__ is
        // c` identity.  Short-circuit before the `is_instance` branch
        // so the type dispatch path matches PyPy's getset semantics.
        // PyPy3 exposes only the dunder names — `im_func` / `im_self`
        // were dropped in 3.x, so do not surface them here.
        if pyre_object::function::is_method(obj) {
            match name {
                "__func__" => {
                    return Ok(pyre_object::function::w_method_get_func(obj));
                }
                "__self__" => {
                    return Ok(pyre_object::function::w_method_get_self(obj));
                }
                // `__class__` resolves to the `method` type itself (handled by
                // the generic type dispatch below), never forwarded.
                "__class__" => {}
                _ => {
                    // `classobject.c method_getattro` — attributes defined on
                    // the method type win (`__call__` / `__repr__` / `__eq__`
                    // / `__hash__`); any other name is forwarded to `__func__`
                    // (`__name__` / `__qualname__` / `__code__` / `__doc__` /
                    // `__defaults__` / `__annotations__` / …).
                    let on_method_type = crate::typedef::r#type(obj)
                        .map(|t| lookup_in_type_where(t, name).is_some())
                        .unwrap_or(false);
                    if !on_method_type {
                        let func = pyre_object::function::w_method_get_func(obj);
                        if !func.is_null() {
                            return getattr_str(func, name);
                        }
                    }
                }
            }
        }
        if is_instance(obj) {
            let w_type = w_instance_get_type(obj);

            // `pypy/objspace/descroperation.py descr__getattribute__`
            // dispatches through the receiver type's `__getattribute__`
            // slot before running the default descriptor protocol
            // (objspace.py:663-666).  Users routinely override this to
            // customise *all* attribute access (e.g. lazy proxies,
            // validating wrappers).  `getattribute_if_not_from_object`
            // returns a non-default override or `None`, memoizing the
            // default in `uses_object_getattribute`.
            //
            // descroperation.py:234-245 `_handle_getattribute`: the custom
            // slot runs alone, and an AttributeError it raises falls back to
            // `__getattr__` (objspace.py:691-699 / 707-710).
            // descroperation.py:87 — `space.getattr(obj, "__getattribute__")`
            // still routes through the type's custom `__getattribute__`, so the
            // slot dispatch runs for every name, including "__getattribute__".
            if let Some(slot) = getattribute_if_not_from_object(w_type) {
                let name_obj = w_str_new(name);
                // objspace.py:666 / descroperation.py:238
                // `space.get_and_call_function(w_descr, w_obj, w_name)` —
                // bind the `__getattribute__` slot through `__get__`.
                match get_and_call_function(slot, obj, w_type, &[name_obj]) {
                    Ok(v) => return Ok(v),
                    Err(e) if e.kind == PyErrorKind::AttributeError => {
                        return instance_getattr_hook_or_err(w_type, obj, name, e);
                    }
                    Err(e) => return Err(e),
                }
            }

            // Step 1: look up in type MRO
            let w_descr = lookup_in_type_where(w_type, name);

            // Step 2: data descriptor takes priority over instance dict.
            // objspace.py:694-699 — a descriptor `__get__` raising
            // AttributeError falls back to `__getattr__`, not propagated.
            if let Some(descr) = w_descr {
                if is_data_descr(descr) {
                    match get(descr, obj, w_type) {
                        Ok(Some(result)) => return Ok(result),
                        Ok(None) => {}
                        Err(e) if e.kind == PyErrorKind::AttributeError => {
                            return instance_getattr_hook_or_err(w_type, obj, name, e);
                        }
                        Err(e) => return Err(e),
                    }
                }
            }

            // Step 3: instance dict — the mapdict map+storage is the sole
            // authority for instance attributes.  Read the node directly
            // (getdictvalue, mapdict.py:846-847) rather than materialising the
            // MapDictStrategy `__dict__` view through `getdict_backing`;
            // MapDictStrategy.getitem_str (mapdict.py:1168-1175) delegates to the
            // same `instance_node_getdictvalue`, so the value is identical and the
            // `__dict__` wrapper is built only on explicit `__dict__` access.
            let value = unsafe {
                crate::objspace::std::mapdict::instance_node_getdictvalue(obj, Wtf8::new(name))
            };
            if let Some(value) = value {
                return Ok(value);
            }

            // Step 4: non-data descriptor
            // PyPy: descroperation.py — invoke __get__ to bind methods.
            // objspace.py:694-699 — a non-data descriptor `__get__` raising
            // AttributeError falls back to `__getattr__` too.
            if let Some(descr) = w_descr {
                match get(descr, obj, w_type) {
                    Ok(Some(result)) => return Ok(result),
                    Ok(None) => {}
                    Err(e) if e.kind == PyErrorKind::AttributeError => {
                        return instance_getattr_hook_or_err(w_type, obj, name, e);
                    }
                    Err(e) => return Err(e),
                }
                // Step 5: builtin methods found in base type MRO need binding
                // CPython: PyFunction_GET_CODE slot → bound method
                if crate::is_function(descr)
                    && !crate::is_builtin_code(
                        crate::function_get_code(descr) as pyre_object::PyObjectRef
                    )
                {
                    return Ok(pyre_object::w_method_new(descr, obj, w_type));
                }
                return Ok(descr);
            }

            // Special attributes — PyPy: descroperation.py
            if name == "__class__" {
                return Ok(w_type);
            }

            // descroperation.py:243-252 `_handle_getattribute`: on the terminal
            // miss, `__getattr__` is the last resort.  Used by every wrapper
            // class that delegates attribute lookup to a backing stream/buffer
            // (unittest._WritelnDecorator, pathlib, etc.).
            return instance_getattr_hook_or_err(
                w_type,
                obj,
                name,
                PyError::attribute_error_with_context(
                    format!(
                        "'{}' object has no attribute '{name}'",
                        w_type_get_name(w_type)
                    ),
                    obj,
                    name,
                ),
            );
        }
    }

    let result = object_getattr_miss(obj, name, call_getattr);
    // module.py:130-142 `Module.descr_getattribute` — PEP 562: after the
    // normal lookup misses with AttributeError, a module-level `__getattr__`
    // stored in the module's own dict gets the final say, called with just the
    // attribute name.  Only `space.getattr` consults it.
    if let Err(ref e) = result {
        if call_getattr && e.kind == PyErrorKind::AttributeError && unsafe { is_module(obj) } {
            let w_dict = unsafe { pyre_object::w_module_get_w_dict(obj) };
            if !w_dict.is_null() {
                if let Some(mod_getattr) = finditem_str(w_dict, "__getattr__")? {
                    if !mod_getattr.is_null() {
                        let name_obj = w_str_new(name);
                        return crate::call::call_function_impl_result(mod_getattr, &[name_obj]);
                    }
                }
                // No module `__getattr__`: phrase the miss with the module's
                // `__name__` (`module '<name>' has no attribute '<attr>'`, the
                // `'%U'` form), which requires a str `__name__` and falls back
                // to the bare form otherwise.  (The `__spec__`-based
                // circular-import diagnostics are not ported.)
                let msg = match finditem_str(w_dict, "__name__")? {
                    Some(w) if !w.is_null() && unsafe { pyre_object::is_str(w) } => {
                        let nm = unsafe { pyre_object::w_str_get_wtf8(w) };
                        format!("module '{nm}' has no attribute '{name}'")
                    }
                    _ => format!("module has no attribute '{name}'"),
                };
                return Err(PyError::new(PyErrorKind::AttributeError, msg));
            }
        }
    }
    result
}

// ─── `w_name`-taking attribute API ───
//
// `descroperation.py:225/247/255 getattr/setattr/delattr(space, w_obj,
// w_name)` take the attribute name as a wrapped str object and only
// extract the WTF-8 bytes at the `object.__*__` boundary
// (`get_attribute_name → space.text_w(w_name)`, descroperation.py:69-75).
// `text_w` returns raw WTF-8 and never raises on a lone surrogate
// (`unicodeobject.py:133`), so a surrogate attribute name flows through.
//
// A lone-surrogate name can never equal any type/builtin attribute (all
// valid identifiers), so it is dispatched to the generic instance /
// module `__dict__` lookup only — the strict subset of `getattr_str`/
// `setattr_str`/`delattr_str` that a non-identifier name can reach.  A
// valid UTF-8 name takes the `&str` fast path unchanged.

/// `space.getattr(w_obj, w_name)`.
pub fn getattr(obj: PyObjectRef, w_name: PyObjectRef) -> PyResult {
    let name = unsafe { pyre_object::w_str_get_wtf8(w_name) };
    match name.as_str() {
        Ok(s) => getattr_str(obj, s),
        Err(_) => unsafe { getattr_surrogate(obj, w_name, name) },
    }
}

/// `space.setattr(w_obj, w_name, w_val)`.
pub fn setattr(obj: PyObjectRef, w_name: PyObjectRef, value: PyObjectRef) -> PyResult {
    let name = unsafe { pyre_object::w_str_get_wtf8(w_name) };
    match name.as_str() {
        Ok(s) => setattr_str(obj, s, value),
        Err(_) => unsafe { setattr_surrogate(obj, w_name, name, value) },
    }
}

/// `space.delattr(w_obj, w_name)`.
pub fn delattr(obj: PyObjectRef, w_name: PyObjectRef) -> PyResult {
    let name = unsafe { pyre_object::w_str_get_wtf8(w_name) };
    match name.as_str() {
        Ok(s) => delattr_str(obj, s),
        Err(_) => unsafe { delattr_surrogate(obj, w_name, name) },
    }
}

/// `space.getattr` for a lone-surrogate name — mirrors `getattr_str`'s
/// generic-instance tail: terminal `__dict__` read, then the
/// `__getattr__` hook on miss.  (A surrogate cannot match any of
/// `getattr_str`'s builtin-type special-cases, all valid identifiers.)
unsafe fn getattr_surrogate(obj: PyObjectRef, w_name: PyObjectRef, name: &Wtf8) -> PyResult {
    unsafe {
        match object_getattribute_surrogate(obj, w_name, name) {
            Ok(v) => Ok(v),
            Err(e) => {
                // descroperation.py:243-252 `_handle_getattribute`: only an
                // AttributeError from `__getattribute__` (here a descriptor
                // `__get__` or the dict miss) triggers the `__getattr__`
                // fallback; any other exception propagates unchanged.
                if e.kind != crate::PyErrorKind::AttributeError {
                    return Err(e);
                }
                // module.py:139-142 PEP 562: a module-level `__getattr__` in the
                // module's own dict is consulted on miss, called unbound with
                // just the name (a module hook is a dict value, not a type
                // descriptor).
                if is_module(obj) {
                    let w_dict = pyre_object::w_module_get_w_dict(obj);
                    if !w_dict.is_null() {
                        if let Some(mod_getattr) = finditem_str(w_dict, "__getattr__")? {
                            if !mod_getattr.is_null() {
                                return crate::call::call_function_impl_result(
                                    mod_getattr,
                                    &[w_name],
                                );
                            }
                        }
                    }
                    return Err(e);
                }
                // `space.lookup(w_obj, '__getattr__')` walks `type(w_obj)` —
                // the metaclass for a type receiver, the class for an
                // instance — so the hook is found for any object, not just
                // instances.  objspace.py:710 `get_and_call_function` binds the
                // hook through `__get__` before calling it, so a
                // staticmethod / classmethod / custom-descriptor `__getattr__`
                // is handled.  The hook's result or its own exception is final.
                let w_objtype = crate::typedef::r#type(obj).unwrap_or(std::ptr::null_mut());
                if !w_objtype.is_null() {
                    if let Some(getattr_fn) = lookup_in_type_where(w_objtype, "__getattr__") {
                        return get_and_call_function(getattr_fn, obj, w_objtype, &[w_name]);
                    }
                }
                Err(e)
            }
        }
    }
}

/// `object.__getattribute__` terminal for a lone-surrogate name —
/// generic module / instance `__dict__` read, AttributeError on miss.
/// `w_name` is passed straight to the `ObjectKey`-keyed dict ops
/// (already WTF-8 safe).
pub(crate) unsafe fn object_getattribute_surrogate(
    obj: PyObjectRef,
    w_name: PyObjectRef,
    name: &Wtf8,
) -> PyResult {
    unsafe {
        if is_module(obj) {
            let w_dict = pyre_object::w_module_get_w_dict(obj);
            if !w_dict.is_null() {
                if let Some(v) = pyre_object::w_dict_lookup(w_dict, w_name) {
                    if !v.is_null() {
                        return Ok(v);
                    }
                }
            }
            return Err(attr_error_wtf8(obj, name));
        }
        if is_type(obj) {
            // typeobject.py:811-828 W_TypeObject.descr_getattribute. A
            // surrogate name can reach a metatype descriptor via
            // `setattr(type(cls), '\udc80', descr)`, so the full protocol
            // applies: a metatype data descriptor wins first, then the type's
            // own MRO value bound through `__get__(None, type)`, then a
            // metatype non-data descriptor.
            let metatype = crate::typedef::r#type(obj).unwrap_or(std::ptr::null_mut());
            let w_descr = if metatype.is_null() {
                None
            } else {
                lookup_in_type_wtf8(metatype, name)
            };
            // typeobject.py:814-819: metatype data descriptor, bound as
            // `__get__(self, type(self))`.
            if let Some(descr) = w_descr {
                if is_data_descr(descr) {
                    if let Some(result) = get(descr, obj, metatype)? {
                        return Ok(result);
                    }
                }
            }
            // typeobject.py:820-823: the type's own MRO value, bound through
            // `space.get(w_value, space.w_None, self)` = `__get__(None, type)`.
            if let Some(w_value) = lookup_in_type_wtf8(obj, name) {
                if let Some(result) = get(w_value, w_none(), obj)? {
                    return Ok(result);
                }
                return Ok(w_value);
            }
            // typeobject.py:824-825: a metatype non-data descriptor, bound as
            // `space.get(w_descr, self)`.
            if let Some(descr) = w_descr {
                if let Some(result) = get(descr, obj, metatype)? {
                    return Ok(result);
                }
                return Ok(descr);
            }
            return Err(attr_error_wtf8(obj, name));
        }
        // Instance / general object: the full descriptor protocol keyed
        // through the WTF-8 MRO view, mirroring object_getattribute.  A
        // surrogate name can legitimately reach a descriptor via
        // `setattr(cls, '\udc80', descr)`, so a data descriptor's
        // `__get__` takes priority over the instance dict, and a non-data
        // descriptor binds after it (descroperation.py:88-112).
        let w_type = crate::typedef::r#type(obj).unwrap_or(std::ptr::null_mut());
        let w_descr = if w_type.is_null() {
            None
        } else {
            lookup_in_type_wtf8(w_type, name)
        };
        if let Some(descr) = w_descr {
            if is_data_descr(descr) {
                if let Some(result) = get(descr, obj, w_type)? {
                    return Ok(result);
                }
            }
        }
        let w_dict = getdict_backing(obj);
        if !w_dict.is_null() {
            if let Some(v) = pyre_object::w_dict_lookup(w_dict, w_name) {
                if !v.is_null() {
                    return Ok(v);
                }
            }
        }
        if let Some(descr) = w_descr {
            if let Some(result) = get(descr, obj, w_type)? {
                return Ok(result);
            }
            if crate::is_function(descr)
                && !crate::is_builtin_code(
                    crate::function_get_code(descr) as pyre_object::PyObjectRef
                )
            {
                return Ok(pyre_object::w_method_new(descr, obj, w_type));
            }
            return Ok(descr);
        }
        Err(attr_error_wtf8(obj, name))
    }
}

/// `space.setattr` for a lone-surrogate name — mirrors `setattr_str`:
/// dispatch through the type's `__setattr__` (the slot is WTF-8 safe),
/// else the terminal store.
unsafe fn setattr_surrogate(
    obj: PyObjectRef,
    w_name: PyObjectRef,
    name: &Wtf8,
    value: PyObjectRef,
) -> PyResult {
    let value = unwrap_cell(value);
    let obj = crate::module::_weakref::interp__weakref::force(obj)?;
    unsafe {
        if is_instance(obj) {
            let w_type = w_instance_get_type(obj);
            if let Some(sa) = lookup_in_type(w_type, "__setattr__") {
                return crate::call::call_function_impl_result(sa, &[obj, w_name, value])
                    .map(|_| w_none());
            }
        }
    }
    unsafe { object_setattr_surrogate(obj, w_name, name, value) }
}

/// `object.__setattr__` terminal for a lone-surrogate name — mirrors
/// `object_setattr`: a data descriptor's `__set__` takes priority, then
/// the module / type / instance `__dict__` store.  A surrogate name can
/// legitimately reach a data descriptor (`setattr(cls, '\udc80', descr)`),
/// so the descriptor walk runs the same as for an identifier name, just
/// keyed through the WTF-8 MRO view.
pub(crate) unsafe fn object_setattr_surrogate(
    obj: PyObjectRef,
    w_name: PyObjectRef,
    name: &Wtf8,
    value: PyObjectRef,
) -> PyResult {
    let value = unwrap_cell(value);
    let obj = crate::module::_weakref::interp__weakref::force(obj)?;
    unsafe {
        // descroperation.py:114-123 — a data descriptor's `__set__` takes
        // priority over the dict store.  Walk `space.type(obj)` (the
        // metaclass for a type receiver, mirroring object_setattr:4483-4493)
        // comparing WTF-8 keys so a surrogate-named descriptor is found.
        let w_type = if is_instance(obj) {
            w_instance_get_type(obj)
        } else {
            crate::typedef::r#type(obj).unwrap_or(std::ptr::null_mut())
        };
        if !w_type.is_null() {
            if let Some(descr) = lookup_in_type_wtf8(w_type, name) {
                if set(descr, obj, value)? {
                    return Ok(w_none());
                }
                // descroperation.py:124-126 — `__delete__` but no `__set__`
                // is a read-only data descriptor; reject rather than shadow
                // it with the namespace store.
                if descr_has_delete(descr) {
                    return Err(descr_not_settable_error(descr));
                }
            }
        }
        // Type objects: store into the type's own WTF-8 keyed namespace
        // and reset the method caches — `typeobject.py type.__setattr__`
        // → `w_type.dict_w[name] = w_value; self.mutated(name)`.  A lone
        // surrogate has no `&str` form, so the precise key is unavailable
        // and `mutated` falls back to the conservative whole-cache reset
        // (correct: a surrogate can never name `__eq__`/`__hash__`).
        if is_type(obj) {
            // typeobject.py:416 — only heap types may have their dict mutated.
            if !pyre_object::w_type_is_heaptype(obj) {
                return Err(PyError::type_error(format!(
                    "cannot set {} attribute of immutable type '{}'",
                    crate::display::format_wtf8_repr(name),
                    w_type_get_name(obj)
                )));
            }
            let dict_ptr = w_type_get_dict_ptr(obj) as *mut crate::DictStorage;
            if !dict_ptr.is_null() {
                crate::dict_storage_store_wtf8(&mut *dict_ptr, name, value);
                mutated(obj, name.as_str().ok());
                return Ok(w_none());
            }
        }
        // Module namespace store at the `setdictvalue` position — after the
        // data-descriptor walk so a surrogate-named data descriptor on the
        // module's type fires first (`descr__setattr__` order).
        if is_module(obj) {
            let w_dict = pyre_object::w_module_get_w_dict(obj);
            if !w_dict.is_null() {
                setitem(w_dict, w_name, value)?;
                return Ok(w_none());
            }
        }
        let w_dict = getdict(obj);
        if !w_dict.is_null() {
            setitem(w_dict, w_name, value)?;
            return Ok(w_none());
        }
        Err(attr_error_wtf8(obj, name))
    }
}

/// `space.delattr` for a lone-surrogate name — mirrors `delattr_str`.
unsafe fn delattr_surrogate(obj: PyObjectRef, w_name: PyObjectRef, name: &Wtf8) -> PyResult {
    let obj = crate::module::_weakref::interp__weakref::force(obj)?;
    unsafe {
        if is_instance(obj) {
            let w_type = w_instance_get_type(obj);
            if let Some(da) = lookup_in_type(w_type, "__delattr__") {
                return crate::call::call_function_impl_result(da, &[obj, w_name])
                    .map(|_| w_none());
            }
        }
    }
    unsafe { object_delattr_surrogate(obj, w_name, name) }
}

/// `object.__delattr__` terminal for a lone-surrogate name.
pub(crate) unsafe fn object_delattr_surrogate(
    obj: PyObjectRef,
    w_name: PyObjectRef,
    name: &Wtf8,
) -> PyResult {
    let obj = crate::module::_weakref::interp__weakref::force(obj)?;
    unsafe {
        // descroperation.py:131-137 — a data descriptor's `__delete__`
        // takes priority over the dict removal.  Mirror object_delattr,
        // comparing WTF-8 keys so a surrogate-named descriptor is found,
        // and run before the module/type/instance dict removal so a
        // surrogate-named descriptor on the module's/metaclass's type is
        // not shadowed by the namespace delete.
        let w_type = if is_instance(obj) {
            w_instance_get_type(obj)
        } else {
            crate::typedef::r#type(obj).unwrap_or(std::ptr::null_mut())
        };
        if !w_type.is_null() {
            if let Some(descr) = lookup_in_type_wtf8(w_type, name) {
                if is_data_descr(descr) {
                    delete(descr, obj)?;
                    return Ok(w_none());
                }
            }
        }
        if is_module(obj) {
            let w_dict = pyre_object::w_module_get_w_dict(obj);
            if !w_dict.is_null() && pyre_object::w_dict_delitem(w_dict, w_name) {
                return Ok(w_none());
            }
            return Err(attr_error_wtf8(obj, name));
        }
        if is_type(obj) {
            // typeobject.py:437 — only heap types may have attributes deleted.
            if !pyre_object::w_type_is_heaptype(obj) {
                return Err(PyError::type_error(format!(
                    "cannot delete attributes on immutable type object '{}'",
                    w_type_get_name(obj)
                )));
            }
            let dict_ptr = w_type_get_dict_ptr(obj) as *mut crate::DictStorage;
            if !dict_ptr.is_null() && crate::dict_storage_delete_wtf8(&mut *dict_ptr, name) {
                // A lone surrogate has no `&str` form, so `mutated` falls
                // back to the conservative whole-cache reset (correct: a
                // surrogate can never name `__eq__`/`__hash__`).
                mutated(obj, name.as_str().ok());
                return Ok(w_none());
            }
            return Err(attr_error_wtf8(obj, name));
        }
        let w_dict = getdict_backing(obj);
        if !w_dict.is_null() && pyre_object::w_dict_delitem(w_dict, w_name) {
            return Ok(w_none());
        }
        Err(attr_error_wtf8(obj, name))
    }
}

/// `raiseattrerror` for a lone-surrogate name.  descroperation.py:58-64
/// renders the name with `%R` (its repr), so a lone surrogate prints as
/// `\udcXX` rather than a lossy replacement char.  The repr already
/// supplies the surrounding quotes (`format_wtf8_repr`), matching the
/// `%R` substitution in `"'%T' object has no attribute %R"`.
fn attr_error_wtf8(obj: PyObjectRef, name: &Wtf8) -> PyError {
    let tp_name = unsafe {
        match crate::typedef::r#type(obj) {
            Some(tp) => pyre_object::w_type_get_name(tp).to_string(),
            None => (*(*obj).ob_type).name.to_string(),
        }
    };
    let name_repr = crate::display::format_wtf8_repr(name);
    let mut err = PyError::new(
        PyErrorKind::AttributeError,
        format!("'{tp_name}' object has no attribute {name_repr}"),
    );
    err.w_name_context = pyre_object::w_str_from_wtf8(name.to_wtf8_buf());
    err.w_obj_context = obj;
    err
}

/// `object.__getattribute__` terminal — the default descriptor protocol
/// without the user `__getattribute__` override check.
pub fn object_getattribute(obj: PyObjectRef, name: &str) -> PyResult {
    let obj = unwrap_cell(obj);
    unsafe {
        if is_instance(obj) {
            let w_type = w_instance_get_type(obj);
            let w_descr = lookup_in_type_where(w_type, name);
            if let Some(descr) = w_descr {
                if is_data_descr(descr) {
                    if let Some(result) = get(descr, obj, w_type)? {
                        return Ok(result);
                    }
                }
            }
            // Instance dict is the sole authority for instance attributes:
            // read the mapdict node directly (getdictvalue, mapdict.py:846-847)
            // rather than materialising the MapDictStrategy `__dict__` view, which
            // MapDictStrategy.getitem_str (mapdict.py:1168-1175) delegates to
            // anyway.  No side-table fallback.
            let value =
                crate::objspace::std::mapdict::instance_node_getdictvalue(obj, Wtf8::new(name));
            if let Some(value) = value {
                return Ok(value);
            }
            if let Some(descr) = w_descr {
                if let Some(result) = get(descr, obj, w_type)? {
                    return Ok(result);
                }
                if crate::is_function(descr)
                    && !crate::is_builtin_code(
                        crate::function_get_code(descr) as pyre_object::PyObjectRef
                    )
                {
                    return Ok(pyre_object::w_method_new(descr, obj, w_type));
                }
                return Ok(descr);
            }
            if name == "__class__" {
                return Ok(w_type);
            }
            // descroperation.py:88 — object.__getattribute__ raises
            // AttributeError on miss. __getattr__ is space.getattr's job.
            return Err(PyError::attribute_error_with_context(
                format!(
                    "'{}' object has no attribute '{name}'",
                    w_type_get_name(w_type),
                ),
                obj,
                name,
            ));
        }
    }
    // Non-instance receiver (module, type, builtin object): the pure descriptor
    // protocol with no `__getattr__` fallback — that belongs to space.getattr,
    // not the bare object.__getattribute__ slot (descroperation.py:88).
    getattr_str_impl(obj, name, false)
}

/// `descroperation.py:242-245` `_handle_getattribute` tail: on an
/// AttributeError from the descriptor protocol (a custom `__getattribute__`,
/// a descriptor `__get__`, or the terminal miss), look up `__getattr__` on the
/// receiver type and call it; its result — or its own exception — is final.
/// Returns the original error when there is no `__getattr__`.
unsafe fn instance_getattr_hook_or_err(
    w_type: PyObjectRef,
    obj: PyObjectRef,
    name: &str,
    e: crate::PyError,
) -> PyResult {
    unsafe {
        if let Some(getattr_fn) = lookup_in_type_where(w_type, "__getattr__") {
            let name_obj = w_str_new(name);
            // objspace.py:710 `space.get_and_call_function(w_descr, w_obj,
            // w_name)` — bind `__getattr__` through `__get__` so a
            // staticmethod / classmethod / custom-descriptor hook receives the
            // arguments PyPy gives it.
            return get_and_call_function(getattr_fn, obj, w_type, &[name_obj]);
        }
    }
    Err(e)
}

/// `_handle_getattribute` fallback for a type receiver: an AttributeError
/// raised anywhere in `type.__getattribute__` — a metatype data descriptor's
/// `__get__`, a class MRO descriptor's `__get__`, a metatype non-data
/// descriptor's `__get__`, or the terminal miss — consults `__getattr__` on the
/// metaclass (descroperation.py:234-245).  `call_getattr` gates the fallback
/// off for the bare `object.__getattribute__` slot, which propagates instead.
unsafe fn type_getattr_hook_or_err(
    obj: PyObjectRef,
    w_metaclasses: &[Option<PyObjectRef>; 2],
    name: &str,
    e: crate::PyError,
    call_getattr: bool,
) -> PyResult {
    if call_getattr && e.kind == PyErrorKind::AttributeError {
        for w_metaclass in w_metaclasses.iter().flatten() {
            let w_metaclass = *w_metaclass;
            if unsafe { is_type(w_metaclass) } {
                if let Some(getattr_fn) =
                    unsafe { lookup_in_type_where(w_metaclass, "__getattr__") }
                {
                    let name_obj = w_str_new(name);
                    return unsafe {
                        get_and_call_function(getattr_fn, obj, w_metaclass, &[name_obj])
                    };
                }
            }
        }
    }
    Err(e)
}

fn object_getattr_miss(obj: PyObjectRef, name: &str, call_getattr: bool) -> PyResult {
    // Type objects: look up in type's own dict → base dicts
    // PyPy: typeobject.py lookup_where → MRO search + descriptor unwrap
    unsafe {
        if is_type(obj) {
            // baseobjspace.py:76 — the metaclass is type(C), read from w_class.
            let w_type_type = crate::typedef::w_type();
            let w_object = crate::typedef::w_object();
            let w_metaclass = {
                let w_class = (*obj).w_class;
                if !w_class.is_null() && !std::ptr::eq(w_class, w_type_type) {
                    Some(w_class)
                } else {
                    None
                }
            };
            let w_metaclasses: [Option<PyObjectRef>; 2] =
                [w_metaclass, crate::typedef::gettypefor((*obj).ob_type)];
            // typeobject.py:811-819 — a metatype DATA descriptor is consulted
            // before anything else, including the hardcoded type attributes
            // below.  Only honor one defined on a user metaclass (its owner is
            // neither `type` nor `object`): the builtin getsets those bases
            // carry are served by the dedicated short-circuits below, so
            // letting them through here would re-enter this lookup.
            for w_metaclass in w_metaclasses.iter().flatten() {
                let w_metaclass = *w_metaclass;
                if is_type(w_metaclass) {
                    if let Some((src, descr)) = lookup_where(w_metaclass, name) {
                        if !std::ptr::eq(src, w_type_type)
                            && !std::ptr::eq(src, w_object)
                            && is_data_descr(descr)
                        {
                            match get(descr, obj, w_metaclass) {
                                Ok(Some(result)) => return Ok(result),
                                Ok(None) => {}
                                Err(e) => {
                                    return type_getattr_hook_or_err(
                                        obj,
                                        &w_metaclasses,
                                        name,
                                        e,
                                        call_getattr,
                                    );
                                }
                            }
                        }
                    }
                }
            }

            // Special type attributes — PyPy: typeobject.py
            if name == "__class__" {
                // `pypy/objspace/std/typeobject.py:198 type___class__getter`
                // returns `self.w_metaclass` (the metaclass).  pyre stamps
                // each registered builtin type's `w_class` to the
                // `type` typeobject in `init_typeobjects`'s post-loop
                // (typedef.rs:489-499).  Return that directly; falling
                // through to `lookup_in_type_where` would hit the
                // `__class__` getset descriptor on the metatype and
                // recurse.  When `w_class` is null (bootstrap or a
                // type built before `init_typeobjects`), fall back to
                // the `type` typeobject so `int.__class__ is type`
                // still holds.
                let mc = (*obj).w_class;
                if !mc.is_null() {
                    return Ok(mc);
                }
                let w_type_type = crate::typedef::w_type();
                if !w_type_type.is_null() {
                    return Ok(w_type_type);
                }
            }
            if name == "__name__" {
                // `type.__name__` is the bare type name; a dotted tp_name
                // (e.g. "types.UnionType") carries its module prefix only
                // in repr, so strip to the final component here.
                let full = w_type_get_name(obj);
                let bare = full.rsplit('.').next().unwrap_or(full);
                return Ok(w_str_new(bare));
            }
            if name == "__qualname__" {
                // Check if __qualname__ was explicitly set in class body
                if let Some(qn) = lookup_in_type_where(obj, "__qualname__") {
                    return Ok(qn);
                }
                // A static type's qualname is its bare name; a dotted tp_name
                // (e.g. "re.Pattern") carries its module prefix only in repr,
                // so strip to the final component here, matching __name__.
                let full = w_type_get_name(obj);
                let bare = full.rsplit('.').next().unwrap_or(full);
                return Ok(w_str_new(bare));
            }
            if name == "__mro__" {
                let mro_ptr = w_type_get_mro(obj);
                if !mro_ptr.is_null() {
                    return Ok(w_tuple_new((*mro_ptr).clone()));
                }
            }
            if name == "__flags__" {
                // typeobject.py:1237 descr__flags — the `tp_flags` bitmask.
                // A getset on `type`, so a metaclass (a `type` subclass) carries
                // it in its own MRO; without this short-circuit the type's-own-MRO
                // path below would bind it with `obj=None` and yield the raw
                // descriptor instead of the bitmask.
                return Ok(w_int_new(w_type_get_flags(obj)));
            }
            if name == "__dict__" {
                // `pypy/objspace/std/typeobject.py:1277 descr_get_dict`
                // returns `W_DictProxyObject(w_dict)` — a read-only
                // **live** view of the type's namespace.  The proxy's
                // identity is fresh per call (a new wrapper) but its
                // `w_mapping` is the type's canonical W_DictObject, so
                // a subsequent `cls.x = 1; d['x']` resolves through the
                // dict_storage_proxy and surfaces the live binding.
                let dict_ptr = w_type_get_dict_ptr(obj) as *const crate::DictStorage;
                if dict_ptr.is_null() {
                    return Ok(pyre_object::w_dict_proxy_new(pyre_object::w_dict_new()));
                }
                // `pypy/objspace/std/typeobject.py:1277 descr_get_dict`
                // wraps the type's regular W_DictObject — not a
                // module-strategy dict — into the proxy.  Pass
                // `Instance` kind so the type's namespace lives on
                // the EmptyDictStrategy/typed-strategy ladder rather
                // than ModuleDictStrategy's GlobalCache machinery.
                let canonical = dict_storage_to_dict_kind(dict_ptr, DictWrapKind::Instance);
                return Ok(pyre_object::w_dict_proxy_new(canonical));
            }
            if name == "__bases__" {
                // typeobject.py:1027 descr_get__bases__ — `object` (the root
                // type) carries no bases tuple; surface the empty tuple rather
                // than the null sentinel so `reversed(cls.__bases__)` and the
                // C3 helpers in `functools` don't dereference null.
                let bases = w_type_get_bases(obj);
                if bases.is_null() {
                    return Ok(w_tuple_new(vec![]));
                }
                return Ok(bases);
            }
            // PEP 649 lazy annotations: when `cls.__annotations__` is
            // requested and only `__annotate_func__` (or `__annotate__`)
            // is set, call the annotate function with format=1 to
            // produce the actual dict.  CPython 3.14+ stops emitting
            // `__annotations__` directly in class bodies in favour of
            // this lazy form.
            if name == "__annotations__" {
                if let Some(v) = lookup_in_type_where(obj, "__annotations__") {
                    return Ok(v);
                }
                if let Some(annotate_fn) = lookup_in_type_where(obj, "__annotate_func__")
                    .or_else(|| lookup_in_type_where(obj, "__annotate__"))
                {
                    if !annotate_fn.is_null() && !is_none(annotate_fn) {
                        // format=1 (VALUE) — return runtime values.
                        return Ok(crate::call_function(annotate_fn, &[w_int_new(1)]));
                    }
                }
                return Ok(pyre_object::w_dict_new());
            }
            // PEP 649: `__annotate__` and `__annotate_func__` are the
            // same slot. Bytecode stores it as `__annotate_func__` in the
            // class dict; user code reads it as `__annotate__`. Forward
            // either name to the other, matching CPython's mapping in
            // typeobject.c type_get___annotate__.
            if name == "__annotate__" || name == "__annotate_func__" {
                if let Some(v) = lookup_in_type_where(obj, name) {
                    return Ok(v);
                }
                let alt = if name == "__annotate__" {
                    "__annotate_func__"
                } else {
                    "__annotate__"
                };
                if let Some(v) = lookup_in_type_where(obj, alt) {
                    return Ok(v);
                }
                return Ok(w_none());
            }
            // `__abstractmethods__` is a descriptor on `type` that raises
            // AttributeError when the slot is not populated, NOT a getter
            // that returns None. abc.update_abstractmethods relies on
            // hasattr() returning False to short-circuit non-ABCs.
            if name == "__abstractmethods__" {
                if let Some(v) = lookup_in_type_where(obj, name) {
                    return Ok(v);
                }
                // descroperation.py:234 wraps the whole getattribute slot, so
                // even this hardcoded AttributeError consults the metaclass
                // `__getattr__` before propagating.
                return type_getattr_hook_or_err(
                    obj,
                    &w_metaclasses,
                    name,
                    PyError::new(
                        PyErrorKind::AttributeError,
                        format!(
                            "type object '{}' has no attribute '__abstractmethods__'",
                            w_type_get_name(obj),
                        ),
                    ),
                    call_getattr,
                );
            }
            if name == "__doc__"
                || name == "__code__"
                || name == "__func__"
                || name == "__self__"
                || name == "__globals__"
                || name == "__closure__"
                || name == "__defaults__"
                || name == "__kwdefaults__"
            {
                // Check class dict first, then return None.  `__wrapped__` is
                // excluded so an unset value raises AttributeError rather than
                // feeding `inspect.unwrap` a bogus None chain.
                if let Some(v) = lookup_in_type_where(obj, name) {
                    return Ok(v);
                }
                return Ok(w_none());
            }
            // `__module__` is NOT in the short-circuit list — it falls
            // through to the normal descriptor protocol so type's
            // `__module__` GetSetProperty (`typedef.rs init_type_type`)
            // can resolve via PyPy's `typeobject.py:614-624 get_module`
            // (heaptype reads class dict, builtin types use the dot-
            // split of the class name with `"builtins"` fallback).

            // typeobject.py:820-823 — the type's own MRO value, bound via
            // `space.get(w_value, space.w_None, self)` = `__get__(None, type)`
            // (functions stay unbound, classmethod binds the class, a custom
            // descriptor sees `obj is None`).  The receiver must be the `None`
            // singleton, not a null pointer, or a Python
            // `__get__(self, obj, objtype=None)` loses its `obj` argument.
            if let Some(value) = lookup_in_type_where(obj, name) {
                match get(value, w_none(), obj) {
                    Ok(Some(result)) => return Ok(result),
                    Ok(None) => return Ok(value),
                    Err(e) => {
                        return type_getattr_hook_or_err(
                            obj,
                            &w_metaclasses,
                            name,
                            e,
                            call_getattr,
                        );
                    }
                }
            }
            // typeobject.py:824-825 — a metatype non-data descriptor, bound as
            // `space.get(w_descr, self)`.  Binding is handled by load_method.
            for w_metaclass in w_metaclasses.iter().flatten() {
                let w_metaclass = *w_metaclass;
                if is_type(w_metaclass) {
                    if let Some(value) = lookup_in_type_where(w_metaclass, name) {
                        match get(value, obj, w_metaclass) {
                            Ok(Some(result)) => return Ok(result),
                            Ok(None) => return Ok(value),
                            Err(e) => {
                                return type_getattr_hook_or_err(
                                    obj,
                                    &w_metaclasses,
                                    name,
                                    e,
                                    call_getattr,
                                );
                            }
                        }
                    }
                }
            }
            // descroperation.py:243-252 _handle_getattribute: on the terminal
            // miss, consult `__getattr__` on the metaclass (gated off for the
            // bare object.__getattribute__ slot by `call_getattr`); otherwise
            // raise. The terminal AttributeError carries the obj/name context.
            return type_getattr_hook_or_err(
                obj,
                &w_metaclasses,
                name,
                PyError::attribute_error_with_context(
                    format!(
                        "type object '{}' has no attribute '{name}'",
                        w_type_get_name(obj)
                    ),
                    obj,
                    name,
                ),
                call_getattr,
            );
        }
    }

    // `Objects/methodobject.c meth_reduce` — a module-level builtin
    // pickles by reference: `__reduce__` / `__reduce_ex__` return the
    // `__qualname__` string, which `pickle.Pickler.save` routes to
    // `save_global`.  pyre's builtins carry no bound `__self__`, so the
    // bare-qualname branch always applies.  This must precede the generic
    // MRO lookup below, which would otherwise bind `object.__reduce__`.
    if (name == "__reduce__" || name == "__reduce_ex__")
        && unsafe { pyre_object::py_type_check(obj, &crate::function::BUILTIN_FUNCTION_TYPE) }
    {
        let reduce_fn: fn(&[PyObjectRef]) -> PyResult = |args| {
            Ok(w_str_new(&unsafe {
                crate::function::function_get_qualname(args[0])
            }))
        };
        let (sname, arity): (&'static str, u16) = if name == "__reduce_ex__" {
            ("__reduce_ex__", 2)
        } else {
            ("__reduce__", 1)
        };
        let func_obj = crate::make_builtin_function_with_arity(sname, reduce_fn, arity);
        return Ok(pyre_object::w_method_new(
            func_obj,
            obj,
            pyre_object::PY_NULL,
        ));
    }

    // Builtin type method lookup via TypeDef registry.
    //
    // PyPy: space.type(w_obj) → W_TypeObject → MRO lookup in type dict.
    // Each builtin type (list, str, dict, etc.) has a W_TypeObject with
    // methods pre-installed, matching PyPy's TypeDef interpleveldefs.
    if let Some(w_type) = crate::typedef::r#type(obj) {
        if let Some(method) = unsafe { lookup_in_type_where(w_type, name) } {
            if unsafe { crate::is_function(method) } {
                return Ok(pyre_object::w_method_new(method, obj, w_type));
            }
            if let Some(result) = unsafe { get(method, obj, w_type)? } {
                return Ok(result);
            }
            return Ok(method);
        }
    }

    // Function object attributes — PyPy: function.py Function
    // Check the live W_DictObject (functions are hasdict per typedef.py:735
    // __dict__ = getset_func_dict).
    if unsafe { crate::is_function(obj) } {
        let w_dict = getdict_backing(obj);
        if !w_dict.is_null() {
            if let Some(v) = unsafe { pyre_object::w_dict_getitem_str(w_dict, name) } {
                return Ok(v);
            }
        }
    }
    unsafe {
        if crate::is_function(obj) {
            match name {
                "__code__" => {
                    // function_get_code returns Code-level pointer (PyCode or BuiltinCode)
                    let code = crate::function_get_code(obj) as PyObjectRef;
                    if code.is_null() {
                        return Ok(w_none());
                    }
                    return Ok(code);
                }
                "__name__" => {
                    return Ok(w_str_new(crate::function_get_name(obj)));
                }
                "__closure__" => {
                    let closure = crate::function_get_closure(obj);
                    return Ok(if closure.is_null() { w_none() } else { closure });
                }
                "__globals__" => {
                    // `funcobject.py:325 fget_func_globals` returns
                    // `self.w_func_globals` directly — the function's
                    // `w_func_globals_obj` field, the canonical W_DictObject
                    // shared with the defining module's `__dict__`.
                    return Ok(unsafe { crate::function_get_globals_obj(obj) });
                }
                "__defaults__" => {
                    let defaults = crate::function_get_defaults(obj);
                    return Ok(if defaults.is_null() {
                        w_none()
                    } else {
                        defaults
                    });
                }
                "__kwdefaults__" => {
                    let kwdefaults = crate::function_get_kwdefaults(obj);
                    return Ok(if kwdefaults.is_null() {
                        w_none()
                    } else {
                        kwdefaults
                    });
                }
                "__qualname__" => {
                    // function.py:470-471 fget_func_qualname returns
                    // space.newtext(self.qualname); the typed
                    // `function_get_qualname` mirrors PyPy's `qualname or
                    // self.name` short-circuit (w_qualname slot →
                    // code.co_qualname → name).
                    let s = crate::function::function_get_qualname(obj);
                    return Ok(w_str_new(&s));
                }
                "__doc__" => {
                    // `pypy/interpreter/function.py:395-398 fget_func_doc`
                    // — instance dict override first, then lazy
                    // `code.getdocstring(space)`.  Pyre's
                    // `function_get_doc` mirrors that shape (instance
                    // dict → BuiltinCode `docstring` → user
                    // CodeObject `HAS_DOCSTRING` first const).  The
                    // generic `__doc__` fallback would otherwise
                    // return None for every user-defined function
                    // because no caller routes to `function_get_doc`.
                    return Ok(crate::function::function_get_doc(obj));
                }
                "__module__" => {
                    // `pypy/interpreter/function.py:507 fget___module__`
                    // lazy-resolves from `w_func_globals['__name__']` on
                    // first read and caches into `self.w_module`.  Pyre's
                    // `crate::function::fget___module__` mirrors that
                    // shape — `(*func).w_module` stamps on first call so
                    // subsequent reads (and explicit `setattr(f,
                    // '__module__', x)`) take the cache path.  The
                    // generic `__module__` fallback at the end of
                    // `getattr` would otherwise return `None` for every
                    // function (function.rs:48 init `w_module = PY_NULL`).
                    return Ok(unsafe { crate::function::fget___module__(obj) });
                }
                "__annotations__" => {
                    // `pypy/interpreter/function.py:548-551
                    // fget_func_annotations` returns
                    // `self.w_ann`, allocating an empty dict on first
                    // access if none was set, and stamping it back so
                    // identity holds.
                    //
                    // Pyre stores the eager dict on the typed
                    // `Function.w_ann` slot via
                    // `function_set_annotations` at MAKE_FUNCTION
                    // ANNOTATIONS time (eval.rs); PEP 649 lazy
                    // annotations keep the `__annotate__` callable on
                    // the typed `Function.w_annotate` slot.  The
                    // helper resolves both forms and stamps `w_ann`
                    // so `f.__annotations__ is f.__annotations__`
                    // identity holds across reads.
                    return Ok(unsafe { crate::function::function_get_annotations(obj) });
                }
                "__annotate__" => {
                    // PEP 649 `func_annotate` surface: the stored
                    // callable, or None when annotations were eager or
                    // absent.
                    let annotate_fn =
                        unsafe { (*(obj as *mut crate::function::Function)).w_annotate };
                    if !annotate_fn.is_null() {
                        return Ok(annotate_fn);
                    }
                    return Ok(w_none());
                }
                _ => {}
            }
        }
        // PyPy parity: `__func__` / `__wrapped__` for staticmethod and
        // classmethod are bound through their typedef descriptors
        // (`typedef.py:870-871, 884-885 interp_attrproperty_w(
        // 'w_function')`); pyre registers the same descriptors in
        // `init_staticmethod_type` / `init_classmethod_type`, so the
        // generic type-dict fallback below reaches them.  The hardcoded
        // arm previously here predated the descriptor registration.
        if crate::pycode::is_code(obj) {
            let code_ptr = crate::pycode::w_code_get_ptr(obj) as *const crate::CodeObject;
            if code_ptr.is_null() {
                return Ok(w_none());
            }
            let code = &*code_ptr;
            match name {
                "co_varnames" => {
                    let items = code
                        .varnames
                        .iter()
                        .map(|item| w_str_new(item.as_ref()))
                        .collect();
                    return Ok(w_tuple_new(items));
                }
                // `pycode.py:335-336 fget_co_cellvars`:
                //     return space.newtuple([space.newtext(name)
                //                            for name in self.co_cellvars])
                "co_cellvars" => {
                    let items = code
                        .cellvars
                        .iter()
                        .map(|item| w_str_new(item.as_ref()))
                        .collect();
                    return Ok(w_tuple_new(items));
                }
                // `pycode.py:338-339 fget_co_freevars`:
                //     return space.newtuple([space.newtext(name)
                //                            for name in self.co_freevars])
                "co_freevars" => {
                    let items = code
                        .freevars
                        .iter()
                        .map(|item| w_str_new(item.as_ref()))
                        .collect();
                    return Ok(w_tuple_new(items));
                }
                "co_argcount" => return Ok(w_int_new(code.arg_count as i64)),
                "co_kwonlyargcount" => return Ok(w_int_new(code.kwonlyarg_count as i64)),
                "co_name" => return Ok(w_str_new(code.obj_name.as_ref())),
                // `pycode.py` `co_qualname` (3.11+) — the dotted qualified
                // name the compiler stamped (`<module>.Class.method`).
                "co_qualname" => return Ok(w_str_new(code.qualname.as_ref())),
                "co_filename" => return Ok(w_str_new(code.source_path.as_ref())),
                "co_flags" => return Ok(w_int_new(code.flags.bits() as i64)),
                // `pypy/interpreter/pycode.py:143` — `self.co_firstlineno = firstlineno`,
                // `typedef.py:718` — `co_firstlineno = interp_attrproperty('co_firstlineno', cls=PyCode, wrapfn="newint")`.
                // RustPython exposes the field as `Option<OneIndexed>`; map None to 1
                // (matching CPython's default for module-level code).
                "co_firstlineno" => {
                    return Ok(w_int_new(
                        code.first_line_number.map_or(1, |n| n.get() as i64),
                    ));
                }
                _ => {}
            }
        }
    }

    // Common special attributes — return defaults for any object type.
    // `__wrapped__` is deliberately excluded: it must raise AttributeError
    // when unset (functools.wraps stores it in the instance dict, and
    // staticmethod/classmethod expose it via their type's GetSetProperty),
    // otherwise `inspect.unwrap` sees an endless None chain and reports a
    // bogus "wrapper loop".
    if name == "__doc__" || name == "__module__" || name == "__annotations__" {
        // baseobjspace.py:46-50 W_Root.getdictvalue — consult the
        // instance dict (exception `w_dict` slot, hasdict objects).
        let w_dict = getdict_backing(obj);
        if !w_dict.is_null() {
            if let Some(value) = unsafe { pyre_object::w_dict_getitem_str(w_dict, name) } {
                return Ok(value);
            }
        }
        // `__module__` is not a universal attribute: an object whose
        // type-MRO carries no `__module__` (e.g. a builtin instance like
        // `(0).__module__`) raises AttributeError rather than reporting
        // None.  `__doc__`/`__annotations__` keep the None default.
        if name != "__module__" {
            return Ok(w_none());
        }
    }
    // Exception attributes — PyPy: W_BaseException attributes
    if unsafe { pyre_object::is_exception(obj) } {
        match name {
            "__traceback__" => {
                // `interp_exceptions.py:196-201 W_BaseException.descr_gettraceback`
                // returns the `w_traceback` slot stamped by
                // `descr_settraceback` and the `raise` machinery's
                // `record_application_traceback`; `None` when none has
                // been set.
                let stored =
                    unsafe { pyre_object::interp_exceptions::w_exception_get_traceback(obj) };
                return Ok(if stored.is_null() { w_none() } else { stored });
            }
            "__cause__" => {
                // `interp_exceptions.py:163-164 descr_getcause`.
                let stored = unsafe { pyre_object::interp_exceptions::w_exception_get_cause(obj) };
                return Ok(if stored.is_null() { w_none() } else { stored });
            }
            "__context__" => {
                // `interp_exceptions.py:180-181 descr_getcontext`.
                let stored =
                    unsafe { pyre_object::interp_exceptions::w_exception_get_context(obj) };
                return Ok(if stored.is_null() { w_none() } else { stored });
            }
            "__suppress_context__" => {
                // `interp_exceptions.py:212-213 descr_getsuppresscontext`
                // returns `space.newbool(self.suppress_context)`.
                // Defaults to False per `:117 W_BaseException` class
                // default; `descr_setcause` flips to True.
                let b = unsafe {
                    pyre_object::interp_exceptions::w_exception_get_suppress_context(obj)
                };
                return Ok(pyre_object::w_bool_from(b));
            }
            "args" => {
                // `pypy/module/exceptions/interp_exceptions.py:153
                // W_BaseException.descr_getargs` returns
                // `space.newtuple(self.args_w)` — a freshly-built
                // tuple per call.  `w_exception_get_args` does the
                // same: it walks the internal list slot and rebuilds
                // a `W_TupleObject`, returning the empty tuple when
                // the slot was never stamped.
                return Ok(unsafe { pyre_object::interp_exceptions::w_exception_get_args(obj) });
            }
            "value" => {
                // `pypy/module/exceptions/interp_exceptions.py
                // W_StopIteration.descr_init` stores `value = w_args[0]`,
                // exposed as `fget_value`.  `generator_send_ex` stamps
                // the generator's return value into the exception's
                // `args` tuple; mirror PyPy by returning `args[0]` and
                // defaulting to `None`.  Only StopIteration uses this
                // attribute — other exception kinds keep the regular
                // attribute lookup fall-through.
                let kind = unsafe { pyre_object::w_exception_get_kind(obj) };
                if kind == pyre_object::interp_exceptions::ExcKind::StopIteration {
                    let args_tuple =
                        unsafe { pyre_object::interp_exceptions::w_exception_get_args(obj) };
                    // `w_exception_get_args` always returns a real
                    // tuple — empty tuple when `args_w` was never
                    // stamped — so the null-check above is unneeded.
                    let len = unsafe { pyre_object::w_tuple_len(args_tuple) };
                    if len > 0 {
                        if let Some(v) = unsafe { pyre_object::w_tuple_getitem(args_tuple, 0) } {
                            return Ok(v);
                        }
                    }
                    return Ok(w_none());
                }
            }
            "code" => {
                // `interp_exceptions.py:986-1006 W_SystemExit`: `code` is a
                // writable `readwrite_attrproperty_w('w_code')` slot
                // (`:1006`) set by `descr_init` to `args_w[0]` for a single
                // argument, `newtuple(args_w)` for several, and the
                // `__init__` default `None` when the instance carries no
                // arguments.  Read the slot first so an explicit
                // `e.code = x` write persists, then derive from `args_w`
                // (the internal-constructor path that bypasses the public
                // setter), mirroring the OSError `errno` arm.
                let kind = unsafe { pyre_object::w_exception_get_kind(obj) };
                if kind == pyre_object::interp_exceptions::ExcKind::SystemExit {
                    let stored =
                        unsafe { pyre_object::interp_exceptions::w_exception_get_code(obj) };
                    if !stored.is_null() {
                        return Ok(stored);
                    }
                    let args = unsafe { pyre_object::interp_exceptions::w_exception_get_args(obj) };
                    let len = unsafe { pyre_object::w_tuple_len(args) };
                    if len == 1 {
                        if let Some(v) = unsafe { pyre_object::w_tuple_getitem(args, 0) } {
                            return Ok(v);
                        }
                    } else if len > 1 {
                        return Ok(args);
                    }
                    return Ok(w_none());
                }
            }
            // `interp_exceptions.py:739-742 W_OSError` exposes
            // `errno` / `strerror` / `filename` / `filename2` as
            // `readwrite_attrproperty_w('w_errno', ...)` slots, populated
            // by the 2..=5-argument constructor (`errno = args[0]`,
            // `strerror = args[1]`, `filename = args[2]`,
            // `filename2 = args[4]`).  Read the writable slot first so a
            // `e.errno = ...` assignment (`object_setattr`) persists; when
            // the slot is `PY_NULL` (the internal-constructor path that
            // never goes through the public setter) fall back to deriving
            // the value from `args_w` with the same argument-count gate.
            // Fewer than two arguments leaves all four `None` (the class
            // defaults).
            "errno" | "strerror" => {
                let kind = unsafe { pyre_object::w_exception_get_kind(obj) };
                if matches!(
                    kind,
                    pyre_object::interp_exceptions::ExcKind::OSError
                        | pyre_object::interp_exceptions::ExcKind::FileNotFoundError
                ) {
                    let stored = if name == "errno" {
                        unsafe { pyre_object::interp_exceptions::w_exception_get_errno(obj) }
                    } else {
                        unsafe { pyre_object::interp_exceptions::w_exception_get_strerror(obj) }
                    };
                    if !stored.is_null() {
                        return Ok(stored);
                    }
                    let args = unsafe { pyre_object::interp_exceptions::w_exception_get_args(obj) };
                    let n = unsafe { pyre_object::w_tuple_len(args) };
                    if (2..=5).contains(&n) {
                        let idx = if name == "errno" { 0 } else { 1 };
                        if let Some(v) = unsafe { pyre_object::w_tuple_getitem(args, idx) } {
                            return Ok(v);
                        }
                    }
                    return Ok(w_none());
                }
            }
            "filename" | "filename2" => {
                let kind = unsafe { pyre_object::w_exception_get_kind(obj) };
                if matches!(
                    kind,
                    pyre_object::interp_exceptions::ExcKind::OSError
                        | pyre_object::interp_exceptions::ExcKind::FileNotFoundError
                ) {
                    let stored = if name == "filename" {
                        unsafe { pyre_object::interp_exceptions::w_exception_get_filename(obj) }
                    } else {
                        unsafe { pyre_object::interp_exceptions::w_exception_get_filename2(obj) }
                    };
                    if !stored.is_null() {
                        return Ok(stored);
                    }
                    // A `BlockingIOError` keeps `characters_written` (a number)
                    // in `args_w[2]`; it is not a filename (`_init_error`).
                    if name == "filename" && exc_blocking_written(obj) {
                        return Ok(w_none());
                    }
                    let args = unsafe { pyre_object::interp_exceptions::w_exception_get_args(obj) };
                    let n = unsafe { pyre_object::w_tuple_len(args) };
                    let idx: usize = if name == "filename" { 2 } else { 4 };
                    if (3..=5).contains(&n) && idx < n {
                        if let Some(v) = unsafe { pyre_object::w_tuple_getitem(args, idx as i64) } {
                            return Ok(v);
                        }
                    }
                    return Ok(w_none());
                }
                // `W_SyntaxError` also exposes `filename`, derived from its
                // `(filename, lineno, ...)` details tuple (`filename2` is OSError-only).
                if kind == pyre_object::interp_exceptions::ExcKind::SyntaxError
                    && name == "filename"
                {
                    return Ok(syntax_error_attr(obj, name));
                }
            }
            // `interp_exceptions.py:704-707 descr_get_written` — a
            // `BlockingIOError` constructed with a numeric third argument keeps
            // it in `args_w[2]` as `characters_written`; otherwise the slot is
            // unset (`written == -1`) and the attribute raises `AttributeError`.
            "characters_written" if exc_blocking_written(obj) => {
                let args = unsafe { pyre_object::interp_exceptions::w_exception_get_args(obj) };
                if let Some(v) = unsafe { pyre_object::w_tuple_getitem(args, 2) } {
                    return Ok(v);
                }
            }
            // `interp_exceptions.py:409-411 W_ImportError` exposes
            // `msg` / `path` / `name_from` as `readwrite_attrproperty_w`
            // slots stamped by `descr_init` from the keyword/positional
            // arguments, with class default `None` (`:360`).  Each is a
            // plain slot read: an instance allocated via `__new__` (which
            // never touches the slot) reads `None`.  Gated on the
            // ImportError-family kind (ImportError / ModuleNotFoundError).
            // `name` is handled by the shared arm below since NameError /
            // AttributeError expose it too.
            "msg" | "path" | "name_from" => {
                let kind = unsafe { pyre_object::w_exception_get_kind(obj) };
                if matches!(
                    kind,
                    pyre_object::interp_exceptions::ExcKind::ImportError
                        | pyre_object::interp_exceptions::ExcKind::ModuleNotFoundError
                ) {
                    let stored = unsafe {
                        match name {
                            "msg" => {
                                pyre_object::interp_exceptions::w_exception_get_import_msg(obj)
                            }
                            "path" => {
                                pyre_object::interp_exceptions::w_exception_get_import_path(obj)
                            }
                            _ => pyre_object::interp_exceptions::w_exception_get_import_name_from(
                                obj,
                            ),
                        }
                    };
                    if !stored.is_null() {
                        return Ok(stored);
                    }
                    return Ok(w_none());
                }
                // `W_SyntaxError.msg` — the first constructor argument.
                if kind == pyre_object::interp_exceptions::ExcKind::SyntaxError && name == "msg" {
                    return Ok(syntax_error_attr(obj, name));
                }
            }
            // Shared `name` attribute for the kinds that expose it —
            // `W_ImportError` (and `W_ModuleNotFoundError`), `W_NameError`,
            // and `W_AttributeError` (Python 3.10+).  Read from the shared
            // `w_exc_name` slot (default `None`); falls through to normal
            // attribute lookup on every other exception kind.
            "name" => {
                let kind = unsafe { pyre_object::w_exception_get_kind(obj) };
                if matches!(
                    kind,
                    pyre_object::interp_exceptions::ExcKind::ImportError
                        | pyre_object::interp_exceptions::ExcKind::ModuleNotFoundError
                        | pyre_object::interp_exceptions::ExcKind::NameError
                        | pyre_object::interp_exceptions::ExcKind::AttributeError
                ) {
                    let stored =
                        unsafe { pyre_object::interp_exceptions::w_exception_get_name(obj) };
                    if !stored.is_null() {
                        return Ok(stored);
                    }
                    return Ok(w_none());
                }
            }
            // `W_AttributeError.obj` (Python 3.10+) — the object whose
            // attribute lookup failed; default `None`.
            "obj" => {
                let kind = unsafe { pyre_object::w_exception_get_kind(obj) };
                if kind == pyre_object::interp_exceptions::ExcKind::AttributeError {
                    let stored =
                        unsafe { pyre_object::interp_exceptions::w_exception_get_attr_obj(obj) };
                    if !stored.is_null() {
                        return Ok(stored);
                    }
                    return Ok(w_none());
                }
            }
            // `interp_exceptions.py:468-471`
            // `readwrite_attrproperty_w('w_object', W_UnicodeTranslateError)`
            // (and `:1081-1083` / `:1201-1203` for Decode / Encode).
            // PyPy surfaces these as direct slot reads — `None` when the
            // exception was constructed without going through
            // `descr_init`.  Pyre stores `PY_NULL` in that case and
            // resolves to `space.w_None` here, matching PyPy's
            // class-default `w_object = None`.
            //
            // Gated on the three Unicode*Error kinds because PyPy
            // attaches these `attrproperty_w` descriptors only on
            // those typedefs — other exception kinds keep the regular
            // attribute lookup fall-through.
            "object" => {
                let kind = unsafe { pyre_object::w_exception_get_kind(obj) };
                if matches!(
                    kind,
                    pyre_object::interp_exceptions::ExcKind::UnicodeTranslateError
                        | pyre_object::interp_exceptions::ExcKind::UnicodeDecodeError
                        | pyre_object::interp_exceptions::ExcKind::UnicodeEncodeError
                ) {
                    let stored =
                        unsafe { pyre_object::interp_exceptions::w_exception_get_object(obj) };
                    return Ok(if stored.is_null() { w_none() } else { stored });
                }
            }
            "start" => {
                let kind = unsafe { pyre_object::w_exception_get_kind(obj) };
                if matches!(
                    kind,
                    pyre_object::interp_exceptions::ExcKind::UnicodeTranslateError
                        | pyre_object::interp_exceptions::ExcKind::UnicodeDecodeError
                        | pyre_object::interp_exceptions::ExcKind::UnicodeEncodeError
                ) {
                    let stored =
                        unsafe { pyre_object::interp_exceptions::w_exception_get_start(obj) };
                    return Ok(if stored.is_null() { w_none() } else { stored });
                }
            }
            "end" => {
                let kind = unsafe { pyre_object::w_exception_get_kind(obj) };
                if matches!(
                    kind,
                    pyre_object::interp_exceptions::ExcKind::UnicodeTranslateError
                        | pyre_object::interp_exceptions::ExcKind::UnicodeDecodeError
                        | pyre_object::interp_exceptions::ExcKind::UnicodeEncodeError
                ) {
                    let stored =
                        unsafe { pyre_object::interp_exceptions::w_exception_get_end(obj) };
                    return Ok(if stored.is_null() { w_none() } else { stored });
                }
            }
            "reason" => {
                let kind = unsafe { pyre_object::w_exception_get_kind(obj) };
                if matches!(
                    kind,
                    pyre_object::interp_exceptions::ExcKind::UnicodeTranslateError
                        | pyre_object::interp_exceptions::ExcKind::UnicodeDecodeError
                        | pyre_object::interp_exceptions::ExcKind::UnicodeEncodeError
                ) {
                    let stored =
                        unsafe { pyre_object::interp_exceptions::w_exception_get_reason(obj) };
                    return Ok(if stored.is_null() { w_none() } else { stored });
                }
            }
            "encoding" => {
                // `interp_exceptions.py:1080 W_UnicodeDecodeError.encoding`
                // / `:1200 W_UnicodeEncodeError.encoding`.
                // `W_UnicodeTranslateError` has no encoding property per
                // PyPy; the kind check here excludes Translate so
                // attribute lookup on `UnicodeTranslateError().encoding`
                // falls through to the generic AttributeError, matching
                // `interp_exceptions.py:461-471 typedef` (no `encoding`
                // attrproperty).
                let kind = unsafe { pyre_object::w_exception_get_kind(obj) };
                if matches!(
                    kind,
                    pyre_object::interp_exceptions::ExcKind::UnicodeDecodeError
                        | pyre_object::interp_exceptions::ExcKind::UnicodeEncodeError
                ) {
                    let stored =
                        unsafe { pyre_object::interp_exceptions::w_exception_get_encoding(obj) };
                    return Ok(if stored.is_null() { w_none() } else { stored });
                }
            }
            // `W_SyntaxError` location attributes, derived from the
            // `(filename, lineno, offset, text[, end_lineno, end_offset])`
            // details tuple; `print_file_and_line` is a vestigial slot.
            // `filename` / `msg` are handled by the shared arms above.
            "lineno" | "offset" | "text" | "end_lineno" | "end_offset" | "print_file_and_line" => {
                let kind = unsafe { pyre_object::w_exception_get_kind(obj) };
                if kind == pyre_object::interp_exceptions::ExcKind::SyntaxError {
                    return Ok(syntax_error_attr(obj, name));
                }
            }
            _ => {}
        }
    }
    // __dict__: use getdict() — only returns a dict for hasdict objects,
    // matching PyPy's descriptor-based __dict__ control.
    if name == "__dict__" {
        let w_dict = getdict(obj);
        if !w_dict.is_null() {
            return Ok(w_dict);
        }
    }
    // __class__: read directly from w_class field (the single source of truth).
    // objectobject.py:133-134 descr_get___class__ → space.type(w_obj)
    if name == "__class__" {
        if let Some(tp) = crate::typedef::r#type(obj) {
            return Ok(tp);
        }
    }

    // objspace/std/mapdict.py:826-840 `MapdictDictSupport.getdict` parity.
    //
    // User subclasses of builtin types (`class MyInt(int): ...`) have
    // `hasdict=True` on the subclass type and their instances are still
    // laid out as the builtin (W_IntObject etc.), so `is_instance(obj)`
    // is False and the early descriptor-protocol block at :2858 skipped
    // the instance dict. `setattr` however stores into
    // `INSTANCE_DICT[obj as usize]` via `setdictvalue` → `_obj_setdict`,
    // so the dict is populated but would never be read back.
    //
    // Check the per-instance W_DictObject here (same API PyPy's
    // `descr__getattribute__` uses at descroperation.py:50). This is the
    // second half of the "hasdict instance dict" protocol.
    let w_dict = getdict_backing(obj);
    if !w_dict.is_null() {
        if let Some(value) = unsafe { pyre_object::w_dict_getitem_str(w_dict, name) } {
            return Ok(value);
        }
    }

    // MRO lookup on the object's Python class (w_class) for method resolution.
    let w_class = unsafe { (*obj).w_class };
    if !w_class.is_null() && unsafe { is_type(w_class) } {
        if let Some(method) = unsafe { lookup_in_type_where(w_class, name) } {
            if unsafe {
                crate::is_function(method)
                    && !crate::is_builtin_code(
                        crate::function_get_code(method) as pyre_object::PyObjectRef
                    )
            } {
                return Ok(pyre_object::w_method_new(method, obj, w_class));
            }
            if let Some(result) = unsafe { get(method, obj, w_class)? } {
                return Ok(result);
            }
            return Ok(method);
        }
    }

    unsafe {
        let tp_name = if obj.is_null() {
            "NULL"
        } else {
            (*(*obj).ob_type).name
        };
        Err(PyError::attribute_error_with_context(
            format!("'{tp_name}' object has no attribute '{name}'"),
            obj,
            name,
        ))
    }
}

// Builtin type method implementations moved to type_methods.rs
// (PyPy: listobject.py, unicodeobject.py, dictmultiobject.py, tupleobject.py)

/// baseobjspace.py:317-339 `W_Root.int(space)` — the number protocol
/// portion of `space.int(w_obj)`. Look up `__int__`; if absent, fall
/// back to `__index__`. Validate the result is a `W_AbstractIntObject`.
///
/// Note: `__trunc__` is NOT consulted here. `__trunc__` belongs to the
/// `int(...)` builtin path (`intobject.py:989 _new_baseint`), not to
/// `space.int()` / `space.int_w()`.
pub(crate) fn space_int(obj: PyObjectRef) -> Result<PyObjectRef, PyError> {
    // baseobjspace.py:319 `w_impl = space.lookup(self, '__int__')`
    let w_impl = unsafe { lookup(obj, "__int__") }
        // baseobjspace.py:321-323 `w_impl = space.lookup(self, '__index__')`
        .or_else(|| unsafe { lookup(obj, "__index__") });
    let Some(method) = w_impl else {
        // baseobjspace.py:323 `self._typed_unwrap_error(space, "integer")`
        return Err(PyError::type_error("expected integer"));
    };
    // baseobjspace.py:324 `w_result = space.get_and_call_function(w_impl, self)`
    let w_result = crate::builtins::call_and_check(method, &[obj])?;
    // baseobjspace.py:326-337 validate that w_result is a W_AbstractIntObject.
    if unsafe { pyre_object::pyobject::is_int_or_long(w_result) } {
        return Ok(w_result);
    }
    // baseobjspace.py:338-339 non-int result → TypeError.
    Err(PyError::type_error("__int__ returned non-int"))
}

/// baseobjspace.py:1811-1824 `ObjSpace.int_w(w_obj,
/// allow_conversion=True)` composed with `baseobjspace.py:279-285
/// W_Root.int_w`:
///
/// ```python
/// # ObjSpace.int_w
/// return w_obj.int_w(self, allow_conversion)
/// # W_Root.int_w
/// w_obj = self
/// if allow_conversion:
///     w_obj = space.int(self)
/// return w_obj._int_w(space)
/// ```
///
/// Fast paths for `W_IntObject` / `W_LongObject` match
/// `intobject.py:558` / `longobject.py` `_int_w`. For non-int/long
/// objects, delegate to `space_int` (the `space.int(self)` protocol)
/// and then re-apply `_int_w`. `allow_conversion=True` is implicit —
/// the `unwrap_spec` call sites that pyre supports all opt in.
///
/// Floats are explicitly rejected by `floatobject.py:177`.
pub fn int_w(obj: PyObjectRef) -> Result<i64, PyError> {
    if obj.is_null() {
        return Err(PyError::type_error("int_w: null object"));
    }
    // floatobject.py:177 `int_w` — floats are explicitly rejected.
    if unsafe { pyre_object::pyobject::is_float(obj) } {
        return Err(PyError::type_error(
            "an integer is required (got type float)",
        ));
    }
    // `is_int` is true for a bool (`BOOL_TYPE`); a bool reads through
    // `w_bool_get_value`, not the int accessor, so test `is_bool` first.
    if unsafe { pyre_object::pyobject::is_bool(obj) } {
        return Ok(unsafe { pyre_object::boolobject::w_bool_get_value(obj) } as i64);
    }
    // intobject.py:558 `W_IntObject._int_w` — self.intval. Fast path.
    if unsafe { pyre_object::pyobject::is_int(obj) } {
        return Ok(unsafe { pyre_object::intobject::w_int_get_value(obj) });
    }
    // longobject.py:157 `W_LongObject._int_w` — self.num.toint(), raises
    // OverflowError if the bigint does not fit in a machine word. Fast path.
    if unsafe { pyre_object::pyobject::is_long(obj) } {
        let big = unsafe { pyre_object::longobject::w_long_get_value(obj) };
        if pyre_object::longobject::jit_bigint_to_i64_fits(big) != 0 {
            return Ok(pyre_object::longobject::jit_bigint_to_i64_value(big));
        }
        return Err(PyError::overflow_error("int too large to convert to int"));
    }
    // baseobjspace.py:284 `w_obj = space.int(self)` — __int__ or __index__.
    let w_obj = space_int(obj)?;
    // baseobjspace.py:285 `return w_obj._int_w(space)` — re-apply the
    // typed unwrap on the (int/long) result space.int returned.
    if unsafe { pyre_object::pyobject::is_int(w_obj) } {
        return Ok(unsafe { pyre_object::intobject::w_int_get_value(w_obj) });
    }
    if unsafe { pyre_object::pyobject::is_long(w_obj) } {
        let big = unsafe { pyre_object::longobject::w_long_get_value(w_obj) };
        if pyre_object::longobject::jit_bigint_to_i64_fits(big) != 0 {
            return Ok(pyre_object::longobject::jit_bigint_to_i64_value(big));
        }
        return Err(PyError::overflow_error("int too large to convert to int"));
    }
    // Unreachable: space_int returns W_AbstractIntObject or errors.
    Err(PyError::type_error("__int__ returned non-int"))
}

/// pypy/interpreter/baseobjspace.py:1957 `gateway_int_w = int_w`.
/// The gateway entry point used by `@unwrap_spec` coercion.
#[inline]
pub fn gateway_int_w(obj: PyObjectRef) -> Result<i64, PyError> {
    int_w(obj)
}

/// pypy/interpreter/baseobjspace.py gateway_nonnegint_w.
pub fn gateway_nonnegint_w(obj: PyObjectRef) -> Result<i64, PyError> {
    let value = int_w(obj)?;
    if value < 0 {
        return Err(PyError::value_error("expected a non-negative integer"));
    }
    Ok(value)
}

/// intobject.py:577 / longobject.py:164 uint_w. Unlike int_w this does NOT
/// apply __int__/__index__ conversion: a non-int object raises TypeError
/// (W_Root.uint_w → _typed_unwrap_error).
pub fn uint_w(obj: PyObjectRef) -> Result<u64, PyError> {
    use num_traits::ToPrimitive;
    if obj.is_null() {
        return Err(PyError::type_error("uint_w: null object"));
    }
    // W_IntObject.uint_w (covers bool).
    if unsafe { pyre_object::pyobject::is_int(obj) } {
        let value = unsafe { pyre_object::intobject::w_int_get_value(obj) };
        if value < 0 {
            return Err(PyError::value_error(
                "cannot convert negative integer to unsigned",
            ));
        }
        return Ok(value as u64);
    }
    // W_LongObject.uint_w — num.touint().
    if unsafe { pyre_object::pyobject::is_long(obj) } {
        let big = unsafe { crate::builtins::obj_to_bigint(obj) };
        if big.sign() == malachite_bigint::Sign::Minus {
            return Err(PyError::value_error(
                "cannot convert negative integer to unsigned int",
            ));
        }
        return big
            .to_u64()
            .ok_or_else(|| PyError::overflow_error("int too large to convert to unsigned int"));
    }
    // W_Root.uint_w → _typed_unwrap_error(space, "integer").
    let tp_name = unsafe { (*(*obj).ob_type).name };
    Err(PyError::type_error(format!(
        "expected integer, got {tp_name} object"
    )))
}

/// pypy/interpreter/baseobjspace.py c_uint_w.
pub fn c_uint_w(obj: PyObjectRef) -> Result<u32, PyError> {
    let value = uint_w(obj)?;
    if value > u32::MAX as u64 {
        return Err(PyError::overflow_error(
            "expected an unsigned 32-bit integer",
        ));
    }
    Ok(value as u32)
}

/// pypy/interpreter/baseobjspace.py c_nonnegint_w.
pub fn c_nonnegint_w(obj: PyObjectRef) -> Result<i32, PyError> {
    let value = int_w(obj)?;
    if value < 0 {
        return Err(PyError::value_error("expected a non-negative integer"));
    }
    if value > i32::MAX as i64 {
        return Err(PyError::overflow_error("expected a 32-bit integer"));
    }
    Ok(value as i32)
}

/// pypy/interpreter/baseobjspace.py c_short_w.
pub fn c_short_w(obj: PyObjectRef) -> Result<i16, PyError> {
    let value = int_w(obj)?;
    if value < i16::MIN as i64 {
        return Err(PyError::overflow_error(
            "signed short integer is less than minimum",
        ));
    }
    if value > i16::MAX as i64 {
        return Err(PyError::overflow_error(
            "signed short integer is greater than maximum",
        ));
    }
    Ok(value as i16)
}

/// pypy/interpreter/baseobjspace.py c_ushort_w.
pub fn c_ushort_w(obj: PyObjectRef) -> Result<u16, PyError> {
    let value = int_w(obj)?;
    if value < 0 {
        return Err(PyError::value_error("value must be positive"));
    }
    if value > u16::MAX as i64 {
        return Err(PyError::overflow_error(
            "Python int too large for C unsigned short",
        ));
    }
    Ok(value as u16)
}

/// pypy/interpreter/baseobjspace.py c_uid_t_w. Equivalent to c_uint_w,
/// except -1 maps to UINT_MAX ((uid_t)-1) and values below -1 raise
/// OverflowError rather than ValueError. `uint_w` does not run any
/// __index__ conversion, so the `int_w` retry on the negative branch sees
/// only the real int and is side-effect free.
pub fn c_uid_t_w(obj: PyObjectRef) -> Result<u32, PyError> {
    match c_uint_w(obj) {
        Ok(value) => Ok(value),
        Err(e) if e.kind == PyErrorKind::ValueError => {
            if int_w(obj)? == -1 {
                Ok(u32::MAX)
            } else {
                Err(PyError::overflow_error(
                    "user/group id smaller than minimum (-1)",
                ))
            }
        }
        Err(e) => Err(e),
    }
}

/// baseobjspace.py:2063 c_filedescriptor_w — an int or an object exposing
/// a `fileno()` method (deliberately NOT an `__int__`), coerced to a
/// non-negative C int.  `isinstance_w(w_fd, w_int)` accepts a long or an
/// int subclass as well, so the membership test is `is_int_or_long`.
pub fn c_filedescriptor_w(obj: PyObjectRef) -> Result<i32, PyError> {
    let w_fd = if unsafe { pyre_object::pyobject::is_int_or_long(obj) } {
        obj
    } else {
        let fileno = getattr_str(obj, "fileno").map_err(|e| {
            if e.kind == PyErrorKind::AttributeError {
                PyError::type_error("argument must be an int, or have a fileno() method.")
            } else {
                e
            }
        })?;
        let w_fd = crate::builtins::call_and_check(fileno, &[])?;
        if unsafe { !pyre_object::pyobject::is_int_or_long(w_fd) } {
            return Err(PyError::type_error("fileno() returned a non-integer"));
        }
        w_fd
    };
    let fd = c_int_w(w_fd)?;
    if fd < 0 {
        return Err(PyError::value_error(format!(
            "file descriptor cannot be a negative integer ({fd})"
        )));
    }
    Ok(fd)
}

/// pypy/interpreter/baseobjspace.py truncatedint_w.
pub fn truncatedint_w(obj: PyObjectRef) -> Result<i64, PyError> {
    match int_w(obj) {
        Ok(value) => Ok(value),
        Err(e) if e.kind == PyErrorKind::OverflowError => {
            use num_traits::ToPrimitive;
            // intmask(self.bigint_w(w_obj).uintmask()): bigint_w applies
            // __int__/__index__ conversion, so read the bigint from the
            // converted int object rather than the raw argument.
            let w_int_obj = if unsafe { pyre_object::pyobject::is_int_or_long(obj) } {
                obj
            } else {
                space_int(obj)?
            };
            let big = unsafe { crate::builtins::obj_to_bigint(w_int_obj) };
            let low = (&big & BigInt::from(u64::MAX)).to_u64().unwrap_or(0);
            Ok(low as i64)
        }
        Err(e) => Err(e),
    }
}

/// pypy/interpreter/baseobjspace.py:1976-1982 `c_int_w(w_obj)`.
///
/// ```python
/// def c_int_w(self, w_obj):
///     value = self.gateway_int_w(w_obj)
///     if value < INT_MIN or value > INT_MAX:
///         raise oefmt(self.w_OverflowError, "expected a 32-bit integer")
///     return value
/// ```
///
/// Used by `@unwrap_spec(name="c_int")` (gateway.py). The only caller
/// today is `sys.setrecursionlimit` (pypy/module/sys/vm.py:63), whose
/// argument is typed as `c_int`; values outside the 32-bit signed
/// range surface as `OverflowError` rather than a silent clamp.
pub fn c_int_w(obj: PyObjectRef) -> Result<i32, PyError> {
    let value = gateway_int_w(obj)?;
    if !(i32::MIN as i64..=i32::MAX as i64).contains(&value) {
        return Err(PyError::overflow_error("expected a 32-bit integer"));
    }
    Ok(value as i32)
}

/// baseobjspace.py:1784 text_w.
pub fn text_w(obj: PyObjectRef) -> Result<&'static str, PyError> {
    if unsafe { !isinstance_str_w(obj) } {
        return Err(PyError::type_error("expected str"));
    }
    Ok(unsafe { pyre_object::w_str_get_value(obj) })
}

/// baseobjspace.py:1791 utf8_w.
pub fn utf8_w(obj: PyObjectRef) -> Result<&'static str, PyError> {
    text_w(obj)
}

/// baseobjspace.py realunicode_w.
pub fn realunicode_w(obj: PyObjectRef) -> Result<&'static str, PyError> {
    if unsafe { !isinstance_str_w(obj) } {
        return Err(PyError::type_error("expected unicode"));
    }
    Ok(unsafe { pyre_object::w_str_get_value(obj) })
}

/// baseobjspace.py text0_w — rejects an embedded null character.
pub fn text0_w(obj: PyObjectRef) -> Result<&'static str, PyError> {
    let s = text_w(obj)?;
    if s.contains('\0') {
        return Err(PyError::value_error("embedded null character"));
    }
    Ok(s)
}

/// baseobjspace.py:1819 charbuf_w — a read-only character buffer as raw bytes.
pub fn charbuf_w(obj: PyObjectRef) -> Result<&'static [u8], PyError> {
    if unsafe { !pyre_object::bytesobject::is_bytes_like(obj) } {
        return Err(PyError::type_error("expected a readable buffer"));
    }
    Ok(unsafe { pyre_object::bytesobject::bytes_like_data(obj) })
}

/// Look up a descriptor on an object's type.
///
/// PyPy equivalent: `space.lookup(w_obj, name)`.
pub unsafe fn lookup(obj: PyObjectRef, name: &str) -> Option<PyObjectRef> {
    let w_type = crate::typedef::r#type(obj)?;
    lookup_in_type(w_type, name)
}

/// Look up a name on a type by walking the C3 MRO.
///
/// PyPy equivalent: `space.lookup_in_type(w_type, name)`.
pub unsafe fn lookup_in_type(w_type: PyObjectRef, name: &str) -> Option<PyObjectRef> {
    lookup_in_type_where(w_type, name)
}

/// `typeobject.py:353-371 W_TypeObject.compares_by_identity` — walk
/// the MRO checking whether any class **before `object`** defines
/// `__eq__` or `__hash__`.
///
/// The cached status slot on W_TypeObject short-circuits repeat
/// calls; cache miss recomputes and writes back.  Cache validity is
/// maintained by [`mutated`] below — the setattr / delattr paths
/// invoke it on every type-dict change, so adding `__eq__` /
/// `__hash__` to a live class resets the slot back to UNKNOWN
/// across the subclass tree.
///
/// PyPy reads `object_hash(self.space)` and `type_eq(self.space)` —
/// static singletons resolved at translation time.  Pyre walks the
/// MRO and stops at `w_object()` (`typedef.rs:734`); any class on
/// the path that owns `__eq__` or `__hash__` short-circuits to
/// `OVERRIDES_EQ_CMP_OR_HASH`.
///
/// # Safety
/// `w_type` must point at a valid `W_TypeObject` (null tolerated).
pub unsafe fn compares_by_identity(w_type: PyObjectRef) -> bool {
    if w_type.is_null() || !is_type(w_type) {
        return false;
    }
    let cached = pyre_object::typeobject::w_type_compares_by_identity_status(w_type);
    if cached == pyre_object::typeobject::COMPARES_BY_IDENTITY_YES {
        return true;
    }
    if cached == pyre_object::typeobject::COMPARES_BY_IDENTITY_NO {
        return false;
    }
    let object_type = crate::typedef::w_object();
    let cached_mro = pyre_object::typeobject::w_type_get_mro(w_type);
    let mro_owned;
    let mro: &[PyObjectRef] = if !cached_mro.is_null() {
        &*cached_mro
    } else {
        mro_owned = compute_mro(w_type);
        &mro_owned
    };
    let mut compares_by_identity = true;
    for cls in mro {
        if (*cls).is_null() || !is_type(*cls) {
            continue;
        }
        if *cls == object_type {
            break;
        }
        let ns_ptr = pyre_object::typeobject::w_type_get_dict_ptr(*cls) as *mut crate::DictStorage;
        if ns_ptr.is_null() {
            continue;
        }
        let ns = &*ns_ptr;
        if let Some(&v) = ns.get("__eq__") {
            if !v.is_null() {
                compares_by_identity = false;
                break;
            }
        }
        if let Some(&v) = ns.get("__hash__") {
            if !v.is_null() {
                compares_by_identity = false;
                break;
            }
        }
    }
    let result = if compares_by_identity {
        pyre_object::typeobject::COMPARES_BY_IDENTITY_YES
    } else {
        pyre_object::typeobject::COMPARES_BY_IDENTITY_NO
    };
    pyre_object::typeobject::w_type_set_compares_by_identity_status(w_type, result);
    compares_by_identity
}

/// `typeobject.py:266-291 W_TypeObject.mutated` — type-dict change
/// observer.  Resets cached lookup state on `w_type` and recurses
/// into `weak_subclasses` so cross-subclass caches stay coherent.
///
/// `key` is either the mutated attribute name or `None` for a
/// generic invalidation; `compares_by_identity_status` reset is
/// gated on the key being `__eq__` / `__hash__` per PyPy line 279.
/// The `uses_object_getattribute` / `uses_object_setattr` flags
/// (typeobject.py:275-276) and the `_version_tag` bump
/// (typeobject.py:285-286) are reset here; the remaining slot PyPy
/// resets (`w_new_function`) hooks in once that cache lands.
///
/// # Safety
/// `w_type` must be a valid `PyObjectRef` pointing at a
/// `W_TypeObject` (null tolerated).
pub unsafe fn mutated(w_type: PyObjectRef, key: Option<&str>) {
    if w_type.is_null() || !is_type(w_type) {
        return;
    }
    // typeobject.py:275-276 — conservative default; the attribute fast
    // paths re-confirm the flag on the next lookup.
    pyre_object::typeobject::w_type_set_uses_object_getattribute(w_type, false);
    pyre_object::typeobject::w_type_set_uses_object_setattr(w_type, false);
    // typeobject.py:279 — `if (key is None or key == '__eq__' or
    // key == '__hash__'): self.compares_by_identity_status =
    // UNKNOWN`.
    let resets_compare = match key {
        None => true,
        Some(k) => k == "__eq__" || k == "__hash__",
    };
    if resets_compare {
        pyre_object::typeobject::w_type_set_compares_by_identity_status(
            w_type,
            pyre_object::typeobject::COMPARES_BY_IDENTITY_UNKNOWN,
        );
    }
    // typeobject.py:285-286 — `if self._version_tag is not None:
    // self._version_tag = VersionTag()`. A fresh identity invalidates every
    // cache keyed on the old tag; a 0 (None) tag stays uncacheable.
    if pyre_object::typeobject::w_type_get_version_tag(w_type) != 0 {
        pyre_object::typeobject::w_type_set_version_tag(
            w_type,
            pyre_object::typeobject::new_version_tag(),
        );
    }
    // typeobject.py:288-291 — walk direct subclasses recursively.
    let subs = pyre_object::typeobject::w_type_get_subclasses(w_type);
    for w_sub in subs {
        mutated(w_sub, key);
    }
}

/// typeobject.py `_lookup_where(self, key)` — linear search through `self.mro_w`,
/// returning the MRO class that defines `key` together with the found descriptor.
/// The interpreter reaches this through the `MethodCache` front door
/// (`lookup_in_type_where`); the elidable wrapper
/// `_pure_lookup_where_with_method_cache(name, version_tag)` that lets the JIT
/// constant-fold lookups on a promoted `version_tag` is the remaining slice —
/// this raw walk stays non-elidable until then.
pub(crate) unsafe fn lookup_where(
    w_type: PyObjectRef,
    name: &str,
) -> Option<(PyObjectRef, PyObjectRef)> {
    if w_type.is_null() || !is_type(w_type) {
        return None;
    }
    // Use cached MRO if available (PyPy: W_TypeObject.mro_w)
    let cached = w_type_get_mro(w_type);
    let mro_owned;
    let mro: &[PyObjectRef] = if !cached.is_null() {
        &*cached
    } else {
        mro_owned = compute_mro(w_type);
        &mro_owned
    };
    for cls in mro {
        if (*cls).is_null() || !is_type(*cls) {
            continue;
        }
        let ns_ptr = w_type_get_dict_ptr(*cls) as *mut crate::DictStorage;
        if !ns_ptr.is_null() {
            let ns = &*ns_ptr;
            if let Some(&value) = ns.get(name) {
                if !value.is_null() {
                    return Some((*cls, value));
                }
            }
        }
    }
    None
}

/// `lookup_in_type` / `lookup_in_type_where` descriptor-only projection of
/// [`lookup_where`], for the callers that do not need the defining class.
/// The raw walk underneath the method cache; also the JIT path (no
/// thread-local state) so the trace can lower it directly.
unsafe fn lookup_in_type_where_uncached(w_type: PyObjectRef, name: &str) -> Option<PyObjectRef> {
    lookup_where(w_type, name).map(|(_src, value)| value)
}

/// WTF-8 keyed MRO attribute lookup — surrogate-safe sibling of
/// `lookup_in_type` / `lookup_in_type_where`.  A lone-surrogate name
/// can only live in a class namespace's `DictStorage` (never as a
/// descriptor or special slot), so this walks the MRO comparing raw
/// WTF-8 bytes via `DictStorage::get_wtf8`.
///
/// # Safety
/// `w_type` must point at a valid `W_TypeObject` (null tolerated).
pub(crate) unsafe fn lookup_in_type_wtf8(w_type: PyObjectRef, name: &Wtf8) -> Option<PyObjectRef> {
    if w_type.is_null() || !is_type(w_type) {
        return None;
    }
    let cached = w_type_get_mro(w_type);
    let mro_owned;
    let mro: &[PyObjectRef] = if !cached.is_null() {
        &*cached
    } else {
        mro_owned = compute_mro(w_type);
        &mro_owned
    };
    for cls in mro {
        if (*cls).is_null() || !is_type(*cls) {
            continue;
        }
        let ns_ptr = w_type_get_dict_ptr(*cls) as *mut crate::DictStorage;
        if !ns_ptr.is_null() {
            let ns = &*ns_ptr;
            if let Some(&value) = ns.get_wtf8(name) {
                if !value.is_null() {
                    return Some(value);
                }
            }
        }
    }
    None
}

/// `typeobject.py:76-101 MethodCache` — the per-space method-lookup
/// cache.  `space.fromcache(MethodCache)` returns one instance per
/// space; pyre is single-space, so a thread-local singleton stands in
/// (the role the dict-strategy `space.fromcache` singletons already
/// play).  `versions[h]` is the `version_tag()` the slot was filled
/// under (`0` = empty), `names[h]` the looked-up name, `lookup_where[h]`
/// the cached `(w_class, w_value)` pair from the MRO walk — a
/// `(null, null)` pair on a valid `(version, name)` entry is the cached
/// negative result (`_lookup_where_all_typeobjects` returned
/// `(None, None)`).
struct MethodCache {
    versions: Vec<u64>,
    names: Vec<Option<String>>,
    lookup_where: Vec<(PyObjectRef, PyObjectRef)>,
}

/// `space.config.objspace.std.methodcachesizeexp` default.
const METHOD_CACHE_SIZE_EXP: u32 = 11;
const METHOD_CACHE_SIZE: usize = 1 << METHOD_CACHE_SIZE_EXP;

thread_local! {
    static METHOD_CACHE: std::cell::RefCell<MethodCache> = std::cell::RefCell::new(MethodCache {
        versions: vec![0u64; METHOD_CACHE_SIZE],
        names: vec![None; METHOD_CACHE_SIZE],
        lookup_where: vec![(std::ptr::null_mut(), std::ptr::null_mut()); METHOD_CACHE_SIZE],
    });
}

/// `typeobject.py:520-535` method-hash.  `version_tag` is pyre's u64
/// version token directly (PyPy hashes `current_object_addr_as_int(
/// version_tag)`; the u64 is its own address-stable surrogate).
/// `name_hash` only needs to be deterministic — the slot's validity is
/// the exact `(version, name)` match, not the hash — so an FNV-1a over
/// the bytes stands in for `compute_hash(name)`.
fn method_hash(version_tag: u64, name: &str) -> usize {
    let mut name_hash: u64 = 0xcbf2_9ce4_8422_2325;
    for b in name.as_bytes() {
        name_hash ^= *b as u64;
        name_hash = name_hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    const SHIFT2: u32 = 64 - METHOD_CACHE_SIZE_EXP;
    const SHIFT1: u32 = SHIFT2 - 5;
    let product = version_tag.wrapping_mul(name_hash);
    let h = (product ^ (product << SHIFT1)) >> SHIFT2;
    (h as usize) & (METHOD_CACHE_SIZE - 1)
}

/// typeobject.py:299-301 `_pure_version_tag` — the `@elidable_promote`
/// half of `version_tag()`.  The body is the raw `_version_tag` field
/// read; the macro renames it to the hidden elidable original and
/// `_pure_version_tag` becomes the promoting wrapper (mirrors
/// `function.rs::_get_immutable_code`).  For a prebuilt (non-heap) type
/// the tag never changes, so promoting it lets the trace fold every
/// `version_tag`-keyed lookup to a constant.
#[majit_macros::elidable_promote]
#[inline]
pub unsafe fn _pure_version_tag(w_type: PyObjectRef) -> u64 {
    unsafe { pyre_object::typeobject::w_type_get_version_tag(w_type) }
}

/// typeobject.py:293-297 `version_tag()` — the cache-version reader.
/// In the interpreter, and for a heap type whose tag can still change,
/// it reads the live `_version_tag` field; under the JIT for a prebuilt
/// type it goes through the `@elidable_promote` `_pure_version_tag` so
/// the value folds away on the trace.  Mirrors the nesting of
/// `function.rs::getcode`.
#[inline]
pub(crate) unsafe fn w_type_version_tag(w_type: PyObjectRef) -> u64 {
    if majit_metainterp::jit::we_are_jitted() {
        if pyre_object::typeobject::w_type_is_heaptype(w_type) {
            // Heap types can still be mutated; read the live field (the
            // caller promotes the result).
            return pyre_object::typeobject::w_type_get_version_tag(w_type);
        }
        // Prebuilt objects cannot get their version_tag changed.
        return _pure_version_tag(w_type);
    }
    pyre_object::typeobject::w_type_get_version_tag(w_type)
}

/// `typeobject.py:516-552 _pure_lookup_where_with_method_cache` — the
/// `@elidable` lookup keyed on `(version_tag, name)`.  Consults the
/// `MethodCache`; on a miss it runs the raw MRO walk and fills the slot
/// (`typeobject.py:545-549`).  `@elidable`: the result depends only on
/// `(w_type, name, version_tag)`, so the trace records a `CALL_PURE_R`
/// and folds repeated same-key lookups; the `MethodCache` mutation is
/// the idempotent memo side-effect the elidable contract tolerates
/// (typeobject.py:546 `_side_effects_ok()`).  Being a residual call, the
/// thread-local / `String` machinery stays off the trace surface, as the
/// former `dont_look_inside` ensured.
///
/// `name` arrives as `w_name`, an interned immortal str object
/// (`box_str_constant`), because the elidable call ABI cannot pass a
/// `&str` and an interned pointer is the green token the trace folds on;
/// the body reads it back via `w_str_get_value`.  The result is a raw
/// pointer — null is the cached negative result (`None`), since the call
/// ABI cannot carry `Option`.  The `is_type` / `version_tag == 0` guards
/// live in the front door, so this is only ever entered with a valid
/// promoted `version_tag`.
#[majit_macros::elidable]
pub unsafe fn _pure_lookup_where_with_method_cache(
    w_type: PyObjectRef,
    w_name: PyObjectRef,
    version_tag: u64,
) -> PyObjectRef {
    let name = pyre_object::unicodeobject::w_str_get_value(w_name);
    // PyPy's elidable returns the cached `(w_class, w_value)` tuple
    // object; the residual-call ABI here carries one raw register, so
    // the elidable surface projects the `w_value` half.  Callers that
    // need the defining class go through the interpreter front door
    // `lookup_where_with_method_cache`, which reads the same cache
    // entry.
    _cached_lookup_where(w_type, name, version_tag).1
}

/// `lookup_where` *class* projection — the `@elidable` companion of
/// [`_pure_lookup_where_with_method_cache`] returning the defining-class
/// half (`(w_class, w_value).0`) of the `(version_tag, name)`-keyed
/// `MethodCache` entry.  PyPy's single elidable returns the whole
/// `(w_class, w_value)` tuple (`typeobject.py:510-511`); the residual-call
/// ABI carries one raw register, so the two halves are exposed as two
/// single-register elidable surfaces over the same cache entry.  The front
/// door [`lookup_where_with_method_cache`] reads both halves through these
/// residuals so the thread-local `MethodCache` machinery stays off the
/// trace surface.  The value half runs first and fills the slot, so this
/// call is a guaranteed cache hit.
///
/// See [`_pure_lookup_where_with_method_cache`] for the interned-`w_name`
/// ABI and the null-as-`None` convention (a negative result has a null
/// `w_class`).
#[majit_macros::elidable]
pub unsafe fn _pure_lookup_class_with_method_cache(
    w_type: PyObjectRef,
    w_name: PyObjectRef,
    version_tag: u64,
) -> PyObjectRef {
    let name = pyre_object::unicodeobject::w_str_get_value(w_name);
    _cached_lookup_where(w_type, name, version_tag).0
}

/// The `MethodCache` probe/fill shared by the `@elidable` JIT surface
/// and the interpreter front door.  Probes the `(version_tag, name)`
/// slot; on a miss runs the raw MRO walk (`typeobject.py:478-489
/// _lookup_where_all_typeobjects`) and fills the slot with the
/// `(w_class, w_value)` pair (`typeobject.py:545-549`).
unsafe fn _cached_lookup_where(
    w_type: PyObjectRef,
    name: &str,
    version_tag: u64,
) -> (PyObjectRef, PyObjectRef) {
    let h = method_hash(version_tag, name);
    // Probe without holding the borrow across the MRO walk below.
    let hit = METHOD_CACHE.with(|c| {
        let cache = c.borrow();
        if cache.versions[h] == version_tag && cache.names[h].as_deref() == Some(name) {
            // A valid entry with null pointers is the cached negative result.
            Some(cache.lookup_where[h])
        } else {
            None
        }
    });
    if let Some(tup) = hit {
        return tup;
    }
    let tup = lookup_where(w_type, name).unwrap_or((std::ptr::null_mut(), std::ptr::null_mut()));
    // Prebuilt-family store: the cache slot is reached only by
    // `walk_method_cache_gc`, skipped on clean minor collections.
    pyre_object::gc_roots::mark_prebuilt_roots_dirty();
    METHOD_CACHE.with(|c| {
        let mut cache = c.borrow_mut();
        cache.versions[h] = version_tag;
        cache.names[h] = Some(name.to_string());
        cache.lookup_where[h] = tup;
    });
    tup
}

/// Forward every cached `values[h]` slot during collection — the
/// faithful equivalent of RPython's GC tracing `MethodCache.lookup_where`
/// (which holds live `(w_class, w_value)` refs).
///
/// Every cached value is an MRO type's namespace-dict resident (or a
/// Box-immortal builtin descriptor), so it is already kept *reachable* by
/// `walk_type_dicts_gc` — this walk is therefore not load-bearing for
/// reclamation in the current model, and old-gen residents are not
/// relocated by a collection, so it forwards no slot in practice today.
/// It becomes load-bearing once those values become movable (the
/// movable-object GC phases): `version_tag` is bumped only by `mutated()`,
/// never by a relocating move, so the cache's *own* copy of each pointer
/// must be forwarded here or a later hit would read a stale address.
pub(crate) unsafe fn walk_method_cache_gc(forward: &mut dyn FnMut(&mut PyObjectRef)) {
    METHOD_CACHE.with(|c| {
        let mut cache = c.borrow_mut();
        for (w_class, w_value) in cache.lookup_where.iter_mut() {
            // Empty / negative-cache slots hold nulls: nothing to forward.
            if !w_class.is_null() {
                forward(w_class);
            }
            if !w_value.is_null() {
                forward(w_value);
            }
        }
    });
}

/// `typeobject.py:503-514 lookup_where_with_method_cache` — the method
/// cache front door.  One path for interpreter and JIT (PyPy branches
/// only on `version_tag is None`, never on `we_are_jitted()`): promote
/// the type and its `version_tag`, then consult the
/// `(version_tag, name)`-keyed `MethodCache` for the `(w_class,
/// w_value)` pair.  pyre's type namespaces hold plain values (no
/// `MutableCell` strategy yet), so the `typeobject.py:511-513
/// unwrap_cell` boundary has nothing to unwrap.
pub(crate) unsafe fn lookup_where_with_method_cache(
    w_type: PyObjectRef,
    name: &str,
) -> Option<(PyObjectRef, PyObjectRef)> {
    if w_type.is_null() || !is_type(w_type) {
        return lookup_where(w_type, name);
    }
    // typeobject.py:505 — `promote(self)`.
    let _ = majit_metainterp::jit::promote(w_type);
    // typeobject.py:506 — `version_tag = promote(self.version_tag())`;
    // `w_type_version_tag` routes a prebuilt type through the
    // `elidable_promote` `_pure_version_tag`, so the tag folds on the trace.
    let version_tag = w_type_version_tag(w_type);
    if version_tag == 0 {
        // typeobject.py:507-509 — no version tag: uncacheable.
        return lookup_where(w_type, name);
    }
    // typeobject.py:510-511 — `w_class, w_value =
    // self._pure_lookup_where_with_method_cache(name, version_tag)`.  The
    // tuple is split across two single-register elidable residuals over the
    // same cache entry (see `_pure_lookup_class_with_method_cache`), so the
    // thread-local `MethodCache` read stays off the trace surface and the
    // lookup folds to a `CALL_PURE_R` instead of aborting the trace.  The
    // interned, immortal `w_name` (`box_str_constant`) is the green token the
    // trace folds on; both residuals share it.
    let w_name = pyre_object::unicodeobject::box_str_constant(rustpython_wtf8::Wtf8::new(name));
    let w_value = _pure_lookup_where_with_method_cache(w_type, w_name, version_tag);
    if w_value.is_null() {
        None
    } else {
        let w_class = _pure_lookup_class_with_method_cache(w_type, w_name, version_tag);
        Some((w_class, w_value))
    }
}

/// `lookup` value projection of [`lookup_where_with_method_cache`]
/// (typeobject.py:476 `lookup` = `self.lookup_where(name)[1]`).  Under
/// the JIT this routes through the `@elidable`
/// `_pure_lookup_where_with_method_cache` so the trace records a
/// `CALL_PURE_R` and folds the `(version_tag, name)`-keyed lookup to a
/// constant.
pub(crate) unsafe fn lookup_in_type_where(w_type: PyObjectRef, name: &str) -> Option<PyObjectRef> {
    if w_type.is_null() || !is_type(w_type) {
        return lookup_in_type_where_uncached(w_type, name);
    }
    // typeobject.py:505 — `promote(self)`.
    let _ = majit_metainterp::jit::promote(w_type);
    // typeobject.py:506 — `version_tag = promote(self.version_tag())`.
    let version_tag = w_type_version_tag(w_type);
    if version_tag == 0 {
        // typeobject.py:507-509 — no version tag: uncacheable.
        return lookup_in_type_where_uncached(w_type, name);
    }
    // The elidable takes the name as an interned, immortal str object
    // (`box_str_constant`: content-keyed, never freed).  The elidable call
    // ABI cannot pass a `&str`, and the interned pointer is the green token
    // the trace folds the lookup on (PyPy's `name` is already an interned
    // W_StringObject green constant from the bytecode).
    let w_name = pyre_object::unicodeobject::box_str_constant(rustpython_wtf8::Wtf8::new(name));
    // typeobject.py:510 — `_pure_lookup_where_with_method_cache(name, version_tag)`.
    let v = _pure_lookup_where_with_method_cache(w_type, w_name, version_tag);
    if v.is_null() { None } else { Some(v) }
}

/// `objspace.py:817 getfulltypename` — the type name used by the default
/// object repr.  A heaptype renders as `<module>.<qualname>` when it
/// carries a string `__module__`; a builtin type is just its `name`.
///
/// # Safety
/// `w_obj` must be a valid, non-null `PyObject`.
pub unsafe fn getfulltypename(w_obj: PyObjectRef) -> String {
    match crate::typedef::r#type(w_obj) {
        Some(w_type) => getfulltypename_of_type(w_type),
        None => "object".to_string(),
    }
}

/// [`getfulltypename`] for an already-resolved type.
///
/// # Safety
/// `w_type` must be a valid `W_TypeObject`.
pub unsafe fn getfulltypename_of_type(w_type: PyObjectRef) -> String {
    if !pyre_object::w_type_is_heaptype(w_type) {
        return w_type_get_name(w_type).to_string();
    }
    // `w_type.getqualname(space)` — an explicit `__qualname__` set in the
    // class body, else the bare name.
    let qualname = match lookup_in_type_where(w_type, "__qualname__") {
        Some(qn) if is_str(qn) => w_str_get_value(qn).to_string(),
        _ => w_type_get_name(w_type).to_string(),
    };
    // `w_type.lookup("__module__")` prepends a string module name; a
    // non-string `__module__` is ignored (`utf8_w` raises `TypeError`).
    match lookup_in_type(w_type, "__module__") {
        Some(m) if is_str(m) => format!("{}.{qualname}", w_str_get_value(m)),
        _ => qualname,
    }
}

/// `typeobject.py:797 descr_repr` name component — the `module.qualname`
/// rendered inside `<class '…'>` when a heaptype carries a string
/// `__module__` other than `builtins`, else the bare `name` (a builtin
/// type's dotted `name` already carries its module).
///
/// # Safety
/// `w_type` must be a valid `W_TypeObject`.
pub unsafe fn type_repr_qualified_name(w_type: PyObjectRef) -> String {
    let name = w_type_get_name(w_type).to_string();
    if !pyre_object::w_type_is_heaptype(w_type) {
        return name;
    }
    let module = lookup_in_type_where(w_type, "__module__")
        .filter(|m| is_str(*m))
        .map(|m| w_str_get_value(m).to_string());
    match module {
        Some(m) if m != "builtins" => {
            let qualname = match lookup_in_type_where(w_type, "__qualname__") {
                Some(qn) if is_str(qn) => w_str_get_value(qn).to_string(),
                _ => name,
            };
            format!("{m}.{qualname}")
        }
        _ => name,
    }
}

/// `callmethod.py:25-85 LOAD_METHOD` fast-path decision, shared by the
/// interpreter (`eval::load_method`) and the JIT tracer
/// (`trace_opcode::try_load_method_fast_path`) so both produce the
/// identical `[w_descr, w_obj]` stack shape — the tracer otherwise records
/// a `[descr, self]` shape while the concrete frame keeps the `getattr`
/// `[bound_method, null]` shape, and the two desync at the following
/// `CALL`.
///
/// Returns `Some((w_type, version_tag, w_descr))` for the
/// plain-instance-method case the fast path binds `self` for
/// (callmethod.py:55-68); `None` (fall back to `getattr`) for every other
/// receiver / descriptor shape.  Mirror of `callmethod.py`:
/// `has_object_getattribute()` (line 33), `version_tag()` (line 56),
/// `_pure_lookup_where_with_method_cache` (line 59),
/// `flag_method_descriptor` (line 66), and the instance-dict shadowing
/// check (line 66).
///
/// # Safety
/// `w_obj` must be a valid object pointer (null tolerated).
pub unsafe fn load_method_fast_path(
    w_obj: PyObjectRef,
    name: &str,
) -> Option<(PyObjectRef, u64, PyObjectRef)> {
    if w_obj.is_null() || !is_instance(w_obj) {
        return None;
    }
    let w_type = w_instance_get_type(w_obj);
    if w_type.is_null() {
        return None;
    }
    // typeobject.py:56-58 `version_tag = self.version_tag()`; `None`
    // (uncacheable) is `0` here.
    let version_tag = pyre_object::typeobject::w_type_get_version_tag(w_type);
    if version_tag == 0 {
        return None;
    }
    // callmethod.py:46 `w_type.has_object_getattribute()` — only the default
    // `__getattribute__` is sound to bypass (typeobject.py:303-326).  The
    // computing form looks `__getattribute__` up and memoizes the default
    // in `uses_object_getattribute`, so the first access takes the fast
    // path too (not only after the interpreter `getattr` primed the flag).
    if !has_object_getattribute(w_type) {
        return None;
    }
    // callmethod.py:59 `_pure_lookup_where_with_method_cache(name, vt)`.
    let w_descr = lookup_in_type(w_type, name)?;
    // callmethod.py:66 `space.type(w_descr).flag_method_descriptor`: only a
    // method-descriptor type (the `function` typedef, typedef.py:807) binds
    // `self` here; builtin functions, staticmethod / classmethod / property
    // / member / type descriptors all carry the `False` default.
    let w_descr_type = crate::typedef::r#type(w_descr)?;
    if !pyre_object::typeobject::w_type_get_flag_method_descriptor(w_descr_type) {
        return None;
    }
    // callmethod.py:66-67 `w_value = w_obj.getdictvalue(space, name)`: a
    // shadowing instance attribute means the method is not bound.
    if crate::objspace::std::mapdict::instance_node_getdictvalue(w_obj, Wtf8::new(name)).is_some() {
        return None;
    }
    Some((w_type, version_tag, w_descr))
}

/// descroperation.py:12-15 `object_getattribute(space)` — the canonical
/// `object.__getattribute__` descriptor used as the identity anchor for
/// the `uses_object_getattribute` fast path.  Returns true iff `w_descr`
/// is that descriptor.
unsafe fn is_object_getattribute_descr(w_descr: PyObjectRef) -> bool {
    match lookup_in_type_where(crate::typedef::w_object(), "__getattribute__") {
        Some(d) => std::ptr::eq(w_descr, d),
        None => false,
    }
}

/// descroperation.py:17-20 `object_setattr(space)` — the canonical
/// `object.__setattr__` descriptor anchor (see
/// [`is_object_getattribute_descr`]).
unsafe fn is_object_setattr_descr(w_descr: PyObjectRef) -> bool {
    match lookup_in_type_where(crate::typedef::w_object(), "__setattr__") {
        Some(d) => std::ptr::eq(w_descr, d),
        None => false,
    }
}

/// typeobject.py:303-326 `getattribute_if_not_from_object` — returns the
/// app-level `__getattribute__` if it is NOT `object.__getattribute__`,
/// otherwise `None`.  In the interpreter the negative result is memoized
/// in `uses_object_getattribute` so repeat accesses skip the MRO walk +
/// identity compare; under the JIT the lookup is left raw and folded away
/// by the type's `version_tag`.
pub(crate) unsafe fn getattribute_if_not_from_object(w_type: PyObjectRef) -> Option<PyObjectRef> {
    if majit_metainterp::jit::we_are_jitted() {
        // typeobject.py:319-323 — just a lookup, folded by version_tag.
        if let Some(w_descr) = lookup_in_type_where(w_type, "__getattribute__") {
            if is_object_getattribute_descr(w_descr) {
                return None;
            }
            return Some(w_descr);
        }
        return None;
    }
    // typeobject.py:308 — fast path once the default is confirmed.
    if pyre_object::typeobject::w_type_get_uses_object_getattribute(w_type) {
        return None;
    }
    if let Some(w_descr) = lookup_in_type_where(w_type, "__getattribute__") {
        if is_object_getattribute_descr(w_descr) {
            // typeobject.py:313-315 — remember the default (`_side_effects_ok()`
            // is true in normal builds).
            pyre_object::typeobject::w_type_set_uses_object_getattribute(w_type, true);
            return None;
        }
        return Some(w_descr);
    }
    None
}

/// typeobject.py:325-326 `has_object_getattribute` — true iff the type
/// inherits `object.__getattribute__` unchanged.  Computing form: looks
/// the descriptor up (memoizing the default in
/// `uses_object_getattribute`) rather than reading the raw flag.
pub(crate) unsafe fn has_object_getattribute(w_type: PyObjectRef) -> bool {
    getattribute_if_not_from_object(w_type).is_none()
}

/// typeobject.py:328-348 `setattr_if_not_from_object` — the `__setattr__`
/// companion of [`getattribute_if_not_from_object`].
pub(crate) unsafe fn setattr_if_not_from_object(w_type: PyObjectRef) -> Option<PyObjectRef> {
    if majit_metainterp::jit::we_are_jitted() {
        if let Some(w_descr) = lookup_in_type_where(w_type, "__setattr__") {
            if is_object_setattr_descr(w_descr) {
                return None;
            }
            return Some(w_descr);
        }
        return None;
    }
    if pyre_object::typeobject::w_type_get_uses_object_setattr(w_type) {
        return None;
    }
    if let Some(w_descr) = lookup_in_type_where(w_type, "__setattr__") {
        if is_object_setattr_descr(w_descr) {
            pyre_object::typeobject::w_type_set_uses_object_setattr(w_type, true);
            return None;
        }
        return Some(w_descr);
    }
    None
}

/// Determine what `self` value to bind for a super-resolved attribute.
///
/// Walks the MRO of `self_obj` starting after `super_type`, finds the
/// raw descriptor for `name`, and returns:
///   - PY_NULL       if it is a staticmethod (no binding)
///   - the class obj if it is a classmethod  (bind class)
///   - `self_obj`    otherwise                (bind instance)
pub unsafe fn super_lookup_binding(
    super_type: PyObjectRef,
    self_obj: PyObjectRef,
    name: &str,
) -> PyObjectRef {
    use pyre_object::*;
    let w_obj_type = if is_instance(self_obj) {
        w_instance_get_type(self_obj)
    } else if is_type(self_obj) {
        self_obj
    } else {
        return self_obj;
    };
    let mro_ptr = w_type_get_mro(w_obj_type);
    if !mro_ptr.is_null() {
        let mro = &*mro_ptr;
        let mut past_super = false;
        for &t in mro {
            if std::ptr::eq(t, super_type) {
                past_super = true;
                continue;
            }
            if !past_super {
                continue;
            }
            if is_type(t) {
                if let Some(raw) = lookup_in_type_where(t, name) {
                    if is_staticmethod(raw) {
                        return PY_NULL;
                    }
                    if is_classmethod(raw) {
                        return w_obj_type;
                    }
                    // `__new__` is implicitly static (type.__new__ is a
                    // builtin_function_or_method, not a Python function)
                    if name == "__new__" {
                        return PY_NULL;
                    }
                    return self_obj;
                }
            }
        }
    }
    self_obj
}

/// C3 linearization — PyPy: typeobject.py `compute_default_mro`.
///
/// Computes the Method Resolution Order for a type following the C3
/// algorithm (Python 2.3+). Handles diamond inheritance correctly.
///
/// Public wrapper for use by isinstance and other external callers.
pub unsafe fn compute_default_mro(w_type: PyObjectRef) -> Vec<PyObjectRef> {
    compute_mro(w_type)
}

pub(crate) unsafe fn compute_mro(w_type: PyObjectRef) -> Vec<PyObjectRef> {
    let mut result = vec![w_type];
    let bases = w_type_get_bases(w_type);
    if bases.is_null() || !is_tuple(bases) {
        return result;
    }
    let n = w_tuple_len(bases);
    if n == 0 {
        return result;
    }

    // Build candidate lists: [base.mro() for base in bases] + [list(bases)]
    // Accept metaclass-created classes too, not just `is_type` ones —
    // ABCMeta's `class Rational(Real): pass` still produces a proper
    // W_TypeObject layout, just with a non-default `ob_type`.
    let mut lists: Vec<Vec<PyObjectRef>> = Vec::with_capacity(n + 1);
    for i in 0..n {
        if let Some(base) = w_tuple_getitem(bases, i as i64) {
            if is_type_like_w(base) {
                lists.push(compute_mro(base));
            }
        }
    }
    let mut bases_list = Vec::with_capacity(n);
    for i in 0..n {
        if let Some(base) = w_tuple_getitem(bases, i as i64) {
            bases_list.push(base);
        }
    }
    lists.push(bases_list);

    // C3 merge
    loop {
        // Remove empty lists
        lists.retain(|l| !l.is_empty());
        if lists.is_empty() {
            break;
        }
        // Find a candidate: head of some list that doesn't appear in
        // the tail of any other list.
        let mut found = None;
        for list in &lists {
            let candidate = list[0];
            let in_tail = lists.iter().any(|other| {
                other.len() > 1 && other[1..].iter().any(|&x| std::ptr::eq(x, candidate))
            });
            if !in_tail {
                found = Some(candidate);
                break;
            }
        }
        let Some(next) = found else {
            // C3 inconsistency — fall back to first available
            break;
        };
        result.push(next);
        // Remove next from the head of all lists
        for list in &mut lists {
            if !list.is_empty() && std::ptr::eq(list[0], next) {
                list.remove(0);
            }
        }
    }
    result
}

// ── Descriptor protocol ──────────────────────────────────────────────
// PyPy equivalent: descroperation.py is_data_descr / space.get

/// Check if a descriptor is a data descriptor (has __set__ or __delete__).
///
/// PyPy: descroperation.py `space.is_data_descr(w_descr)`
///
/// In Python, a data descriptor is any object whose type defines __set__
/// or __delete__. For pyre's current object model, we check the type dict
/// for these names.
/// baseobjspace.py isinstance_w: check if w_obj is instance of w_cls
/// by walking the MRO of type(w_obj) and comparing with w_cls.
pub unsafe fn isinstance_w(w_obj: PyObjectRef, w_cls: PyObjectRef) -> bool {
    let w_obj_type = if is_instance(w_obj) {
        w_instance_get_type(w_obj)
    } else {
        crate::typedef::r#type(w_obj).unwrap_or(pyre_object::PY_NULL)
    };
    if w_obj_type.is_null() {
        return false;
    }
    if std::ptr::eq(w_obj_type, w_cls) {
        return true;
    }
    // Walk MRO
    let mro_ptr = w_type_get_mro(w_obj_type);
    if !mro_ptr.is_null() {
        for &t in &*mro_ptr {
            if std::ptr::eq(t, w_cls) {
                return true;
            }
        }
    }
    false
}

/// pypy/interpreter/baseobjspace.py:419-420 DescrMismatch.
///
/// Construct a DescrMismatch error. Used internally by
/// `descr_self_interp_w`; caught by GetSetProperty.descr_property_get/set/del
/// which then call `descr_call_mismatch` to raise the user-visible TypeError.
#[inline]
pub fn descr_mismatch_error() -> PyError {
    PyError::new(PyErrorKind::DescrMismatch, String::new())
}

/// pypy/interpreter/baseobjspace.py:929-933 ObjSpace.descr_self_interp_w.
///
/// ```python
/// @specialize.arg(1)
/// def descr_self_interp_w(self, RequiredClass, w_obj):
///     if not isinstance(w_obj, RequiredClass):
///         raise DescrMismatch()
///     return w_obj
/// ```
pub fn descr_self_interp_w(
    required_class: PyObjectRef,
    w_obj: PyObjectRef,
) -> Result<PyObjectRef, PyError> {
    if required_class.is_null() {
        return Ok(w_obj);
    }
    if w_obj.is_null() {
        return Err(descr_mismatch_error());
    }
    if !unsafe { isinstance_w(w_obj, required_class) } {
        return Err(descr_mismatch_error());
    }
    Ok(w_obj)
}

/// pypy/interpreter/baseobjspace.py:132-138 W_Root.descr_call_mismatch.
///
/// ```python
/// def descr_call_mismatch(self, space, opname, RequiredClass, args):
///     if RequiredClass is None:
///         classname = '?'
///     else:
///         classname = wrappable_class_name(RequiredClass)
///     raise oefmt(space.w_TypeError,
///                 "'%s' object expected, got '%T' instead", classname, self)
/// ```
///
/// `_opname` is preserved for parity with PyPy's signature even though the
/// error message ignores it (PyPy raises the same TypeError regardless of
/// whether the mismatch came through __getattribute__/__setattr__/__delattr__).
pub fn descr_call_mismatch(
    w_obj: PyObjectRef,
    _opname: &str,
    required_class: PyObjectRef,
) -> PyError {
    let classname: String = if required_class.is_null() {
        "?".to_string()
    } else {
        unsafe { pyre_object::w_type_get_name(required_class).to_string() }
    };
    // PyPy `'%T' % obj` formats space.type(obj).getname(space) — the
    // user-visible class name from `w_obj.w_class`, not the underlying
    // ob_type tag. Pyre's `crate::typedef::r#type` walks the same chain.
    let obj_typename: String = if w_obj.is_null() {
        "NoneType".to_string()
    } else {
        match crate::typedef::r#type(w_obj) {
            Some(tp) => unsafe { pyre_object::w_type_get_name(tp).to_string() },
            None => unsafe { (*(*w_obj).ob_type).name.to_string() },
        }
    };
    PyError::type_error(format!(
        "'{}' object expected, got '{}' instead",
        classname, obj_typename
    ))
}

pub(crate) unsafe fn is_data_descr(descr: PyObjectRef) -> bool {
    if descr.is_null() {
        return false;
    }
    // property objects are always data descriptors
    if is_property(descr) {
        return true;
    }
    // typedef.py:533-540 Member is a data descriptor (__get__, __set__, __delete__)
    if pyre_object::is_member(descr) {
        return true;
    }
    // `typedef.py:312-320 GetSetProperty` is a data descriptor by
    // virtue of always exposing `__set__`/`__delete__` slots in its
    // typedef (regardless of whether `fset`/`fdel` are non-null —
    // `descr_property_set` raises `readonly_attribute` for the
    // null-fset case).  Pyre's GetSetProperty no longer rides on
    // INSTANCE_TYPE so the generic `is_instance + lookup_in_type`
    // branch below would miss it; short-circuit here.
    if pyre_object::typedef::is_getset_property(descr) {
        return true;
    }
    // Check if the descriptor's class has __set__ or __delete__
    if is_instance(descr) {
        let w_type = w_instance_get_type(descr);
        if !w_type.is_null() && is_type(w_type) {
            return lookup_in_type_where(w_type, "__set__").is_some()
                || lookup_in_type_where(w_type, "__delete__").is_some();
        }
    }
    false
}

/// `space.lookup(w_descr, '__delete__') is not None` — whether the
/// descriptor exposes a `__delete__`.  `setattr` consults this after a
/// failed `__set__` lookup to reject a data descriptor that has a deleter
/// but no setter (descroperation.py:124-126).
unsafe fn descr_has_delete(descr: PyObjectRef) -> bool {
    if descr.is_null() {
        return false;
    }
    if is_property(descr)
        || pyre_object::is_member(descr)
        || pyre_object::typedef::is_getset_property(descr)
    {
        return true;
    }
    if let Some(descr_type) = crate::typedef::r#type(descr) {
        return lookup_in_type_where(descr_type, "__delete__").is_some();
    }
    false
}

/// descroperation.py:124-126 — a data descriptor exposing `__delete__`
/// but no `__set__` rejects assignment with this AttributeError.  `%T`
/// renders the descriptor's type name.
// dont_look_inside: read-only-descriptor AttributeError construction; slow path.
#[majit_macros::dont_look_inside]
unsafe fn descr_not_settable_error(descr: PyObjectRef) -> crate::PyError {
    let tp_name = match crate::typedef::r#type(descr) {
        Some(tp) => pyre_object::w_type_get_name(tp).to_string(),
        None => (*(*descr).ob_type).name.to_string(),
    };
    crate::PyError::new(
        crate::PyErrorKind::AttributeError,
        format!("'{tp_name}' object is not a descriptor with set"),
    )
}

/// Call a descriptor's __get__ method.
///
/// PyPy: descroperation.py `space.get(w_descr, w_obj)` →
/// `w_descr.__get__(w_obj, w_type)`
///
/// Returns Some(result) if __get__ was found and called, None otherwise.
/// Call a descriptor's __get__ method.
///
/// PyPy: descroperation.py `space.get(w_descr, w_obj)` →
/// dispatch on descriptor type, then fallback to __get__ MRO lookup.
pub(crate) unsafe fn get(
    descr: PyObjectRef,
    obj: PyObjectRef,
    w_type: PyObjectRef,
) -> Result<Option<PyObjectRef>, crate::PyError> {
    if descr.is_null() {
        return Ok(None);
    }

    // PyPy splits BuiltinFunction from FunctionWithFixedCode at the typedef
    // layer: BuiltinFunction omits __get__, while FunctionWithFixedCode keeps
    // Function.__get__ and binds like a normal method descriptor.
    if crate::is_function(descr) {
        let ob_type = unsafe { (*descr).ob_type };
        if std::ptr::eq(ob_type, &crate::BUILTIN_FUNCTION_TYPE as *const _) {
            return Ok(Some(descr));
        }
        if std::ptr::eq(ob_type, &crate::FUNCTION_TYPE as *const _)
            && crate::is_builtin_code(crate::function_get_code(descr) as pyre_object::PyObjectRef)
        {
            if obj.is_null() || is_none(obj) {
                return Ok(Some(descr));
            }
            return Ok(Some(pyre_object::w_method_new(descr, obj, w_type)));
        }
    }

    // property: PyPy W_Property.get → call fget(obj)
    if is_property(descr) {
        // W_Property.get: `if space.is_w(w_obj, space.w_None): return self`
        // — a `None` receiver (class-level access via `__get__(None, type)`)
        // returns the property itself, the same as a null receiver.
        if obj.is_null() || is_none(obj) {
            return Ok(Some(descr));
        }
        let fget = w_property_get_fget(descr);
        if fget.is_null() || is_none(fget) {
            return Err(property_no_accessor(descr, obj, "getter"));
        }
        // W_Property.get → `space.call_function(self.fget, w_obj)`: the
        // getter's exception propagates rather than being swallowed.
        return Ok(Some(crate::call::call_function_impl_result(fget, &[obj])?));
    }

    // typedef.py:504-516 Member.descr_member_get:
    //   if space.is_w(w_obj, space.w_None): return self
    //   self.typecheck(space, w_obj)
    //   w_result = w_obj.getslotvalue(self.index)
    //   if w_result is None: raise AttributeError(...)
    //   return w_result
    if pyre_object::is_member(descr) {
        // typedef.py:507-508
        if obj.is_null() || is_none(obj) {
            return Ok(Some(descr));
        }
        // typedef.py:510: self.typecheck(space, w_obj) → TypeError
        let w_cls = pyre_object::w_member_get_cls(descr);
        if !w_cls.is_null() && is_type(w_cls) && !isinstance_w(obj, w_cls) {
            let slot_name = pyre_object::w_member_get_name(descr);
            return Err(crate::PyError::type_error(format!(
                "descriptor '{}' for '{}' objects doesn't apply to '{}' object",
                slot_name,
                pyre_object::w_type_get_name(w_cls),
                (*(*obj).ob_type).name,
            )));
        }
        // typedef.py:511: w_result = w_obj.getslotvalue(self.index)
        let index = pyre_object::w_member_get_index(descr);
        let found = if is_instance(obj) {
            crate::objspace::std::mapdict::getslotvalue(obj, index)
        } else {
            // Native-layout subclass instance — slot backed by __dict__.
            native_slot_get(obj, pyre_object::w_member_get_name(descr))
        };
        // typedef.py:512-516: if w_result is None: raise
        // AttributeError("'%T' object has no attribute '%s'")
        if found.is_none() {
            let slot_name = pyre_object::w_member_get_name(descr);
            return Err(raiseattrerror(obj, slot_name));
        }
        return Ok(found);
    }

    // `function.py:691-693 StaticMethod.descr_staticmethod_get` and
    // `function.py:738-748 ClassMethod.descr_classmethod_get` are
    // bound through their typedef `__get__` entries
    // (`typedef.py:866, 883`) in `init_staticmethod_type` /
    // `init_classmethod_type`.  The previous hardcoded fast-path here
    // pre-dated the typedef registration; the generic fallback below
    // now reaches them through `lookup_in_type_where(descr_type,
    // '__get__')`.

    // General __get__: look up __get__ on the descriptor's own type MRO
    if let Some(descr_type) = crate::typedef::r#type(descr) {
        if let Some(get_fn) = lookup_in_type_where(descr_type, "__get__") {
            if !get_fn.is_null() {
                let result = crate::call::call_function_impl_result(get_fn, &[descr, obj, w_type])?;
                return Ok(Some(result));
            }
        }
    }
    Ok(None)
}

/// Call a descriptor's __set__ method.
///
/// PyPy: descroperation.py `descr__setattr__` →
/// `space.get_and_call_function(w_set, w_descr, w_obj, w_value)`
unsafe fn set(
    descr: PyObjectRef,
    obj: PyObjectRef,
    value: PyObjectRef,
) -> Result<bool, crate::PyError> {
    if descr.is_null() {
        return Ok(false);
    }

    // property: PyPy W_Property.set → call_function(fset, obj, value).
    // Read-only properties (no `fset` / `@x.setter` never registered)
    // raise AttributeError ("can't set attribute") rather than falling
    // through to the instance dict (`descrobject.c property_descr_set`,
    // mirrored at `pypy/module/__builtin__/descriptor.py W_Property.set`).
    if is_property(descr) {
        let fset = w_property_get_fset(descr);
        if fset.is_null() || is_none(fset) {
            return Err(property_no_accessor(descr, obj, "setter"));
        }
        // descriptor.py:228 W_Property.set → `space.call_function(self.fset,
        // w_obj, w_value)`: the setter's exception propagates rather than
        // being swallowed.
        crate::call::call_function_impl_result(fset, &[obj, value])?;
        return Ok(true);
    }

    // typedef.py:518-522 Member.descr_member_set:
    //   self.typecheck(space, w_obj)
    //   w_obj.setslotvalue(self.index, w_value)
    if pyre_object::is_member(descr) {
        // typedef.py:521: self.typecheck(space, w_obj) → TypeError
        let w_cls = pyre_object::w_member_get_cls(descr);
        if !w_cls.is_null() && is_type(w_cls) && !isinstance_w(obj, w_cls) {
            let slot_name = pyre_object::w_member_get_name(descr);
            return Err(crate::PyError::type_error(format!(
                "descriptor '{}' for '{}' objects doesn't apply to '{}' object",
                slot_name,
                pyre_object::w_type_get_name(w_cls),
                (*(*obj).ob_type).name,
            )));
        }
        // typedef.py:522: w_obj.setslotvalue(self.index, w_value)
        let index = pyre_object::w_member_get_index(descr);
        if is_instance(obj) {
            crate::objspace::std::mapdict::setslotvalue(obj, index, value);
        } else {
            // Native-layout subclass instance — slot backed by __dict__.
            let slot_name = pyre_object::w_member_get_name(descr);
            if !native_slot_set(obj, slot_name, value) {
                return Err(crate::PyError::new(
                    crate::PyErrorKind::AttributeError,
                    format!(
                        "'{}' object attribute '{}' is read-only",
                        (*(*obj).ob_type).name,
                        slot_name,
                    ),
                ));
            }
        }
        return Ok(true);
    }

    // General __set__: look up on descriptor's type MRO.  GetSetProperty
    // is no longer INSTANCE_TYPE-shaped (it carries `GETSET_DESCRIPTOR
    // _TYPE` so its GetSetProperty payload is GC-traced), so resolve
    // the type through `crate::typedef::r#type` rather than the
    // `is_instance` branch.
    let descr_type = if pyre_object::typedef::is_getset_property(descr) {
        crate::typedef::r#type(descr).unwrap_or(std::ptr::null_mut())
    } else if is_instance(descr) {
        w_instance_get_type(descr)
    } else {
        std::ptr::null_mut()
    };
    if !descr_type.is_null() {
        if let Some(set_fn) = lookup_in_type_where(descr_type, "__set__") {
            if !set_fn.is_null() {
                crate::call::call_function_impl_result(set_fn, &[descr, obj, value])?;
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// Call a descriptor's __delete__ method.
///
/// descroperation.py `space.delete(w_descr, w_obj)`
unsafe fn delete(descr: PyObjectRef, obj: PyObjectRef) -> Result<(), crate::PyError> {
    // property: call fdel(obj)
    if is_property(descr) {
        let fdel = w_property_get_fdel(descr);
        if fdel.is_null() || is_none(fdel) {
            return Err(property_no_accessor(descr, obj, "deleter"));
        }
        crate::call::call_function_impl_result(fdel, &[obj])?;
        return Ok(());
    }
    // typedef.py:524-531 Member.descr_member_del
    if pyre_object::is_member(descr) {
        let w_cls = pyre_object::w_member_get_cls(descr);
        if !w_cls.is_null() && is_type(w_cls) && !isinstance_w(obj, w_cls) {
            let slot_name = pyre_object::w_member_get_name(descr);
            return Err(crate::PyError::type_error(format!(
                "descriptor '{}' for '{}' objects doesn't apply to '{}' object",
                slot_name,
                pyre_object::w_type_get_name(w_cls),
                (*(*obj).ob_type).name,
            )));
        }
        // typedef.py:527-531: success = w_obj.delslotvalue(self.index)
        let index = pyre_object::w_member_get_index(descr);
        let removed = if is_instance(obj) {
            crate::objspace::std::mapdict::delslotvalue(obj, index)
        } else {
            // Native-layout subclass instance — slot backed by __dict__.
            native_slot_del(obj, pyre_object::w_member_get_name(descr))
        };
        if !removed {
            let slot_name = pyre_object::w_member_get_name(descr);
            return Err(crate::PyError::new(
                crate::PyErrorKind::AttributeError,
                slot_name.to_string(),
            ));
        }
        return Ok(());
    }
    // General __delete__: look up on descriptor's type MRO — same
    // shape as `set` above (resolve type through `r#type` so non-
    // INSTANCE_TYPE descriptors like `GetSetProperty` are reached).
    let descr_type = if pyre_object::typedef::is_getset_property(descr) {
        crate::typedef::r#type(descr).unwrap_or(std::ptr::null_mut())
    } else if is_instance(descr) {
        w_instance_get_type(descr)
    } else {
        std::ptr::null_mut()
    };
    if !descr_type.is_null() {
        if let Some(del_fn) = lookup_in_type_where(descr_type, "__delete__") {
            if !del_fn.is_null() {
                crate::call::call_function_impl_result(del_fn, &[descr, obj])?;
                return Ok(());
            }
        }
    }
    Err(crate::PyError::new(
        crate::PyErrorKind::AttributeError,
        "cannot delete attribute".to_string(),
    ))
}

/// Set an attribute on an object: `obj.name = value`.
///
/// Stores the attribute in the per-object side table.
/// PyPy: descroperation.py descr__setattr__

/// objectobject.py:137-154 `descr_set___class__(space, w_obj, w_newcls)`.
///
/// Validates and performs `obj.__class__ = newcls`.
fn descr_set___class__(w_obj: PyObjectRef, w_newcls: PyObjectRef) -> PyResult {
    unsafe {
        // objectobject.py:139-142 — w_newcls must be a W_TypeObject
        if !is_type(w_newcls) {
            return Err(crate::PyError::type_error(format!(
                "__class__ must be set to new-style class, not '{}' object",
                (*(*w_newcls).ob_type).name,
            )));
        }
        // objectobject.py:143-145 — w_newcls must be a heap type.
        if !w_type_is_heaptype(w_newcls) {
            return Err(crate::PyError::type_error(
                "__class__ assignment: only for heap types".to_string(),
            ));
        }
        // objectobject.py:146-147 — get the old class
        let w_oldcls = match crate::typedef::r#type(w_obj) {
            Some(c) => c,
            None => {
                return Err(crate::PyError::type_error(
                    "__class__ assignment: cannot determine current class".to_string(),
                ));
            }
        };
        // objectobject.py:148-154 — get_full_instance_layout() must match.
        // typeobject.py:125-129 Layout.expand() compares 5-tuple:
        //   (typedef, newslotnames, base_layout, hasdict, weakrefable)
        let layouts_compatible = pyre_object::typeobject::Layout::expands_equal(
            pyre_object::w_type_get_layout_ptr(w_oldcls),
            pyre_object::w_type_get_hasdict(w_oldcls),
            pyre_object::w_type_get_weakrefable(w_oldcls),
            pyre_object::w_type_get_layout_ptr(w_newcls),
            pyre_object::w_type_get_hasdict(w_newcls),
            pyre_object::w_type_get_weakrefable(w_newcls),
        );
        if !layouts_compatible {
            return Err(crate::PyError::type_error(format!(
                "__class__ assignment: '{}' object layout differs from '{}'",
                pyre_object::w_type_get_name(w_oldcls),
                pyre_object::w_type_get_name(w_newcls),
            )));
        }
        // objectobject.py:150 — w_obj.setclass(space, w_newcls).  For a mapdict
        // instance this re-roots the map chain onto the new class's terminator
        // (mapdict.py:754-756); pyre then keeps w_class authoritative for type().
        if pyre_object::is_instance(w_obj) {
            crate::objspace::std::mapdict::instance_setclass(w_obj, w_newcls);
        }
        (*w_obj).w_class = w_newcls;
    }
    Ok(w_none())
}

pub fn setattr_str(obj: PyObjectRef, name: &str, value: PyObjectRef) -> PyResult {
    let value = unwrap_cell(value);
    let obj = crate::module::_weakref::interp__weakref::force(obj)?;
    // descroperation.py:247 — space.lookup for __setattr__ through MRO,
    // then get_and_call_function which applies descriptor binding.
    unsafe {
        if is_instance(obj) {
            let w_type = w_instance_get_type(obj);
            // objspace.py:721-723 — dispatch only to a non-default
            // `__setattr__`; the object default falls through to the
            // inlined `object_setattr` fast path (the default slot is
            // `object.__setattr__` → `object_setattr`, typedef.rs:7030,
            // so skipping the descriptor call is equivalent).
            if let Some(sa) = setattr_if_not_from_object(w_type) {
                let w_name = w_str_new(name);
                return crate::call::call_function_impl_result(sa, &[obj, w_name, value])
                    .map(|_| w_none());
            }
        } else if let Some(w_type) = crate::typedef::r#type(obj) {
            // descroperation.py:247 looks up __setattr__ on the receiver
            // type regardless of receiver kind.  Non-instance receivers
            // (e.g. structseq tuple subclasses) may install a non-default
            // __setattr__; only a real override (≠ object.__setattr__)
            // needs invoking — the default terminal path is object_setattr.
            if let Some(sa) = lookup_in_type(w_type, "__setattr__") {
                let is_default = lookup_in_type(crate::typedef::w_object(), "__setattr__")
                    .is_some_and(|d| std::ptr::eq(sa, d));
                if !is_default {
                    let w_name = w_str_new(name);
                    return crate::call::call_function_impl_result(sa, &[obj, w_name, value])
                        .map(|_| w_none());
                }
            }
        }
    }
    object_setattr(obj, name, value)
}

/// `objectobject.py descr__setattr__` — the terminal implementation
/// that bypasses user `__setattr__` overrides and writes directly
/// through the descriptor / instance-dict path.  Called by
/// `object.__setattr__` and as the default path in `setattr`.
pub fn object_setattr(obj: PyObjectRef, name: &str, value: PyObjectRef) -> PyResult {
    let value = unwrap_cell(value);
    let obj = crate::module::_weakref::interp__weakref::force(obj)?;
    // Data descriptor __set__ takes priority (PyPy: descroperation.py
    // descr__setattr__ step 1). PyPy walks `space.type(obj)` regardless of
    // whether `obj` is a Python-level instance, so the lookup must run for
    // every object whose type pyre can resolve — not just W_ObjectObject.
    unsafe {
        let w_type = if is_instance(obj) {
            w_instance_get_type(obj)
        } else if is_type(obj) {
            // For type objects pyre stores attributes in the type's own
            // dict below; the descriptor walk uses the metaclass MRO so
            // metatype-installed setters (e.g. on `type`) still fire.
            crate::typedef::r#type(obj).unwrap_or(std::ptr::null_mut())
        } else {
            crate::typedef::r#type(obj).unwrap_or(std::ptr::null_mut())
        };
        if !w_type.is_null() {
            if let Some(descr) = lookup_in_type_where(w_type, name) {
                if set(descr, obj, value)? {
                    return Ok(w_none());
                }
                // descroperation.py:124-126 — `__delete__` but no `__set__`
                // is a read-only data descriptor; reject rather than shadow
                // it with an instance/type dict store.
                if descr_has_delete(descr) {
                    return Err(descr_not_settable_error(descr));
                }
            }
        }
    }
    // Type objects: store in the type's own namespace (class dict).
    // PyPy: typeobject.py type.__setattr__ → w_type.dict_w[name] = w_value
    unsafe {
        if is_type(obj) {
            // typeobject.py:416 — only heap types may have their dict mutated.
            if !pyre_object::w_type_is_heaptype(obj) {
                return Err(PyError::type_error(format!(
                    "cannot set '{}' attribute of immutable type '{}'",
                    name,
                    w_type_get_name(obj)
                )));
            }
            let dict_ptr = w_type_get_dict_ptr(obj) as *mut crate::DictStorage;
            if !dict_ptr.is_null() {
                crate::dict_storage_store(&mut *dict_ptr, name, value);
                // typeobject.py:430 — `self.mutated(name)` after the
                // dict_w write so cached `compares_by_identity_status`
                // (and future per-type caches) reset on this type and
                // every entry in `weak_subclasses` recursively.
                mutated(obj, Some(name));
                return Ok(w_none());
            }
        }
    }
    // objectobject.py:137-154 descr_set___class__
    if name == "__class__" {
        return descr_set___class__(obj, value);
    }
    // descroperation.py:108-123 Object.descr__setattr__:
    //
    //     def descr__setattr__(space, w_obj, w_name, w_value):
    //         name = space.text_w(w_name)
    //         w_descr = space.lookup(w_obj, name)
    //         if w_descr is not None:
    //             w_set = space.lookup(w_descr, '__set__')
    //             if w_set is not None:
    //                 return space.get_and_call_function(w_set, w_descr, w_obj, w_value)
    //             if space.lookup(w_descr, '__delete__') is not None:
    //                 raise oefmt(space.w_AttributeError,
    //                             "'%T' object is not a descriptor with set", w_descr)
    //         if w_obj.setdictvalue(space, name, w_value):
    //             return
    //         raiseattrerror(space, w_obj, name, w_descr)
    //
    // The descriptor + type short-circuits above already handle the
    // first half of this. What remains is `setdictvalue` + raiseattrerror,
    // both at the function tail — the exception arms below emulate the
    // GetSetProperty descriptors of W_BaseException.typedef and so must
    // run before `setdictvalue`, like a descriptor `__set__` would.
    //
    // Module objects: PyPy `module.py:Module` does not override
    // `descr__setattr__`, so the call reaches `setdictvalue` (`W_Root.
    // setdictvalue` → `Module.getdict` → `self.w_dict`) only after the
    // data-descriptor walk above.  pyre's `getdict` does not surface the
    // module dict, so store it explicitly here at the `setdictvalue`
    // position — never before the descriptor walk, or a data descriptor
    // on the module's type (e.g. the read-only `__dict__` getset) would
    // be shadowed by the namespace store.  `setitem` is the generic
    // dispatch: an exact `W_DictMultiObject` goes direct, a dict subclass
    // (`moduledef.py:102-103` user-supplied `__builtins__`) routes through
    // its `__setitem__`.  A module is neither an exception nor a property,
    // so this early return is order-independent from the arms below; the
    // shared `setdictvalue` for every other object stays at the tail.
    unsafe {
        if is_module(obj) {
            let w_dict = pyre_object::w_module_get_w_dict(obj);
            if !w_dict.is_null() {
                setitem(w_dict, w_str_new(name), value)?;
                return Ok(w_none());
            }
        }
    }
    // descriptor.py:316-318 `__doc__ = GetSetProperty(W_Property.get_doc,
    // W_Property.set_doc)` — the only writable property attribute;
    // `property.__doc__ = "..."` is common in stdlib (dis.py, etc.).
    // Other names (and member descriptors, which expose no setter,
    // typedef.py:533-540) fall through to raiseattrerror.
    unsafe {
        if is_property(obj) {
            if name == "__doc__" {
                pyre_object::descriptor::w_property_set_doc(obj, value);
                return Ok(w_none());
            }
            if name == "__name__" {
                pyre_object::descriptor::w_property_set_name(obj, value);
                return Ok(w_none());
            }
        }
    }
    // Exception instances accept arbitrary attribute writes —
    // `pypy/module/exceptions/interp_exceptions.py` declares
    // W_BaseException.typedef with `__dict__ = GetSetProperty(descr_get_dict)`,
    // so user code routinely does `e.foo = bar` (e.g.
    // `argparse.ArgumentTypeError`'s `e.message = ...` pattern).
    // Non-special names land in the lazily allocated instance dict on
    // `W_BaseException.w_dict` (interp_exceptions.py:113, 222-225).
    if unsafe { pyre_object::is_exception(obj) } {
        // `pypy/module/exceptions/interp_exceptions.py:156-157
        // W_BaseException.descr_setargs` →
        //   self.args_w = space.fixedview(w_newargs)
        // `space.fixedview` materialises any iterable into a list of
        // wrapped objects; pyre stores `args_w` as a tuple `PyObjectRef`,
        // so coerce the incoming value into a tuple shape (tuple stays
        // as-is, list wraps into tuple, anything else iterates).
        if name == "args" {
            let coerced = unsafe { coerce_to_list_for_args(value)? };
            unsafe { pyre_object::interp_exceptions::w_exception_set_args(obj, coerced) };
            return Ok(w_none());
        }
        // `interp_exceptions.py:165-219` — the four special exception
        // attributes (`__cause__`, `__context__`, `__traceback__`,
        // `__suppress_context__`) are registered as `GetSetProperty`
        // setters on `W_BaseException.typedef` and each validates its
        // input before storing into the matching typed slot
        // (`w_cause`/`w_context`/`w_traceback`/`suppress_context`,
        // line 113-117).  Storage lives on `W_BaseException`
        // directly — no side store for these four names.
        match name {
            "__dict__" => {
                // `interp_exceptions.py:293` registers
                // `__dict__ = GetSetProperty(descr_get_dict, descr_set_dict)`
                // whose setter routes to `setdict` (typedef.py
                // descr_set_dict) — replaces the whole instance dict.
                setdict(obj, value)?;
                return Ok(w_none());
            }
            "__cause__" => {
                // `interp_exceptions.py:166-174 descr_setcause` — None
                // OR an instance whose type derives from `BaseException`,
                // and always flips `suppress_context` to True.
                if !unsafe { pyre_object::is_none(value) } {
                    let value_type = crate::typedef::r#type(value).unwrap_or(pyre_object::PY_NULL);
                    if value_type.is_null() || !unsafe { exception_is_valid_class_w(value_type) } {
                        return Err(PyError::type_error(
                            "exception cause must be None or derive from BaseException",
                        ));
                    }
                }
                unsafe {
                    pyre_object::interp_exceptions::w_exception_set_cause(obj, value);
                    pyre_object::interp_exceptions::w_exception_set_suppress_context(obj, true);
                };
                return Ok(w_none());
            }
            "__context__" => {
                // `interp_exceptions.py:183-190 descr_setcontext` — None
                // OR an instance whose type derives from `BaseException`.
                if !unsafe { pyre_object::is_none(value) } {
                    let value_type = crate::typedef::r#type(value).unwrap_or(pyre_object::PY_NULL);
                    if value_type.is_null() || !unsafe { exception_is_valid_class_w(value_type) } {
                        return Err(PyError::type_error(
                            "exception context must be None or derive from BaseException",
                        ));
                    }
                }
                unsafe { pyre_object::interp_exceptions::w_exception_set_context(obj, value) };
                return Ok(w_none());
            }
            "__traceback__" => {
                // `interp_exceptions.py:202-206 descr_settraceback` —
                // accept None or PyTraceback only.  Now that real
                // PyTraceback exists, narrow the type check to the
                // exact pair PyPy accepts; reject everything else as
                // TypeError per PyPy.
                let accept = unsafe {
                    pyre_object::is_none(value) || crate::pytraceback::is_pytraceback(value)
                };
                if !accept {
                    return Err(PyError::type_error(
                        "__traceback__ must be a traceback or None",
                    ));
                }
                let stored = if unsafe { pyre_object::is_none(value) } {
                    pyre_object::PY_NULL
                } else {
                    value
                };
                unsafe { pyre_object::interp_exceptions::w_exception_set_traceback(obj, stored) };
                return Ok(w_none());
            }
            "__suppress_context__" => {
                // `interp_exceptions.py:215-216 descr_setsuppresscontext`
                // — `space.bool_w(w_value)` coerces via `__bool__`.
                let b = is_true(value)?;
                unsafe { pyre_object::interp_exceptions::w_exception_set_suppress_context(obj, b) };
                return Ok(w_none());
            }
            // `interp_exceptions.py:468-471`
            // `readwrite_attrproperty_w('w_object', W_UnicodeTranslateError)`
            // and `:1081-1083` / `:1201-1203` for Decode / Encode.
            // PyPy's `attrproperty_w` writer stores the raw `w_value`
            // into the slot with no type coercion — that matches the
            // direct slot write here.  Gated on the three Unicode*Error
            // kinds because PyPy installs these descriptors only on
            // those typedefs.
            "object" => {
                let kind = unsafe { pyre_object::w_exception_get_kind(obj) };
                if matches!(
                    kind,
                    pyre_object::interp_exceptions::ExcKind::UnicodeTranslateError
                        | pyre_object::interp_exceptions::ExcKind::UnicodeDecodeError
                        | pyre_object::interp_exceptions::ExcKind::UnicodeEncodeError
                ) {
                    unsafe { pyre_object::interp_exceptions::w_exception_set_object(obj, value) };
                    return Ok(w_none());
                }
            }
            "start" => {
                let kind = unsafe { pyre_object::w_exception_get_kind(obj) };
                if matches!(
                    kind,
                    pyre_object::interp_exceptions::ExcKind::UnicodeTranslateError
                        | pyre_object::interp_exceptions::ExcKind::UnicodeDecodeError
                        | pyre_object::interp_exceptions::ExcKind::UnicodeEncodeError
                ) {
                    unsafe { pyre_object::interp_exceptions::w_exception_set_start(obj, value) };
                    return Ok(w_none());
                }
            }
            "end" => {
                let kind = unsafe { pyre_object::w_exception_get_kind(obj) };
                if matches!(
                    kind,
                    pyre_object::interp_exceptions::ExcKind::UnicodeTranslateError
                        | pyre_object::interp_exceptions::ExcKind::UnicodeDecodeError
                        | pyre_object::interp_exceptions::ExcKind::UnicodeEncodeError
                ) {
                    unsafe { pyre_object::interp_exceptions::w_exception_set_end(obj, value) };
                    return Ok(w_none());
                }
            }
            "reason" => {
                let kind = unsafe { pyre_object::w_exception_get_kind(obj) };
                if matches!(
                    kind,
                    pyre_object::interp_exceptions::ExcKind::UnicodeTranslateError
                        | pyre_object::interp_exceptions::ExcKind::UnicodeDecodeError
                        | pyre_object::interp_exceptions::ExcKind::UnicodeEncodeError
                ) {
                    unsafe { pyre_object::interp_exceptions::w_exception_set_reason(obj, value) };
                    return Ok(w_none());
                }
            }
            "encoding" => {
                // `interp_exceptions.py:1080 W_UnicodeDecodeError.encoding`
                // / `:1200 W_UnicodeEncodeError.encoding`.  Translate has
                // no encoding attrproperty per `:461-471` typedef.
                let kind = unsafe { pyre_object::w_exception_get_kind(obj) };
                if matches!(
                    kind,
                    pyre_object::interp_exceptions::ExcKind::UnicodeDecodeError
                        | pyre_object::interp_exceptions::ExcKind::UnicodeEncodeError
                ) {
                    unsafe { pyre_object::interp_exceptions::w_exception_set_encoding(obj, value) };
                    return Ok(w_none());
                }
            }
            // `interp_exceptions.py:739-742` —
            // `readwrite_attrproperty_w('w_errno' / 'w_strerror' /
            // 'w_filename' / 'w_filename2', W_OSError)`.  The
            // `attrproperty_w` writer stores the raw `w_value` into the
            // slot; the matching getattr arm reads it back ahead of the
            // `args_w`-derived fallback.  Gated on the OSError family
            // (OSError / FileNotFoundError) because PyPy installs these
            // descriptors only on `W_OSError.typedef`.
            "errno" | "strerror" | "filename" | "filename2" => {
                let kind = unsafe { pyre_object::w_exception_get_kind(obj) };
                if matches!(
                    kind,
                    pyre_object::interp_exceptions::ExcKind::OSError
                        | pyre_object::interp_exceptions::ExcKind::FileNotFoundError
                ) {
                    unsafe {
                        match name {
                            "errno" => {
                                pyre_object::interp_exceptions::w_exception_set_errno(obj, value)
                            }
                            "strerror" => {
                                pyre_object::interp_exceptions::w_exception_set_strerror(obj, value)
                            }
                            "filename" => {
                                pyre_object::interp_exceptions::w_exception_set_filename(obj, value)
                            }
                            _ => pyre_object::interp_exceptions::w_exception_set_filename2(
                                obj, value,
                            ),
                        }
                    };
                    return Ok(w_none());
                }
            }
            // `interp_exceptions.py:1006
            // readwrite_attrproperty_w('w_code', W_SystemExit)` — the
            // writer stores the raw `w_value` into the slot; the matching
            // getattr arm reads it back ahead of the `args_w`-derived
            // fallback.  Gated on SystemExit because PyPy installs the
            // descriptor only on `W_SystemExit.typedef`.
            "code" => {
                let kind = unsafe { pyre_object::w_exception_get_kind(obj) };
                if kind == pyre_object::interp_exceptions::ExcKind::SystemExit {
                    unsafe { pyre_object::interp_exceptions::w_exception_set_code(obj, value) };
                    return Ok(w_none());
                }
            }
            // `interp_exceptions.py:679-681 W_ImportError` writable
            // `msg` / `name` / `path` (plus `name_from`) slots; the
            // matching getattr arm reads them back.  Gated on the
            // ImportError-family kind (ImportError / ModuleNotFoundError).
            // `name` is handled by the shared arm below.
            "msg" | "path" | "name_from" => {
                let kind = unsafe { pyre_object::w_exception_get_kind(obj) };
                if matches!(
                    kind,
                    pyre_object::interp_exceptions::ExcKind::ImportError
                        | pyre_object::interp_exceptions::ExcKind::ModuleNotFoundError
                ) {
                    unsafe {
                        match name {
                            "msg" => pyre_object::interp_exceptions::w_exception_set_import_msg(
                                obj, value,
                            ),
                            "path" => pyre_object::interp_exceptions::w_exception_set_import_path(
                                obj, value,
                            ),
                            _ => pyre_object::interp_exceptions::w_exception_set_import_name_from(
                                obj, value,
                            ),
                        }
                    };
                    return Ok(w_none());
                }
            }
            // Shared writable `name` slot for ImportError / ModuleNotFoundError
            // / NameError / AttributeError; the matching getattr arm reads it
            // back.
            "name" => {
                let kind = unsafe { pyre_object::w_exception_get_kind(obj) };
                if matches!(
                    kind,
                    pyre_object::interp_exceptions::ExcKind::ImportError
                        | pyre_object::interp_exceptions::ExcKind::ModuleNotFoundError
                        | pyre_object::interp_exceptions::ExcKind::NameError
                        | pyre_object::interp_exceptions::ExcKind::AttributeError
                ) {
                    unsafe { pyre_object::interp_exceptions::w_exception_set_name(obj, value) };
                    return Ok(w_none());
                }
            }
            // Writable `obj` slot (W_AttributeError).
            "obj" => {
                let kind = unsafe { pyre_object::w_exception_get_kind(obj) };
                if kind == pyre_object::interp_exceptions::ExcKind::AttributeError {
                    unsafe { pyre_object::interp_exceptions::w_exception_set_attr_obj(obj, value) };
                    return Ok(w_none());
                }
            }
            _ => {}
        }
    }
    // descroperation.py:121-122 `if w_obj.setdictvalue(space, name, w_value):
    // return` — exception extras land in the lazily allocated
    // `W_BaseException.w_dict` (interp_exceptions.py:113, 222-225) via
    // the `getdict` exception arm.
    if setdictvalue(obj, name, value) {
        return Ok(w_none());
    }
    Err(raiseattrerror(obj, name))
}

/// `pypy/module/exceptions/interp_exceptions.py:156-157
/// W_BaseException.descr_setargs` parity helper:
///
/// ```python
/// def descr_setargs(self, space, w_newargs):
///     self.args_w = space.fixedview(w_newargs)
/// ```
///
/// `space.fixedview` materialises any iterable into a RPython list
/// of `W_Root`; pyre stores `args_w` as a `W_ListObject` so the
/// getter (`w_exception_get_args`) can build a fresh tuple per read
/// (matching `descr_getargs: return space.newtuple(self.args_w)`).
unsafe fn coerce_to_list_for_args(value: PyObjectRef) -> Result<PyObjectRef, PyError> {
    if value.is_null() {
        return Ok(w_list_new(vec![]));
    }
    let items = fixedview(value, -1)?;
    Ok(w_list_new(items))
}

/// baseobjspace.py:52-57 W_Root.setdictvalue (default).
///
/// ```python
/// def setdictvalue(self, space, attr, w_value):
///     w_dict = self.getdict(space)
///     if w_dict is not None:
///         space.setitem_str(w_dict, attr, w_value)
///         return True
///     return False
/// ```
pub(crate) fn setdictvalue(obj: PyObjectRef, name: &str, value: PyObjectRef) -> bool {
    let w_dict = getdict_backing(obj);
    if w_dict.is_null() {
        return false;
    }
    // For a user instance, `getdict` returns the MapDictStrategy view, so this
    // `setitem_str` routes straight to the instance map+storage
    // (MapDictStrategy.setitem_str → setdictvalue → map.write DICT,
    // mapdict.py:849-850). The earlier C1 explicit `instance_node_setdictvalue`
    // dual-write is now subsumed by that routing and removed.
    unsafe { pyre_object::w_dict_setitem_str(w_dict, name, value) };
    true
}

/// descroperation.py:63-69 raiseattrerror.
///
/// ```python
/// def raiseattrerror(space, w_obj, name, w_descr=None):
///     if w_descr is None:
///         raise oefmt(space.w_AttributeError,
///                     "'%T' object has no attribute '%s'", w_obj, name)
///     else:
///         raise oefmt(space.w_AttributeError,
///                     "'%T' object attribute '%s' is read-only", w_obj, name)
/// ```
// dont_look_inside: attribute-miss / read-only AttributeError construction; slow path.
#[majit_macros::dont_look_inside]
fn raiseattrerror(obj: PyObjectRef, name: &str) -> PyError {
    // descroperation.py:58-64 — a type receiver reports its own name through
    // the `type object '%N'` form; every other object reports its type's name
    // through the `'%T' object` form.
    let subject = unsafe {
        if is_type(obj) {
            format!("type object '{}'", pyre_object::w_type_get_name(obj))
        } else {
            let tp_name = match crate::typedef::r#type(obj) {
                Some(tp) => pyre_object::w_type_get_name(tp).to_string(),
                None => (*(*obj).ob_type).name.to_string(),
            };
            format!("'{}' object", tp_name)
        }
    };
    PyError::attribute_error_with_context(
        format!("{} has no attribute '{}'", subject, name),
        obj,
        name,
    )
}

/// Delete an attribute: `del obj.name`.
///
/// PyPy: descroperation.py descr__delattr__
pub fn delattr_str(obj: PyObjectRef, name: &str) -> PyResult {
    let obj = crate::module::_weakref::interp__weakref::force(obj)?;
    // descroperation.py:254 — space.lookup for __delattr__ through MRO
    unsafe {
        if is_instance(obj) {
            let w_type = w_instance_get_type(obj);
            if let Some(da) = lookup_in_type(w_type, "__delattr__") {
                let w_name = w_str_new(name);
                return crate::call::call_function_impl_result(da, &[obj, w_name])
                    .map(|_| w_none());
            }
        }
    }
    object_delattr(obj, name)
}

/// Terminal `object.__delattr__` — bypasses user override.
pub fn object_delattr(obj: PyObjectRef, name: &str) -> PyResult {
    let obj = crate::module::_weakref::interp__weakref::force(obj)?;
    // `property.__name__` is a writable/deletable slot; deleting clears
    // the name recorded by `__set_name__`, and deleting when unset
    // raises like a missing attribute.
    unsafe {
        if is_property(obj) && name == "__name__" {
            let w_name = pyre_object::descriptor::w_property_get_name(obj);
            if w_name.is_null() {
                return Err(crate::PyError::attribute_error(
                    "'property' object has no attribute '__name__'",
                ));
            }
            pyre_object::descriptor::w_property_set_name(obj, pyre_object::PY_NULL);
            return Ok(w_none());
        }
    }
    // descroperation.py:131-140 descr__delattr__: a data descriptor's
    // `__delete__` takes priority over the namespace delete. PyPy walks
    // `space.type(obj)`, so the lookup must run for any object whose type
    // pyre can resolve — not just W_ObjectObject — and before the
    // module/type/instance dict removal below.
    unsafe {
        let w_type = if is_instance(obj) {
            w_instance_get_type(obj)
        } else if is_type(obj) {
            crate::typedef::r#type(obj).unwrap_or(std::ptr::null_mut())
        } else {
            crate::typedef::r#type(obj).unwrap_or(std::ptr::null_mut())
        };
        if !w_type.is_null() {
            if let Some(descr) = lookup_in_type_where(w_type, name) {
                if is_data_descr(descr) {
                    delete(descr, obj)?;
                    return Ok(w_none());
                }
            }
        }
    }
    // Module objects: PyPy `module.py:Module` does not override
    // `descr__delattr__`, so the call falls through to W_Root's
    // `deldictvalue` (`baseobjspace.py:58-67`):
    //
    //     w_dict = self.getdict(space)
    //     if w_dict is not None:
    //         try: space.delitem(w_dict, space.newtext(attr))
    //         except KeyError: ...
    //
    // `space.delitem` is the generic dispatch: exact W_DictObject
    // goes direct, dict subclass (moduledef.py:102-103
    // user-supplied `__builtins__`) routes through the subclass's
    // `__delitem__`.  KeyError is swallowed (returning False from
    // `deldictvalue`); pyre falls through to `raiseattrerror` at
    // the end of the function for the same observable behaviour.
    //
    //     def deldictvalue(self, space, attr):
    //         w_dict = self.getdict(space)
    //         if w_dict is not None:
    //             try:
    //                 space.delitem(w_dict, space.newtext(attr))
    //                 return True
    //             except OperationError as ex:
    //                 if not ex.match(space, space.w_KeyError):
    //                     raise
    //         return False
    unsafe {
        if is_module(obj) {
            let w_dict = pyre_object::w_module_get_w_dict(obj);
            if !w_dict.is_null() {
                match delitem(w_dict, w_str_new(name)) {
                    Ok(()) => return Ok(w_none()),
                    Err(err) if err.kind == crate::PyErrorKind::KeyError => {
                        // descroperation.py descr__delattr__: deldictvalue
                        // returning False raises AttributeError immediately.
                        return Err(raiseattrerror(obj, name));
                    }
                    Err(err) => return Err(err),
                }
            }
        }
    }
    // Type objects: remove the key from the class namespace.  Real
    // removal (not a PY_NULL tombstone) keeps `cls.__dict__` membership,
    // iteration, and length correct and back-mirrors the deletion into a
    // materialized `__dict__`.  A missing key raises AttributeError, the
    // same as the surrogate sibling `object_delattr_surrogate`.
    unsafe {
        if is_type(obj) {
            // typeobject.py:437 — only heap types may have attributes deleted.
            if !pyre_object::w_type_is_heaptype(obj) {
                return Err(PyError::type_error(format!(
                    "cannot delete attributes on immutable type object '{}'",
                    w_type_get_name(obj)
                )));
            }
            let dict_ptr = w_type_get_dict_ptr(obj) as *mut crate::DictStorage;
            if !dict_ptr.is_null() {
                if crate::dict_storage_delete(&mut *dict_ptr, name) {
                    // typeobject.py:445 — `self.mutated(key)` mirrors the
                    // setattr branch's invalidation across the subclass
                    // tree.
                    mutated(obj, Some(name));
                    return Ok(w_none());
                }
                return Err(raiseattrerror(obj, name));
            }
        }
    }
    // `pypy/module/exceptions/interp_exceptions.py:159-161
    // W_BaseException.descr_delargs` → unconditional TypeError
    // ("args may not be deleted").  Reject `del e.args` before the
    // generic instance-dict removal path, which would otherwise
    // succeed silently when an entry existed there.
    if unsafe { pyre_object::is_exception(obj) } && name == "args" {
        return Err(PyError::type_error("args may not be deleted"));
    }
    // Instance/general: remove from the instance dict.
    let w_dict = getdict_backing(obj);
    if !w_dict.is_null() {
        let removed = unsafe { pyre_object::w_dict_delitem_str(w_dict, name) };
        if removed {
            return Ok(w_none());
        }
    }
    let tp_name = unsafe { (*(*obj).ob_type).name };
    Err(PyError::new(
        PyErrorKind::AttributeError,
        format!("'{tp_name}' object has no attribute '{name}'"),
    ))
}

/// PyPy: baseobjspace.py `call`.
///
/// Call a Python callable with packed positional arguments and optional kwargs.
pub fn call(
    callable: PyObjectRef,
    w_args: PyObjectRef,
    w_kwds: Option<PyObjectRef>,
) -> PyObjectRef {
    if let Some(w_kwargs) = w_kwds {
        if !w_kwargs.is_null() && !unsafe { is_none(w_kwargs) } {
            panic!("call with kwargs is not yet implemented in pyre");
        }
    }

    let mut args = Vec::new();
    unsafe {
        if is_tuple(w_args) {
            let len = w_tuple_len(w_args);
            args.reserve(len);
            for i in 0..len {
                if let Some(arg) = w_tuple_getitem(w_args, i as i64) {
                    args.push(arg);
                }
            }
        } else if is_list(w_args) {
            let len = w_list_len(w_args);
            args.reserve(len);
            for i in 0..len {
                if let Some(arg) = w_list_getitem(w_args, i as i64) {
                    args.push(arg);
                }
            }
        } else if !w_args.is_null() {
            panic!("call() expects tuple or list positional arguments");
        }
    }
    call_function(callable, &args)
}

/// PyPy: baseobjspace.py `call_obj_args` — add a leading object before args.
pub fn call_obj_args(callable: PyObjectRef, obj: PyObjectRef, args: &[PyObjectRef]) -> PyObjectRef {
    if obj.is_null() {
        return call_function(callable, args);
    }
    let mut call_args = Vec::with_capacity(1 + args.len());
    call_args.push(obj);
    call_args.extend_from_slice(args);
    call_function(callable, &call_args)
}

/// PyPy: baseobjspace.py `call_valuestack`.
pub fn call_valuestack(
    callable: PyObjectRef,
    nargs: usize,
    frame: &mut crate::pyframe::PyFrame,
    dropvalues: usize,
    methodcall: bool,
) -> PyObjectRef {
    let mut args = Vec::with_capacity(nargs);
    for _ in 0..nargs {
        args.push(frame.pop());
    }
    args.reverse();

    let mut remaining_to_drop = dropvalues.saturating_sub(nargs);

    let null_or_self = if methodcall {
        let value = if remaining_to_drop > 0 {
            remaining_to_drop -= 1;
            Some(frame.pop())
        } else {
            None
        };
        if remaining_to_drop > 0 {
            frame.pop();
            remaining_to_drop -= 1;
        }
        value
    } else {
        if remaining_to_drop > 0 {
            frame.pop();
            remaining_to_drop -= 1;
        }
        None
    };

    for _ in 0..remaining_to_drop {
        frame.pop();
    }

    if let Some(null_or_self) = null_or_self {
        if !null_or_self.is_null() && !unsafe { is_none(null_or_self) } {
            args.insert(0, null_or_self);
        }
    }
    call_function(callable, &args)
}

/// PyPy: baseobjspace.py:1269-1277 `call_args_and_c_profile`.
///
/// ```python
/// def call_args_and_c_profile(self, frame, w_func, args):
///     ec = self.getexecutioncontext()
///     ec.c_call_trace(frame, w_func, args)
///     try:
///         w_res = self.call_args(w_func, args)
///     except OperationError:
///         ec.c_exception_trace(frame, w_func)
///         raise
///     ec.c_return_trace(frame, w_func, args)
///     return w_res
/// ```
///
/// Pyre's `call_function` returns `PyObjectRef` and stashes any error
/// via `set_call_error`; we recover it through `take_call_error` to
/// run the c_exception_trace branch.  Trace-callback errors raised by
/// the c_call/c_return/c_exception events propagate via the same TLS
/// stash so the JIT-side and interpreter-side error paths see them.
///
/// This wrapper is for call sites that already have a positional-only
/// slice.  Call sites that know keyword_names_w / keywords_w must call
/// `call_args_and_c_profile_args` with `Arguments::with_kw`, mirroring
/// pyopcode.py's `CALL_FUNCTION_KW` / `CALL_FUNCTION_EX` construction of
/// a single `Arguments` object before the profiled-builtin branch.
pub fn call_args_and_c_profile(
    frame: &mut crate::pyframe::PyFrame,
    callable: PyObjectRef,
    args: &[PyObjectRef],
) -> PyObjectRef {
    let arguments = crate::argument::Arguments::positional_only(args);
    call_args_and_c_profile_args(frame, callable, &arguments, args)
}

/// `baseobjspace.py:1269-1278 call_args_and_c_profile` with a
/// pre-built `Arguments` instance.
///
/// Step 2 of the Arguments port (continuation of `argument.rs`):
/// callers that have positional and kwargs separated (currently
/// `call::call_with_kwargs` for the builtin path) construct
/// `Arguments::with_kw(pos_args, keyword_names_w, keywords_w)` and
/// route through this helper, instead of wrapping the merged slice
/// as positional-only.  This way `firstarg()` reads `pos_args[0]`
/// rather than surfacing the trailing kwargs dict that pyre's flat
/// call surface otherwise appends.
///
/// `flat_args` is the legacy flat slice (positional + trailing kwargs
/// dict) that `call_function` still expects until the call surface
/// itself learns about Arguments.
pub fn call_args_and_c_profile_args(
    frame: &mut crate::pyframe::PyFrame,
    callable: PyObjectRef,
    arguments: &crate::argument::Arguments,
    flat_args: &[PyObjectRef],
) -> PyObjectRef {
    let ec = crate::call::getexecutioncontext() as *mut crate::PyExecutionContext;
    if !ec.is_null() {
        if let Err(err) = unsafe {
            (*ec).c_call_trace(
                frame as *mut crate::pyframe::PyFrame,
                callable,
                Some(arguments),
            )
        } {
            crate::call::set_call_error(err);
            return pyre_object::PY_NULL;
        }
    }
    let w_res = call_function(callable, flat_args);
    if w_res == pyre_object::PY_NULL {
        if !ec.is_null() {
            // baseobjspace.py:1274-1276 — `except OperationError:
            // ec.c_exception_trace(frame, w_func); raise`. The bare
            // `raise` re-raises the active exception, but Python
            // semantics are that an exception raised from inside an
            // `except` block replaces the in-flight one. Pyre's call
            // stash already holds the original OperationError; if
            // c_exception_trace raises, overwrite the stash so the
            // tracer error is what propagates.
            if let Err(trace_err) =
                unsafe { (*ec).c_exception_trace(frame as *mut crate::pyframe::PyFrame, callable) }
            {
                crate::call::set_call_error(trace_err);
            }
        }
        return pyre_object::PY_NULL;
    }
    if !ec.is_null() {
        if let Err(err) = unsafe {
            (*ec).c_return_trace(
                frame as *mut crate::pyframe::PyFrame,
                callable,
                Some(arguments),
            )
        } {
            crate::call::set_call_error(err);
            return pyre_object::PY_NULL;
        }
    }
    w_res
}

/// PyPy: baseobjspace.py `call_method`.
///
/// Returns `PY_NULL` and stashes the error in `PENDING_CALL_ERROR` when
/// either the attribute lookup or the call itself raises — same bare-
/// PyObjectRef contract as `call_function_impl_raw`.
pub fn call_method(obj: PyObjectRef, methname: &str, args: &[PyObjectRef]) -> PyObjectRef {
    match getattr_str(obj, methname) {
        Ok(method) => call_function(method, args),
        Err(e) => {
            crate::call::set_call_error(e);
            pyre_object::PY_NULL
        }
    }
}

/// PyPy: baseobjspace.py `call_function`.
///
/// Dispatches to builtins, user functions, and type objects.
pub fn call_function(callable: PyObjectRef, args: &[PyObjectRef]) -> PyObjectRef {
    crate::call::call_function_impl(callable, args)
}

/// PyPy: baseobjspace.py `callable_w`.
pub fn callable_w(obj: PyObjectRef) -> bool {
    // `PyCallable_Check` — the builtin callable kinds (function / builtin
    // function, bound method, static- and classmethod, type) dispatch through
    // dedicated slots rather than a `__call__` dict entry, so each is
    // recognised directly; any other object is callable iff its type defines
    // `__call__`.  Mirrors `builtins::builtin_callable`.
    unsafe {
        is_function(obj)
            || is_type(obj)
            || pyre_object::is_method(obj)
            || pyre_object::function::is_staticmethod(obj)
            || pyre_object::function::is_classmethod(obj)
            || crate::typedef::r#type(obj)
                .and_then(|t| lookup_in_type(t, "__call__"))
                .is_some()
    }
}

/// PyPy: baseobjspace.py `callable`.
pub fn callable(obj: PyObjectRef) -> PyObjectRef {
    if callable_w(obj) {
        w_bool_from(true)
    } else {
        w_bool_from(false)
    }
}

/// PyPy `ObjSpace.call_function_or_identity`.
pub fn call_function_or_identity(obj: PyObjectRef, dunder: &str) -> PyObjectRef {
    unsafe {
        if is_instance(obj) {
            if let Some(method) = lookup(obj, dunder) {
                return call_function(method, &[obj]);
            }
        }
    }
    obj
}

/// PyPy baseobjspace.py equivalent.
pub fn get_printable_location(greenkey: PyObjectRef) -> String {
    format!("unpackiterable [{:?}]", greenkey)
}

/// PyPy baseobjspace.py equivalent.
pub fn wrappable_class_name(class: PyObjectRef) -> String {
    if class.is_null() {
        return "internal subclass".to_string();
    }
    unsafe {
        let type_name = (*(*class).ob_type).name;
        if is_type(class) {
            type_name.to_string()
        } else {
            format!("internal subclass of {type_name}")
        }
    }
}

/// pypy/interpreter/baseobjspace.py:983-998 `unpackiterable`.
///
/// ```python
/// def unpackiterable(self, w_iterable, expected_length=-1):
///     """Unpack an iterable into a real (interpreter-level) list.
///     Raise an OperationError(w_ValueError) if the length is wrong."""
///     w_iterator = self.iter(w_iterable)
///     if expected_length == -1:
///         if self.is_generator(w_iterator):
///             # special hack for speed
///             lst_w = []
///             w_iterator.unpack_into(lst_w)
///             return lst_w
///         return self._unpackiterable_unknown_length(w_iterator, w_iterable)
///     else:
///         lst_w = self._unpackiterable_known_length(w_iterator,
///                                                   expected_length)
///         return lst_w[:]     # make the resulting list resizable
/// ```
///
/// `expected_length = -1` is PyPy's sentinel for "any length".  When
/// the caller supplies a positive expected_length, the length-validation
/// arm at `baseobjspace.py:1031-1053
/// `_unpackiterable_known_length_jitlook` runs and raises ValueError
/// on mismatch (`too many values to unpack` /
/// `not enough values to unpack`).
pub fn unpackiterable(
    w_iterable: PyObjectRef,
    expected_length: isize,
) -> Result<Vec<PyObjectRef>, crate::PyError> {
    let w_iterator = iter(w_iterable)?;
    if expected_length == -1 {
        // baseobjspace.py:989-993 — generator fast path.  PyPy comments
        // (`generator.py:322 "This is a hack for performance"`) flag this
        // as an optimization, but the structural difference from the
        // generic next-loop is observable: `unpack_into` runs each yield
        // through the same suspended frame without the per-iteration
        // PyTypeObject/__next__ slot lookup, and uses a private
        // `_invoke_execute_frame(space.w_None)` instead of `space.next`.
        // Port both branches.
        if unsafe { pyre_object::generator::is_generator(w_iterator) } {
            let mut lst_w: Vec<PyObjectRef> = Vec::new();
            generator_unpack_into(w_iterator, &mut lst_w)?;
            return Ok(lst_w);
        }
        _unpackiterable_unknown_length(w_iterator, w_iterable)
    } else {
        // baseobjspace.py:996-998 — known-length path with shape validation.
        _unpackiterable_known_length_jitlook(w_iterator, expected_length as usize)
    }
}

/// pypy/interpreter/baseobjspace.py:368-372 `iterator_greenkey`.
///
/// ```python
/// def iterator_greenkey(self, space):
///     """ Return something that can be used as a green key in jit
///     drivers that iterate over self. by default, it's just the type
///     of self, but custom iterators should override it. """
///     return space.type(self)
/// ```
///
/// Default implementation returning `space.type(w_iterable)`.  Pyre's
/// W_Root subclasses don't carry per-type overrides yet, so every
/// caller hits this default — matching PyPy's
/// `baseobjspace.py:2099-2103 ObjSpace.iterator_greenkey` after the
/// trivial `w_iterable.iterator_greenkey(self)` indirection.
pub fn iterator_greenkey(w_iterable: PyObjectRef) -> PyObjectRef {
    if w_iterable.is_null() {
        return pyre_object::PY_NULL;
    }
    crate::typedef::r#type(w_iterable).unwrap_or(pyre_object::PY_NULL)
}

/// pypy/interpreter/baseobjspace.py:1010 `unpackiterable_driver`
/// JitDriver merge-point hint.
///
/// PyPy declares `unpackiterable_driver = JitDriver(greens=['greenkey'],
/// reds='auto', name='unpackiterable')` and calls
/// `unpackiterable_driver.jit_merge_point(greenkey=greenkey)` once per
/// loop turn so the JIT specialises the loop trace per
/// `iterator_greenkey(w_iterator)` value.
///
/// Pyre's metainterp drives compilation from bytecode-level
/// `BC_JIT_MERGE_POINT` opcodes; an in-Rust `_unpackiterable_unknown_length`
/// is residual-call'd from the JIT'd interpreter loop, so the merge-point
/// inside this body is not visible to the live tracer.  The structural
/// port keeps the greenkey computation + the call so the per-greenkey
/// dispatch contract is documented at the call site; the runtime hook
/// is a no-op until the metainterp grows a Rust-callee merge-point
/// observer.
#[inline]
fn unpackiterable_driver_jit_merge_point(_greenkey: PyObjectRef) {
    // No-op: see doc comment above.
}

/// pypy/interpreter/generator.py:317-343 `_create_unpack_into` body.
///
/// ```python
/// def unpack_into(self, results):
///     """This is a hack for performance: runs the generator and
///     collects all produced items in a list."""
///     frame = self.frame
///     if frame is None:    # already finished
///         return
///     pycode = self.pycode
///     while True:
///         jitdriver.jit_merge_point(pycode=pycode)
///         space = self.space
///         try:
///             w_result = self._invoke_execute_frame(space.w_None)
///         except OperationError as e:
///             if not e.match(space, space.w_StopIteration):
///                 raise
///             break
///         if frame.frame_finished_execution:
///             self.frame_is_finished()
///             break
///         results.append(w_result)     # YIELDed
/// ```
///
/// Pyre stores the suspended PyFrame on the GeneratorIterator as
/// `frame_ptr`; an exhausted generator has either `exhausted=true` or a
/// null frame_ptr.  `_invoke_execute_frame(space.w_None)` corresponds to
/// the frame's own `execute_frame(None, None)` resume — same routing as
/// `generator_send_ex` for the `already_started=true, w_arg=None` path.
fn generator_unpack_into(
    gen_obj: PyObjectRef,
    results: &mut Vec<PyObjectRef>,
) -> Result<(), crate::PyError> {
    use pyre_object::generator::*;
    unsafe {
        // generator.py:325-327 — `frame is None: return`.
        if w_generator_is_running(gen_obj) {
            return Err(PyError::value_error("generator already executing"));
        }
        if w_generator_is_exhausted(gen_obj) {
            return Ok(());
        }
        let frame_ptr = w_generator_get_frame(gen_obj) as *mut crate::pyframe::PyFrame;
        if frame_ptr.is_null() {
            w_generator_set_exhausted(gen_obj);
            return Ok(());
        }
        let frame = &mut *frame_ptr;
        // generator.py:328 `pycode = self.pycode` — pyre stashes pycode on
        // the suspended frame; expose it as the JitDriver greenkey.
        let pycode = frame.pycode as PyObjectRef;
        loop {
            // generator.py:330 `jitdriver.jit_merge_point(pycode=pycode)`.
            unpackiterable_driver_jit_merge_point(pycode);
            // generator.py:331 `space = self.space`.
            // generator.py:332-336 `try: w_result =
            //   self._invoke_execute_frame(space.w_None)`.
            //
            // `_invoke_execute_frame(w_arg_or_err)` calls
            // `frame.execute_frame(w_arg_or_err)` (generator.py:131),
            // which feeds `w_arg_or_err` to `resume_execute_frame` —
            // pushing it onto the YIELD result slot.  unpack_into
            // always passes `space.w_None`, both for the never-started
            // case (frame.last_instr == -1: PyPy
            // `resume_execute_frame` skips the push and returns
            // `r_uint(0)`) and for every subsequent resume.  Pyre's
            // earlier `frame.execute_frame(None, None)` skipped the
            // push entirely, so `yield`-expressions that bind the
            // resume value (e.g. `x = yield`) would observe stale
            // stack on the second iteration.
            w_generator_set_started(gen_obj);
            w_generator_set_running(gen_obj, true);
            let result = frame.execute_frame(Some(pyre_object::w_none()), None);
            w_generator_set_running(gen_obj, false);
            match result {
                // generator.py:132-138 `_invoke_execute_frame`'s
                // `finally: self.frame_is_finished()` runs before the
                // OperationError reaches the unpack_into try/except,
                // so by the time PyPy's `if e.match(StopIteration):
                // break` fires the generator is already marked
                // finished.  Pyre's inline `frame.execute_frame` path
                // skips that finally block, so mirror it explicitly.
                Err(e) if e.kind == crate::PyErrorKind::StopIteration => {
                    // generator.py:131-138 — `_invoke_execute_frame` applies
                    // `_leak_stopiteration` (PEP 479) BEFORE unpack_into's
                    // `if e.match(StopIteration): break`, so a StopIteration
                    // leaked from the body becomes RuntimeError and propagates;
                    // it is not the normal-exhaustion path (which is the
                    // `Ok`/`frame_finished_execution` arm below).
                    w_generator_set_exhausted(gen_obj);
                    return Err(leak_stopiteration(e));
                }
                Err(e) => {
                    w_generator_set_exhausted(gen_obj);
                    return Err(e);
                }
                Ok(w_result) => {
                    // generator.py:339-341 — frame finished ⇒ RETURNed,
                    // mark exhausted and stop without appending.
                    if frame.frame_finished_execution {
                        w_generator_set_exhausted(gen_obj);
                        break;
                    }
                    // generator.py:342 `results.append(w_result)`.
                    results.push(w_result);
                }
            }
        }
        Ok(())
    }
}

/// pypy/interpreter/baseobjspace.py:1000-1021
/// `_unpackiterable_unknown_length`.
///
/// ```python
/// def _unpackiterable_unknown_length(self, w_iterator, w_iterable):
///     try:
///         items = newlist_hint(self.length_hint(w_iterable, 0))
///     except MemoryError:
///         items = []
///     greenkey = self.iterator_greenkey(w_iterator)
///     while True:
///         unpackiterable_driver.jit_merge_point(greenkey=greenkey)
///         try:
///             w_item = self.next(w_iterator)
///         except OperationError as e:
///             if not e.match(self, self.w_StopIteration):
///                 raise
///             break
///         items.append(w_item)
///     return items
/// ```
fn _unpackiterable_unknown_length(
    w_iterator: PyObjectRef,
    w_iterable: PyObjectRef,
) -> Result<Vec<PyObjectRef>, crate::PyError> {
    // baseobjspace.py:1005-1008 — `try: items = newlist_hint(length_hint(...))
    // except MemoryError: items = []`.  Mirror with try_reserve_exact so a
    // hostile / huge `__length_hint__` does not turn into a Rust panic
    // (Vec::with_capacity aborts on capacity overflow).
    let hint = length_hint(w_iterable, 0)?;
    let mut items: Vec<PyObjectRef> = Vec::new();
    if hint > 0 {
        let _ = items.try_reserve_exact(hint as usize);
    }
    // baseobjspace.py:1010 `greenkey = self.iterator_greenkey(w_iterator)`.
    let greenkey = iterator_greenkey(w_iterator);
    loop {
        // baseobjspace.py:1012
        // `unpackiterable_driver.jit_merge_point(greenkey=greenkey)`.
        unpackiterable_driver_jit_merge_point(greenkey);
        match next(w_iterator) {
            Ok(w_item) => items.push(w_item),
            Err(e) if e.kind == crate::PyErrorKind::StopIteration => break,
            Err(e) => return Err(e),
        }
    }
    Ok(items)
}

/// pypy/interpreter/baseobjspace.py:1080-1108 `length_hint`.
///
/// Returns the length of an object, consulting its `__length_hint__`
/// method if necessary.  Errors mirror the upstream contract:
/// `len_w`'s TypeError / AttributeError are absorbed; an
/// `__length_hint__` that raises TypeError / AttributeError returns
/// `default`; a NotImplemented return also yields `default`; a
/// negative return raises ValueError "__length_hint__() should return
/// >= 0"; any other exception propagates.
pub fn length_hint(w_obj: PyObjectRef, default: i64) -> Result<i64, crate::PyError> {
    match len_w(w_obj) {
        Ok(n) => return Ok(n),
        Err(e)
            if e.kind == crate::PyErrorKind::TypeError
                || e.kind == crate::PyErrorKind::AttributeError => {}
        Err(e) => return Err(e),
    }
    // baseobjspace.py:1093 `w_descr = space.lookup(w_obj, '__length_hint__')`
    // — a type-MRO special-method lookup, NOT full attribute access: an
    // instance-dict or `__getattr__`-synthesized `__length_hint__` is not
    // consulted.  pyre's builtin iterators carry `__length_hint__` in the
    // getattr_str method tables rather than the type dict, so a type miss on a
    // non-user object falls back to the bare `__getattribute__` form of
    // getattr_str (`call_getattr = false`): it still reaches the builtin
    // method-table `__length_hint__`, but never fires a module/metaclass
    // `__getattr__` hook, so the lookup stays type-MRO-faithful.  A user-class
    // instance (is_instance) is excluded entirely so its instance dict is
    // never consulted; a type miss there takes the default.
    let w_type = crate::typedef::r#type(w_obj).unwrap_or(std::ptr::null_mut());
    let w_descr = if w_type.is_null() {
        None
    } else {
        unsafe { lookup_in_type_where(w_type, "__length_hint__") }
    };
    let self_args = [w_obj];
    // baseobjspace.py:1095 `space.get_and_call_function(w_descr, w_obj)` — a
    // type-MRO descriptor is called with the object as self; the builtin
    // method-table result is already bound and called with no extra args.
    let (callable, args): (PyObjectRef, &[PyObjectRef]) = match w_descr {
        Some(descr) => (descr, &self_args),
        None => {
            if unsafe { is_instance(w_obj) } {
                return Ok(default);
            }
            match getattr_str_impl(w_obj, "__length_hint__", false) {
                Ok(m) => (m, &[]),
                Err(e) if e.kind == crate::PyErrorKind::AttributeError => return Ok(default),
                Err(e) => return Err(e),
            }
        }
    };
    let w_hint = match crate::call::call_function_impl_result(callable, args) {
        Ok(v) => v,
        Err(err) => {
            if err.kind == crate::PyErrorKind::TypeError
                || err.kind == crate::PyErrorKind::AttributeError
            {
                return Ok(default);
            }
            return Err(err);
        }
    };
    if is_w(w_hint, pyre_object::special::w_not_implemented()) {
        return Ok(default);
    }
    let hint = int_w(w_hint)?;
    if hint < 0 {
        return Err(crate::PyError::value_error(
            "__length_hint__() should return >= 0",
        ));
    }
    Ok(hint)
}

/// pypy/objspace/descroperation.py:310-317 `_check_len_result`.
///
/// ```python
/// def _check_len_result(space, w_int):
///     # Will complain if result is too big.
///     assert space.isinstance_w(w_int, space.w_int)
///     if space.is_true(space.lt(w_int, space.newint(0))):
///         raise oefmt(space.w_ValueError, "__len__() should return >= 0")
///     result = space.getindex_w(w_int, space.w_OverflowError)
///     assert result >= 0
///     return result
/// ```
///
/// `int_w` already mirrors `getindex_w(w_int, w_OverflowError)` for the
/// already-int caller contract here: long values that do not fit `i64`
/// raise `OverflowError` ("int too large to convert to int") via
/// `intobject.py:558` / `longobject.py` `_int_w`.
fn _check_len_result(w_int: PyObjectRef) -> Result<i64, crate::PyError> {
    let n = int_w(w_int)?;
    if n < 0 {
        return Err(crate::PyError::value_error("__len__() should return >= 0"));
    }
    Ok(n)
}

/// pypy/objspace/descroperation.py:300-302 `len_w`.
///
/// ```python
/// def len_w(space, w_obj):
///     w_res = space._len(w_obj)
///     return space._check_len_result(space.index(w_res))
/// ```
///
/// pyre's `len()` covers `_len`; the result is then funnelled through
/// `space.index` (descroperation.py:599 `_index` + line 622 `index`)
/// before `_check_len_result` so `__index__` is consulted but `__int__`
/// is NOT — matching PyPy's stricter contract.
pub fn len_w(w_obj: PyObjectRef) -> Result<i64, crate::PyError> {
    let w_res = len(w_obj)?;
    let w_index = space_index(w_res)?;
    _check_len_result(w_index)
}

/// pypy/objspace/descroperation.py:599-620 `_index` + line 622-627 `index`.
///
/// ```python
/// def _index(space, w_obj):
///     if space.isinstance_w(w_obj, space.w_int):
///         return w_obj
///     w_impl = space.lookup(w_obj, '__index__')
///     if w_impl is None:
///         raise oefmt(space.w_TypeError,
///                     "'%T' object cannot be interpreted as an integer", w_obj)
///     w_result = space.get_and_call_function(w_impl, w_obj)
///     if space.is_w(space.type(w_result), space.w_int):
///         return w_result
///     if not space.isinstance_w(w_result, space.w_int):
///         raise oefmt(space.w_TypeError,
///                 "__index__ returned non-int (type %T)", w_result)
///     ...  # subclass-of-int deprecation warning, then return
///     return w_result
/// ```
///
/// `space.index` (line 622) wraps `_index` and additionally re-wraps
/// strict subclass-of-int results into a fresh `W_IntObject` /
/// `W_LongObject`.  Pyre's `int`/`long` are leaf types so the wrap is a
/// no-op; the body below is `_index` line-for-line.
pub fn space_index(obj: PyObjectRef) -> Result<PyObjectRef, PyError> {
    if obj.is_null() {
        return Err(PyError::type_error("space.index: null object"));
    }
    if unsafe { pyre_object::pyobject::is_int_or_long(obj) } {
        return Ok(obj);
    }
    let Some(method) = (unsafe { lookup(obj, "__index__") }) else {
        return Err(PyError::type_error(format!(
            "'{}' object cannot be interpreted as an integer",
            object_functionstr_type_name(obj),
        )));
    };
    let w_result = crate::builtins::call_and_check(method, &[obj])?;
    if unsafe { pyre_object::pyobject::is_int_or_long(w_result) } {
        return Ok(w_result);
    }
    Err(PyError::type_error(format!(
        "__index__ returned non-int (type {})",
        object_functionstr_type_name(w_result),
    )))
}

/// baseobjspace.py:1847 `float_w` — the `space.float` coercion
/// (descroperation.py:870), i.e. apply `__float__` and unwrap to an
/// interp-level f64. Unlike the `float()` constructor it neither parses
/// strings nor consults `__index__`; a non-float operand raises
/// TypeError "must be real number, not %T".
pub fn float_w(obj: PyObjectRef) -> Result<f64, PyError> {
    if obj.is_null() {
        return Err(PyError::type_error("float_w: null object"));
    }
    unsafe {
        if pyre_object::is_float(obj) {
            return Ok(pyre_object::w_float_get_value(obj));
        }
        // `is_int` is true for a bool (`BOOL_TYPE`), so test `is_bool` first.
        if pyre_object::pyobject::is_bool(obj) {
            return Ok(if pyre_object::boolobject::w_bool_get_value(obj) {
                1.0
            } else {
                0.0
            });
        }
        if pyre_object::pyobject::is_int(obj) {
            return Ok(pyre_object::intobject::w_int_get_value(obj) as f64);
        }
        if pyre_object::pyobject::is_long(obj) {
            use num_traits::ToPrimitive;
            // longobject.py:131-135 `tofloat` — `rbigint.tofloat()` raises
            // OverflowError "int too large to convert to float" when the
            // value does not fit a C double.
            let f = pyre_object::longobject::jit_bigint_to_f64_or_inf(
                pyre_object::longobject::w_long_get_value(obj),
            );
            if !f.is_finite() {
                return Err(PyError::overflow_error("int too large to convert to float"));
            }
            return Ok(f);
        }
    }
    let Some(method) = (unsafe { lookup(obj, "__float__") }) else {
        return Err(PyError::type_error(format!(
            "must be real number, not {}",
            object_functionstr_type_name(obj)
        )));
    };
    let w_result = crate::builtins::call_and_check(method, &[obj])?;
    if unsafe { pyre_object::is_float(w_result) } {
        return Ok(unsafe { pyre_object::w_float_get_value(w_result) });
    }
    Err(PyError::type_error(format!(
        "__float__ returned non-float (type '{}')",
        object_functionstr_type_name(w_result)
    )))
}

/// baseobjspace.py:1564 `getindex_w` with `w_exception=None` — apply
/// `space.index` (`__index__`) then convert to an i64, silently clamping
/// to `i64::MAX` / `i64::MIN` on overflow rather than raising.
pub fn getindex_w(obj: PyObjectRef) -> Result<i64, PyError> {
    let w_index = space_index(obj)?;
    match int_w(w_index) {
        Ok(index) => Ok(index),
        Err(e) if e.kind == PyErrorKind::OverflowError => {
            let big = unsafe { crate::builtins::obj_to_bigint(w_index) };
            if big.sign() == malachite_bigint::Sign::Minus {
                Ok(i64::MIN)
            } else {
                Ok(i64::MAX)
            }
        }
        Err(e) => Err(e),
    }
}

/// `objspace.honor__builtins__` default is False — the frame builtin is
/// `space.builtin`, ignoring a custom `__builtins__` in globals.  The
/// `pick_builtin*` family below is the `honor__builtins__=True` path,
/// reached only when this flag is set.
pub const HONOR_BUILTINS: bool = false;

/// Resolve the frame builtin for a raw-storage globals.  Default
/// (`HONOR_BUILTINS=false`) returns `space.builtin` (`ec.get_builtin()`),
/// ignoring a custom `__builtins__`; the `true` path delegates to
/// [`pick_builtin`].
pub fn frame_builtin(
    w_globals: *mut crate::DictStorage,
    exec_ctx: *const crate::PyExecutionContext,
) -> PyObjectRef {
    if HONOR_BUILTINS {
        return pick_builtin(w_globals, exec_ctx);
    }
    if !exec_ctx.is_null() {
        let b = unsafe { (*exec_ctx).get_builtin() };
        if !b.is_null() {
            return b;
        }
    }
    build_default_pick_builtin_module()
}

/// Resolve the frame builtin for an object globals.  Default
/// (`HONOR_BUILTINS=false`) returns `space.builtin`; the `true` path
/// delegates to [`pick_builtin_obj`].
pub fn frame_builtin_obj(
    w_globals: PyObjectRef,
    exec_ctx: *const crate::PyExecutionContext,
) -> PyObjectRef {
    if HONOR_BUILTINS {
        return pick_builtin_obj(w_globals, exec_ctx);
    }
    if !exec_ctx.is_null() {
        let b = unsafe { (*exec_ctx).get_builtin() };
        if !b.is_null() {
            return b;
        }
    }
    build_default_pick_builtin_module()
}

/// Fallible variant of [`frame_builtin_obj`].  Default
/// (`HONOR_BUILTINS=false`) returns `space.builtin`; the `true` path
/// delegates to [`pick_builtin_obj_checked`].
pub fn frame_builtin_obj_checked(
    w_globals: PyObjectRef,
    exec_ctx: *const crate::PyExecutionContext,
) -> Result<PyObjectRef, crate::PyError> {
    if HONOR_BUILTINS {
        return pick_builtin_obj_checked(w_globals, exec_ctx);
    }
    if !exec_ctx.is_null() {
        let b = unsafe { (*exec_ctx).get_builtin() };
        if !b.is_null() {
            return Ok(b);
        }
    }
    Ok(build_default_pick_builtin_module())
}

/// `pyframe.py:115-116 self.builtin = space.builtin.pick_builtin(
/// w_globals)`.  Body ports `pypy/module/__builtin__/moduledef.py:89-109
/// pick_builtin`:
///   1. `space.getitem(w_globals, '__builtins__')` (`KeyError` ⇒ default)
///   2. recognise `Module` ⇒ return that Module
///   3. recognise dict (incl. dict subclass) ⇒ wrap as
///      `module.Module(space, None, w_builtin)` (a fresh Module per
///      call, with `module.w_dict = w_builtin`).
///   4. absent / not Module-or-dict ⇒ build a default empty Module
///      with only `None=w_None` defined — matches `moduledef.py:106-108`
///      `builtin = module.Module(space, None); space.setitem(builtin
///      .w_dict, 'None', w_None); return builtin`.
pub fn pick_builtin(
    w_globals: *mut crate::DictStorage,
    exec_ctx: *const crate::PyExecutionContext,
) -> PyObjectRef {
    if !w_globals.is_null() {
        if let Some(w_builtin) = crate::dict_storage_get(unsafe { &*w_globals }, "__builtins__") {
            if !w_builtin.is_null() {
                // moduledef.py:100-101 `if w_builtin is space.builtin: return
                // space.builtin` — Module identity short-circuit.
                if !exec_ctx.is_null() {
                    let space_builtin = unsafe { (*exec_ctx).get_builtin() };
                    if !space_builtin.is_null() && std::ptr::eq(w_builtin, space_builtin) {
                        return w_builtin;
                    }
                }
                // moduledef.py:104 `isinstance(w_builtin, module.Module)`.
                if unsafe { pyre_object::is_module(w_builtin) } {
                    return w_builtin;
                }
                // moduledef.py:102-103 `space.isinstance_w(w_builtin, w_dict)`.
                // PyPy: `return module.Module(space, None, w_builtin)` —
                // a Module wrapping the caller's dict.  `LOAD_GLOBAL`
                // falls through to `space.finditem_str(w_module.w_dict,
                // name)`, dispatching through any dict subclass
                // `__getitem__` override.
                let backing = crate::type_methods::resolve_dict_backing(w_builtin);
                if !backing.is_null() {
                    return pyre_object::w_module_new_aliasing_dict(
                        "",
                        std::ptr::null_mut(),
                        w_builtin,
                    );
                }
                // Fall through — `__builtins__` is not Module/dict (e.g.
                // `42`, a list, ...).  PyPy moduledef.py:106-108 builds
                // the default empty Module here.
            }
        }
    }
    // moduledef.py:106-108 default — anonymous Module with only
    // `None=w_None`.  This is reached when (a) `w_globals` is null,
    // (b) `__builtins__` is absent from globals, or (c) `__builtins__`
    // is not Module/dict.
    build_default_pick_builtin_module()
}

/// Object-based `pick_builtin` for call frames whose globals came from
/// `Function.w_func_globals` as a W_DictObject, matching PyPy's
/// `pyframe.py:115 self.builtin = space.builtin.pick_builtin(w_globals)`.
///
/// Propagates a non-KeyError from the `__builtins__` lookup per
/// `moduledef.py:97-98 if not e.match(space, space.w_KeyError): raise`
/// (a dict-subclass globals whose `__getitem__` raises). `finditem_str`
/// already maps KeyError to `None`, so an `Err` here is always a
/// non-KeyError to propagate.
pub fn pick_builtin_obj_checked(
    w_globals: PyObjectRef,
    exec_ctx: *const crate::PyExecutionContext,
) -> Result<PyObjectRef, PyError> {
    if !w_globals.is_null() {
        match finditem_str(w_globals, "__builtins__") {
            Ok(Some(w_builtin)) if !w_builtin.is_null() => {
                if !exec_ctx.is_null() {
                    let space_builtin = unsafe { (*exec_ctx).get_builtin() };
                    if !space_builtin.is_null() && std::ptr::eq(w_builtin, space_builtin) {
                        return Ok(w_builtin);
                    }
                }
                if unsafe { pyre_object::is_module(w_builtin) } {
                    return Ok(w_builtin);
                }
                let backing = crate::type_methods::resolve_dict_backing(w_builtin);
                if !backing.is_null() {
                    return Ok(pyre_object::w_module_new_aliasing_dict(
                        "",
                        std::ptr::null_mut(),
                        w_builtin,
                    ));
                }
            }
            // `__builtins__` absent — moduledef.py:106-108 default Module.
            Ok(_) => {}
            // moduledef.py:97-98 — non-KeyError propagates.
            Err(e) => return Err(e),
        }
    }
    Ok(build_default_pick_builtin_module())
}

/// Infallible adapter over [`pick_builtin_obj_checked`] for the frame
/// builders that have not yet been made fallible (`pyframe.rs
/// new_for_call_*`, JIT `call_jit.rs`, `trace_opcode.rs`).  A non-KeyError
/// `__builtins__` lookup is dropped here, reproducing the pre-existing
/// behavior; the case only arises for a dict-subclass globals with a
/// raising `__getitem__` and is unreachable for a real module-dict
/// `__globals__`.  CONVERGENCE (R3.4 frame-build fallibility): migrate
/// callers to `pick_builtin_obj_checked` and retire this wrapper.
pub fn pick_builtin_obj(
    w_globals: PyObjectRef,
    exec_ctx: *const crate::PyExecutionContext,
) -> PyObjectRef {
    pick_builtin_obj_checked(w_globals, exec_ctx)
        .unwrap_or_else(|_| build_default_pick_builtin_module())
}

/// Allocate the `moduledef.py:106-108` default Module — empty backing
/// storage with `None=w_None`, anonymous (PyPy passes `name=None` to
/// `Module.__init__`; pyre's `w_module_new` requires a `&str` so use
/// the empty string as the anonymous-name sentinel).
fn build_default_pick_builtin_module() -> PyObjectRef {
    // `pypy/module/__builtin__/moduledef.py:106-108` constructs the
    // default Module backed by a `W_ModuleDictObject` whose strategy
    // is `ModuleDictStrategy` (`celldict.py:28`).  Pyre's
    // `w_module_dict_new()` ports that allocation directly; the
    // `Module(space, None, w_builtin)` aliasing-constructor path
    // hands the dict object straight through without the
    // `DictStorage` carrier.
    let w_dict = pyre_object::w_module_dict_new();
    unsafe {
        pyre_object::w_dict_setitem_str(w_dict, "None", pyre_object::w_none());
    }
    pyre_object::w_module_new_aliasing_dict("", std::ptr::null_mut(), w_dict)
}

/// pypy/interpreter/baseobjspace.py:1031-1053
/// `_unpackiterable_known_length_jitlook`.
///
/// ```python
/// @jit.unroll_safe
/// def _unpackiterable_known_length_jitlook(self, w_iterator, expected_length):
///     items = [None] * expected_length
///     idx = 0
///     while True:
///         try:
///             w_item = self.next(w_iterator)
///         except OperationError as e:
///             if not e.match(self, self.w_StopIteration):
///                 raise
///             break
///         if idx == expected_length:
///             raise oefmt(self.w_ValueError,
///                         "too many values to unpack (expected %d)",
///                         expected_length)
///         items[idx] = w_item
///         idx += 1
///     if idx < expected_length:
///         raise oefmt(self.w_ValueError,
///                     "not enough values to unpack (expected %d, got %d)",
///                     expected_length, idx)
///     return items
/// ```
fn _unpackiterable_known_length_jitlook(
    w_iterator: PyObjectRef,
    expected_length: usize,
) -> Result<Vec<PyObjectRef>, crate::PyError> {
    let mut items: Vec<PyObjectRef> = Vec::with_capacity(expected_length);
    loop {
        match next(w_iterator) {
            Ok(w_item) => {
                if items.len() == expected_length {
                    return Err(crate::PyError::value_error(format!(
                        "too many values to unpack (expected {expected_length})",
                    )));
                }
                items.push(w_item);
            }
            Err(e) if e.kind == crate::PyErrorKind::StopIteration => break,
            Err(e) => return Err(e),
        }
    }
    if items.len() < expected_length {
        return Err(crate::PyError::value_error(format!(
            "not enough values to unpack (expected {expected_length}, got {got})",
            got = items.len(),
        )));
    }
    Ok(items)
}

/// pypy/interpreter/baseobjspace.py:1159-1163 base default + the
/// `StdObjSpace` override at `pypy/objspace/std/objspace.py:609-617`.
///
/// ```python
/// # baseobjspace.py:1159-1163 (base default)
/// def view_as_kwargs(self, w_dict):
///     """ if w_dict is a kwargs-dict, return two lists, one of unwrapped
///     strings and one of wrapped values. otherwise return (None, None)
///     """
///     return (None, None)
///
/// # objspace.py:609-617 (StdObjSpace override)
/// def view_as_kwargs(self, w_dict):
///     # ... it never fails for dict subclasses; this emulates CPython's
///     # behavior which often won't call custom __iter__() or keys()
///     # methods in dict subclasses.
///     if isinstance(w_dict, W_DictObject):
///         return w_dict.view_as_kwargs()
///     return (None, None)
///
/// # dictmultiobject.py:307-310 (W_DictObject.view_as_kwargs)
/// def view_as_kwargs(self):
///     if not self.user_overridden_class:
///         return self.get_strategy().view_as_kwargs(self)
///     return None, None
///
/// # dictmultiobject.py:1325-1334 (kwargs strategy)
/// def view_as_kwargs(self, w_dict):
///     d = self.unerase(w_dict.dstorage)
///     l = len(d)
///     keys, values = [None] * l, [None] * l
///     i = 0
///     for w_key, val in d.iteritems():
///         keys[i] = w_key
///         values[i] = val
///         i += 1
///     return keys, values
/// ```
///
/// Pyre's `W_DictObject` does not carry the multi-strategy dispatch
/// (Object/Bytes/Int/Unicode/Kwargs), so the strategy-level
/// `view_as_kwargs` is open-coded here: walk the entries vector and
/// require every key to be a unicode string for the fast path to
/// apply, otherwise return `(None, None)` so callers fall through to
/// the slow `keys()` iteration arm at `argument.py:121-150`.
///
/// `user_overridden_class` (typeobject.py term for "type is exact
/// dict, not a subclass") corresponds to pyre's `is_dict(w_dict)` —
/// pyre dict subclasses live as `W_ObjectObject` with a backing
/// dict (`typedef.rs:820 dict_descr_new`), so an exact-type check on
/// the wrapper rules out user subclasses.  Both tuple slots are
/// `Option` so callers distinguish "no fast path" (None) from "fast
/// path with zero entries" (Some(empty)).
pub fn view_as_kwargs(w_dict: PyObjectRef) -> (Option<Vec<PyObjectRef>>, Option<Vec<PyObjectRef>>) {
    if w_dict.is_null() || !unsafe { pyre_object::is_dict(w_dict) } {
        return (None, None);
    }
    // `dictmultiobject.py:269-272 W_DictMultiObject.view_as_kwargs`:
    //
    // ```python
    // def view_as_kwargs(self):
    //     return self.get_strategy().view_as_kwargs(self)
    // ```
    //
    // Polymorphic dispatch via `w_dict_get_strategy(obj).view_as_kwargs`:
    // UnicodeDictStrategy and KwargsDictStrategy override to return
    // parallel arrays directly (`:1323-1334`, `kwargsdict.py:154-156`);
    // every other strategy returns `(None, None)` from the trait
    // default (`:568-569`), forcing the slow `keys()` path in
    // `argument.py:121-150`.
    unsafe { pyre_object::dictmultiobject::w_dict_get_strategy(w_dict).view_as_kwargs(w_dict) }
}

/// pypy/interpreter/baseobjspace.py:2105-2140 `object_functionstr`.
///
/// Full 4-branch port:
///
/// ```python
/// def object_functionstr(self, w_function):
///     from pypy.interpreter.function import Function, _Method
///     if isinstance(w_function, Function):
///         qualname = w_function.qualname
///         w_module = w_function.fget___module__(self)
///         if not self.is_w(w_module, self.w_None):
///             try:
///                 module = self.text_w(w_module)
///                 if module and module != 'builtins':
///                     return module + '.' + qualname + '()'
///             except OperationError:
///                 pass
///         return qualname + '()'
///     if isinstance(w_function, _Method):
///         return self.object_functionstr(w_function.w_function)
///     w_qualname = self.findattr(w_function, self.newtext('__qualname__'))
///     if w_qualname is not None:
///         try:
///             qualname = self.text_w(w_qualname)
///             w_module = self.findattr(w_function, self.newtext('__module__'))
///             if w_module is not None and not self.is_w(w_module, self.w_None):
///                 module = self.text_w(w_module)
///                 if module and module != 'builtins':
///                     return module + '.' + qualname + '()'
///             return qualname + '()'
///         except OperationError:
///             pass
///     try:
///         return self.text_w(self.str(w_function))
///     except OperationError:
///         return self.type(w_function).getname(self) + ' object'
/// ```
///
/// `object_functionstr` uses small private helpers instead of the
/// public `findattr` / `display::py_str` shortcuts because PyPy's
/// control flow is intentionally narrow here:
///
/// - `findattr` suppresses ordinary `OperationError` and returns
///   `None`, but **re-raises** SystemExit / KeyboardInterrupt
///   (`baseobjspace.py:881-884 if e.async(self): raise`).
/// - the final fallback calls `space.str(w_function)` once, then
///   `space.text_w(...)`; it does not try `repr()` after a failing or
///   non-string `__str__`.
///
/// The async-propagation contract is preserved: the `__qualname__`
/// findattr lives outside the inner try, so any async error there
/// surfaces as `Err(PyError)` to `raise_type_error`, which then
/// returns the async error in place of the TypeError prefix.  The
/// `__module__` findattr and the `text_w(...)` calls live inside the
/// PyPy try/except OperationError block — async OR ordinary errors
/// there fall through to the `str(w_function)` fallback, matching
/// PyPy's `except OperationError: pass`.
///
/// `function.py:53` initialises `self.qualname = qualname or self.name`,
/// so `w_function.qualname` returns the dotted form (e.g.
/// `Class.method`) for nested defs and the bare identifier for free
/// functions.  Pyre's `Function` does not carry the field directly;
/// `crate::function::function_get_qualname` reproduces the same
/// precedence (set-attr override → `code.qualname` → `function.name`).
pub fn object_functionstr(w_function: PyObjectRef) -> Result<String, crate::PyError> {
    // baseobjspace.py:2108-2120 — Function fast path (also covers
    // `FunctionWithFixedCode` and `BuiltinFunction`, both subclasses
    // of `Function` per function.py:783,786).  Pyre's `is_function`
    // unifies all three over `FUNCTION_TYPE` + `BUILTIN_FUNCTION_TYPE`.
    if !w_function.is_null() && unsafe { crate::function::is_function(w_function) } {
        // function.py:2108 `qualname = w_function.qualname` — match
        // PyPy's stored `qualname` field via the helper that walks
        // the stored `qualname` → `code.qualname` → `name`.
        let qualname = unsafe { crate::function::function_get_qualname(w_function) };
        let w_module = unsafe { crate::function::fget___module__(w_function) };
        if !is_w(w_module, w_none()) && unsafe { pyre_object::is_str(w_module) } {
            let module = unsafe { pyre_object::w_str_get_value(w_module) };
            if !module.is_empty() && module != "builtins" {
                return Ok(format!("{module}.{qualname}()"));
            }
        }
        return Ok(format!("{qualname}()"));
    }
    // baseobjspace.py:2121-2122 — `_Method` recursive fast path:
    // unwrap to `w_function.w_function` and recurse.
    if !w_function.is_null() && unsafe { pyre_object::function::is_method(w_function) } {
        let inner = unsafe { pyre_object::function::w_method_get_func(w_function) };
        return object_functionstr(inner);
    }
    // baseobjspace.py:2123 — `w_qualname = self.findattr(...)`.  This
    // findattr lives **outside** the inner try/except, so an async
    // exception (SystemExit/KeyboardInterrupt) here is propagated to
    // the caller via `Err(...)` matching `findattr`'s `e.async(self):
    // raise` re-raise (`baseobjspace.py:881-884`).
    let w_qualname_opt = object_functionstr_findattr(w_function, "__qualname__")?;
    // baseobjspace.py:2125-2135 — `try/except OperationError: pass`.
    // Every fault inside this block (text_w(qualname), findattr(module),
    // text_w(module)) must fall through to the `str(w_function)`
    // fallback rather than propagate.  In particular the second
    // `findattr(__module__)` is **inside** the try, so async errors
    // there are also suppressed — matches PyPy literally.
    'qualname: {
        let Some(w_qualname) = w_qualname_opt else {
            break 'qualname;
        };
        let Ok(qualname) = object_functionstr_text_w(w_qualname) else {
            break 'qualname;
        };
        let w_module = match object_functionstr_findattr(w_function, "__module__") {
            Ok(opt) => opt,
            // try/except OperationError: pass — async findattr suppressed too.
            Err(_) => break 'qualname,
        };
        match w_module {
            // No `__module__` or `__module__ is None`: bare `qualname()`.
            None => return Ok(format!("{qualname}()")),
            Some(w_module) if is_w(w_module, w_none()) => return Ok(format!("{qualname}()")),
            Some(w_module) => {
                // text_w(w_module) — non-string raises in PyPy → except →
                // fall through (do NOT return `qualname()` here, which
                // would mask the OperationError).
                let Ok(module) = object_functionstr_text_w(w_module) else {
                    break 'qualname;
                };
                if !module.is_empty() && module != "builtins" {
                    return Ok(format!("{module}.{qualname}()"));
                }
                // module empty or 'builtins': bare qualname().
                return Ok(format!("{qualname}()"));
            }
        }
    }
    // baseobjspace.py:2137-2140 — `text_w(str(w_function))` fallback,
    // else `type(w_function).getname() + ' object'`.  Both calls live
    // in `try/except OperationError: pass`, so any error (including
    // async) here is swallowed in PyPy — keep the same shape.  PyPy
    // calls `space.str(w_function)`, which dispatches to `__str__`
    // ALONE via `descroperation.str` (it does NOT fall back to
    // `__repr__` — that would require `space.repr(...)`).  Routing
    // through `display::py_str` would mask a failing/non-string
    // `__str__` by calling `__repr__`, producing a different message
    // than upstream.
    if let Ok(w_s) = object_functionstr_str(w_function)
        && unsafe { pyre_object::is_str(w_s) }
    {
        return Ok(unsafe { pyre_object::w_str_get_value(w_s).to_string() });
    }
    Ok(format!(
        "{} object",
        object_functionstr_type_name(w_function)
    ))
}

/// `space.str(w_obj)` — `__str__`-only fast path for
/// `object_functionstr`'s final fallback.
///
/// `pypy/objspace/descroperation.py str(self, space, w_obj)` does
/// `lookup(w_obj, '__str__')` then `space.get_and_call_function(...)`.
/// `__repr__` is never tried here — that would be `space.repr(...)`.
/// Returning `Err` for any of: missing `__str__` slot, descriptor
/// invocation failure, non-string return — caller suppresses to the
/// `<Type> object` fallback per PyPy's `except OperationError`.
fn object_functionstr_str(w_obj: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    if w_obj.is_null() {
        return Err(crate::PyError::type_error("NULL object"));
    }
    unsafe {
        if pyre_object::is_str(w_obj) {
            return Ok(w_obj);
        }
        let Some(w_descr) = lookup(w_obj, "__str__") else {
            return Err(crate::PyError::type_error(format!(
                "'{}' object has no __str__",
                object_functionstr_type_name(w_obj),
            )));
        };
        crate::call::call_function_impl_result(w_descr, &[w_obj])
    }
}

/// `object_functionstr`-local version of
/// `baseobjspace.py:878-885 findattr`.
///
/// ```python
/// def findattr(self, w_object, w_name):
///     try:
///         return self.getattr(w_object, w_name)
///     except OperationError as e:
///         # a PyPy extension: let SystemExit and KeyboardInterrupt go through
///         if e.async(self):
///             raise
///         return None
/// ```
///
/// `Err(_)` carries the propagated async exception
/// (`PyErrorKind::SystemExit`, mirroring `OperationError.async`'s
/// SystemExit/KeyboardInterrupt arm — `error.py:62-65`).  Pyre's
/// `PyError` does not yet carry a `KeyboardInterrupt` kind; SystemExit
/// alone covers the propagation contract for the cases pyre raises
/// today.  Ordinary `OperationError`s (AttributeError, NameError,
/// TypeError from descriptors) collapse to `Ok(None)`, matching
/// PyPy's `return None` arm.
fn object_functionstr_findattr(
    obj: PyObjectRef,
    name: &str,
) -> Result<Option<PyObjectRef>, crate::PyError> {
    if unsafe { is_none(obj) } {
        return Ok(None);
    }
    match getattr_str(obj, name) {
        Ok(value) => Ok(Some(value)),
        Err(e) if e.kind == crate::PyErrorKind::SystemExit => Err(e),
        Err(_) => Ok(None),
    }
}

/// `space.text_w(w_obj)` for the `object_functionstr` try blocks.
fn object_functionstr_text_w(w_obj: PyObjectRef) -> Result<String, crate::PyError> {
    unsafe {
        if pyre_object::is_str(w_obj) {
            Ok(pyre_object::w_str_get_value(w_obj).to_string())
        } else {
            Err(crate::PyError::type_error(format!(
                "expected str, got {} object",
                object_functionstr_type_name(w_obj),
            )))
        }
    }
}

pub(crate) fn object_functionstr_type_name(w_obj: PyObjectRef) -> String {
    unsafe {
        match crate::typedef::r#type(w_obj) {
            Some(tp) => pyre_object::w_type_get_name(tp).to_string(),
            None => "object".to_string(),
        }
    }
}

/// pypy/objspace/descroperation.py:319-326 `is_iterable`.
///
/// ```python
/// def is_iterable(space, w_obj):
///     w_descr = space.lookup(w_obj, '__iter__')
///     if w_descr is None:
///         if space.type(w_obj).flag_map_or_seq != 'M':
///             w_descr = space.lookup(w_obj, '__getitem__')
///         if w_descr is None:
///             return False
///     return True
/// ```
///
/// PyPy's `space.lookup` walks the type's MRO without firing
/// descriptors or `__getattr__`; pyre's `lookup` (`baseobjspace.rs:3945`)
/// has the same MRO-only semantics.  Using `findattr` here would run
/// the descriptor protocol and could surface false positives or
/// side effects in the *args error path, so we route through `lookup`
/// to match upstream exactly.
///
/// `flag_map_or_seq` is read off the resolved `space.type(w_obj)` and
/// gates the `__getitem__` fallback exactly as PyPy does — when
/// `'M'` (mapping) the fallback is skipped so a mapping-shaped type
/// without `__iter__` is reported as not iterable.  Pyre reads the
/// marker from `W_TypeObject`, the same level where PyPy stores
/// `flag_map_or_seq`.
///
/// Builtin shortcuts list/tuple/str/bytes/dict/set/iter/generator/
/// itertools mirror `iter()`'s direct-type arms at
/// `baseobjspace.rs:5158-5208`.
///
/// # Safety
/// Callers may pass any `PyObjectRef`; the function dereferences via
/// the same checks `iter()` uses (null-check, type-tag check) and
/// never reads through a dangling pointer beyond what existing pyre
/// type-tag helpers guarantee.
/// baseobjspace.py:1316-1323 `ismapping_w`.
///
/// ```python
/// def ismapping_w(self, w_obj):
///     flag = self.type(w_obj).flag_map_or_seq
///     if flag == 'M':
///         return True
///     elif flag == 'S':
///         return False
///     else:
///         return self.lookup(w_obj, '__getitem__') is not None
/// ```
///
/// The `is_dict` arm short-circuits the builtin mapping whose
/// `W_TypeObject` flag may not be reachable through `typedef::r#type`;
/// heap types (dict subclasses included) carry the inherited flag via
/// `inherit_flag_map_or_seq`.
pub fn ismapping_w(w_obj: PyObjectRef) -> bool {
    unsafe {
        if is_dict(w_obj) {
            return true;
        }
        let w_type = crate::typedef::r#type(w_obj).unwrap_or(std::ptr::null_mut());
        let flag = pyre_object::typeobject::w_type_get_flag_map_or_seq(w_type);
        if flag == b'M' {
            return true;
        }
        if flag == b'S' {
            return false;
        }
        lookup(w_obj, "__getitem__").is_some()
    }
}

pub fn is_iterable(w_obj: PyObjectRef) -> bool {
    let obj = unwrap_cell(w_obj);
    if obj.is_null() {
        return false;
    }
    unsafe {
        if is_list(obj)
            || is_tuple(obj)
            || is_str(obj)
            || pyre_object::bytesobject::is_bytes_like(obj)
            || is_dict(obj)
            || pyre_object::is_set_or_frozenset(obj)
            || is_range_iter(obj)
            || pyre_object::is_long_range_iter(obj)
            || is_seq_iter(obj)
            || pyre_object::generator::is_generator(obj)
            || pyre_object::interp_itertools::is_count(obj)
            || pyre_object::interp_itertools::is_repeat(obj)
            || pyre_object::interp_itertools::is_takewhile(obj)
            || pyre_object::interp_itertools::is_dropwhile(obj)
            || pyre_object::interp_itertools::is_filterfalse(obj)
            || pyre_object::interp_itertools::is_pairwise(obj)
            || pyre_object::interp_itertools::is_cycle(obj)
        {
            return true;
        }
        // descroperation.py:320 — `space.lookup(w_obj, '__iter__')`.
        // MRO-only walk; no `__getattr__` / descriptor execution.
        if lookup(w_obj, "__iter__").is_some() {
            return true;
        }
        // descroperation.py:322-323 — fallback to `__getitem__` only
        // when `space.type(w_obj).flag_map_or_seq != 'M'` (i.e. the
        // type is not flagged as a mapping).  Mapping types report
        // not-iterable when they don't supply `__iter__`.  The flag
        // lives on `W_TypeObject` (typeobject.py:169) so user-defined
        // `dict`/`list`/`tuple` subclasses inherit the marker via
        // `inherit_flag_map_or_seq` at heap-type construction.
        let w_type = crate::typedef::r#type(w_obj).unwrap_or(std::ptr::null_mut());
        let is_mapping = pyre_object::typeobject::w_type_get_flag_map_or_seq(w_type) == b'M';
        if !is_mapping && lookup(w_obj, "__getitem__").is_some() {
            return true;
        }
    }
    false
}

/// pypy/interpreter/baseobjspace.py:1110-1116 `fixedview`.
///
/// ```python
/// def fixedview(self, w_iterable, expected_length=-1):
///     """ A fixed list view of w_iterable. Don't modify the result """
///     return make_sure_not_resized(self.unpackiterable(w_iterable,
///                                                      expected_length)[:])
///
/// fixedview_unroll = fixedview
/// ```
///
/// Pyre returns a `Vec<PyObjectRef>` directly; the
/// `make_sure_not_resized` annotation is an RPython JIT hint with no
/// runtime effect that translates to "treat the result as immutable
/// at the callsite", which Rust enforces via `&[PyObjectRef]` once
/// the caller binds the return value.
pub fn fixedview(
    w_iterable: PyObjectRef,
    expected_length: isize,
) -> Result<Vec<PyObjectRef>, crate::PyError> {
    unpackiterable(w_iterable, expected_length)
}

/// descroperation.py:343-345 — `iter()` requires the object returned by a
/// dispatched `__iter__` to itself be an iterator (`space.lookup(w_iterator,
/// '__next__') is not None`), raising TypeError otherwise.  `space.lookup` is
/// a type-MRO lookup: a `__next__` reachable only via `__getattr__` or the
/// instance dict does NOT qualify.  pyre's builtin iterators carry `__next__`
/// in the getattr_str method tables rather than the type dict, so a type miss
/// on a non-user object falls back to the bare `__getattribute__` form of
/// getattr_str (`call_getattr = false`): it reaches the builtin method-table
/// `__next__` but fires no `__getattr__` hook.  A user-class instance
/// (is_instance) is excluded so its instance dict is never consulted; a type
/// miss there is not an iterator.
/// The user-visible Python type name (`type(obj).__name__`) for error
/// messages — the `w_class` name rather than the shared builtin vtable name,
/// so a heap subclass reports its own name.
unsafe fn obj_type_name(obj: PyObjectRef) -> &'static str {
    match crate::typedef::r#type(obj) {
        Some(tp) => pyre_object::typeobject::w_type_get_name(tp),
        None => (*(*obj).ob_type).name,
    }
}

unsafe fn iter_check_is_iterator(w_iterator: PyObjectRef) -> PyResult {
    let w_type = crate::typedef::r#type(w_iterator).unwrap_or(std::ptr::null_mut());
    let has_next = if !w_type.is_null() && lookup_in_type_where(w_type, "__next__").is_some() {
        true
    } else if is_instance(w_iterator) {
        false
    } else {
        getattr_str_impl(w_iterator, "__next__", false).is_ok()
    };
    if has_next {
        Ok(w_iterator)
    } else {
        Err(PyError::type_error(format!(
            "iter() returned non-iterator of type '{}'",
            obj_type_name(w_iterator)
        )))
    }
}

/// `iter(obj)` — PyPy: space.iter(w_obj)
/// Calls __iter__ on the object if available.
pub fn iter(obj: PyObjectRef) -> PyResult {
    let obj = unwrap_cell(obj);
    if obj.is_null() {
        return Err(PyError::type_error("'NoneType' object is not iterable"));
    }
    // `pypy/objspace/std/dictproxyobject.py:41 descr_iter` →
    // `space.iter(self.w_mapping)`.
    let obj = unsafe {
        if pyre_object::is_dict_proxy(obj) {
            pyre_object::w_dict_proxy_get_mapping(obj)
        } else {
            obj
        }
    };
    // `pypy/objspace/std/dictmultiobject.py`
    // `W_BaseDictMultiIterObject` line-by-line port — pyre's
    // `W_BaseDictMultiIterObject`
    // captures the source dict + the version counter seen at iter()
    // time, then on each `next()` step compares against `w_dict.version`
    // and raises `RuntimeError("dictionary changed size during
    // iteration")` if the dict was mutated mid-iteration.
    unsafe {
        if pyre_object::dictmultiobject::is_dict_view(obj) {
            let kind = pyre_object::dictmultiobject::w_dict_view_get_kind(obj);
            let w_dict = pyre_object::dictmultiobject::w_dict_view_get_dict(obj);
            return Ok(pyre_object::dictmultiobject::w_dict_view_iterator_new(
                w_dict, kind,
            ));
        }
        // `dict_keyiterator` / `dict_valueiterator` / `dict_itemiterator`
        // — `__iter__` returns self per `dictmultiobject.py:1716-1717
        // `W_BaseDictMultiIterObject.descr_iter`.
        if pyre_object::dictmultiobject::is_dict_view_iterator(obj) {
            return Ok(obj);
        }
    }
    unsafe {
        // Builtin iterables.  An exact list/tuple uses the direct storage
        // iterator; a subclass may override `__iter__`, in which case the
        // override is dispatched.  The inherited base `list/tuple.__iter__`
        // (which itself calls back into `iter()`) is collapsed to the
        // storage iterator to avoid an infinite recursion.
        if is_list(obj) {
            if !pyre_object::is_exact_list(obj) {
                if let Some((src, method)) = lookup_where((*obj).w_class, "__iter__") {
                    if !std::ptr::eq(src, pyre_object::get_instantiate(&pyre_object::LIST_TYPE)) {
                        // descroperation.py:339-341 — an explicit
                        // `__iter__ = None` override marks the subclass
                        // non-iterable even though the lookup succeeds.
                        if is_none(method) {
                            return Err(PyError::type_error(format!(
                                "'{}' object is not iterable",
                                obj_type_name(obj)
                            )));
                        }
                        let w_iter = crate::call::call_function_impl_result(method, &[obj])?;
                        return iter_check_is_iterator(w_iter);
                    }
                }
            }
            return Ok(pyre_object::w_seq_iter_new(obj, w_list_len(obj)));
        }
        if is_tuple(obj) {
            if !pyre_object::is_exact_tuple(obj) {
                if let Some((src, method)) = lookup_where((*obj).w_class, "__iter__") {
                    if !std::ptr::eq(src, pyre_object::get_instantiate(&pyre_object::TUPLE_TYPE)) {
                        // descroperation.py:339-341 — an explicit
                        // `__iter__ = None` override marks the subclass
                        // non-iterable even though the lookup succeeds.
                        if is_none(method) {
                            return Err(PyError::type_error(format!(
                                "'{}' object is not iterable",
                                obj_type_name(obj)
                            )));
                        }
                        let w_iter = crate::call::call_function_impl_result(method, &[obj])?;
                        return iter_check_is_iterator(w_iter);
                    }
                }
            }
            return Ok(pyre_object::w_seq_iter_new(obj, w_tuple_len(obj)));
        }
        if pyre_object::is_generic_alias(obj) {
            // GenericAlias.__iter__ (`_pypy_generic_alias.py:108`) — `yield
            // _make_starred(self)`, a one-shot iterator over the starred copy.
            let starred = crate::_pypy_generic_alias::make_starred(obj)?;
            let list = w_list_new(vec![starred]);
            return Ok(pyre_object::w_seq_iter_new(list, 1));
        }
        if is_str(obj) {
            // Code-point count (not byte count) seeds the sequence
            // iterator, read straight from the cached length so a
            // lone-surrogate backing does not panic.
            let len = w_str_len(obj);
            return Ok(pyre_object::w_seq_iter_new(obj, len));
        }
        if pyre_object::bytesobject::is_bytes_like(obj) {
            let len = pyre_object::bytesobject::bytes_like_len(obj);
            let mut items = Vec::with_capacity(len);
            for i in 0..len {
                items.push(w_int_new(
                    pyre_object::bytesobject::bytes_like_getitem(obj, i) as i64,
                ));
            }
            let list = pyre_object::w_list_new(items);
            return Ok(pyre_object::w_seq_iter_new(list, len));
        }
        // dict → iterate over keys (`pypy/objspace/std/dictmultiobject.py
        // W_DictMultiObject.descr_iter` → `W_DictMultiIterKeysObject`).
        // For W_ModuleDictObject this dispatches through
        // `ModuleDictStrategy.getiterkeys` (`celldict.py:188-189`);
        // pyre's W_BaseDictMultiIterObject captures `startlen` at iter()
        // time and raises `RuntimeError("dictionary changed size
        // during iteration")` mid-iteration — matches PyPy's
        // `_check_modified` (`dictmultiobject.py:1716+`) without the
        // snapshot list materialisation.
        if is_dict(obj) {
            return Ok(pyre_object::dictmultiobject::w_dict_view_iterator_new(
                obj,
                pyre_object::dictmultiobject::DictViewKind::Keys,
            ));
        }
        // set / frozenset → iterate via stable insertion order (PyPy:
        // setobject.py W_BaseSetObject.descr_iter, W_BaseSetIterObject).
        if pyre_object::is_set_or_frozenset(obj) {
            let items = pyre_object::w_set_items(obj);
            let len = items.len();
            let key_list = pyre_object::w_list_new(items);
            return Ok(pyre_object::w_seq_iter_new(key_list, len));
        }
        // `range` sequence → a `rangeiterator` (machine-int, JIT) when the
        // bounds fit a word, else a `longrange_iterator`.
        if pyre_object::is_w_range(obj) {
            return Ok(pyre_object::w_range_iter(obj));
        }
        // Already an iterator
        if is_range_iter(obj)
            || pyre_object::is_long_range_iter(obj)
            || is_seq_iter(obj)
            || pyre_object::generator::is_generator(obj)
        {
            return Ok(obj);
        }
        // itertools native iterators — iter_w returns self.
        // PyPy: W_Count.iter_w / W_Repeat.iter_w / W_TakeWhile.iter_w /
        // W_DropWhile.iter_w / W_Filter.iter_w / W_Pairwise.iter_w
        if pyre_object::interp_itertools::is_count(obj)
            || pyre_object::interp_itertools::is_repeat(obj)
            || pyre_object::interp_itertools::is_takewhile(obj)
            || pyre_object::interp_itertools::is_dropwhile(obj)
            || pyre_object::interp_itertools::is_filterfalse(obj)
            || pyre_object::interp_itertools::is_pairwise(obj)
            || pyre_object::interp_itertools::is_cycle(obj)
        {
            return Ok(obj);
        }
        // `pypy/module/__builtin__/functional.py:277-278
        // W_Enumerate.descr___iter__` — `return self`.
        if pyre_object::functional::is_enumerate(obj) {
            return Ok(obj);
        }
        // `pypy/module/__builtin__/functional.py:371-372
        // W_ReversedIterator.descr___iter__` — `return self`.
        if pyre_object::functional::is_reversed(obj) {
            return Ok(obj);
        }
        // `pypy/module/__builtin__/functional.py:927-928 W_Filter.iter_w` —
        // `return self`.
        if pyre_object::functional::is_filter(obj) {
            return Ok(obj);
        }
        // `functional.py:846-847 W_Map.iter_w` / `:1019-1020 W_Zip.iter_w` —
        // `return self`.
        if pyre_object::functional::is_map(obj) || pyre_object::functional::is_zip(obj) {
            return Ok(obj);
        }
        // `pypy/module/_sre/interp_sre.py:915 W_SRE_Scanner.iter_w` —
        // `return self` (the finditer/scanner iterator).
        if pyre_object::interp_sre::is_sre_scanner(obj) {
            return Ok(obj);
        }
        // `interp_struct.py:192 W_UnpackIter.descr_iter` — `return self`.
        if crate::module::r#struct::is_unpack_iter(obj) {
            return Ok(obj);
        }
        // `iter(callable, sentinel)` product — its own iterator.
        if pyre_object::operation::is_callable_iterator(obj) {
            return Ok(obj);
        }
        // `array.array` — `interp_array.py descr_iter` returns
        // `space.newseqiter(self)` (a fresh index cursor, not self).
        if pyre_object::interp_array::is_array(obj) {
            let len = pyre_object::interp_array::w_array_len(obj);
            return Ok(pyre_object::w_seq_iter_new(obj, len));
        }
        // `memoryview` — `memoryobject.py descr_iter` returns
        // `space.newseqiter(self)`; the cursor fetches each element through
        // `__getitem__`.  Element count is `shape[0]` == length / itemsize.
        if pyre_object::memoryview::is_w_memoryview(obj) {
            if pyre_object::memoryview::w_memoryview_released(obj) {
                return Err(PyError::value_error(
                    "operation forbidden on released memoryview object",
                ));
            }
            let itemsize = pyre_object::memoryview::w_memoryview_itemsize(obj);
            let len = if itemsize > 0 {
                (pyre_object::memoryview::w_memoryview_length(obj) / itemsize) as usize
            } else {
                0
            };
            return Ok(pyre_object::w_seq_iter_new(obj, len));
        }
        // pypy/objspace/descroperation.py:330-346 `def iter(space, w_obj)`
        // — `space.lookup(w_obj, '__iter__')` is type-MRO-only; PyPy never
        // consults the instance dict for special-method lookup (CPython
        // issue 5985 / typeobject `__iter__` slot resolution).  Earlier
        // pyre revisions also walked `getdict(obj)` and a per-object side
        // table, which surfaced per-instance `__iter__` writes (e.g.
        // `obj.__iter__ = method`); those paths are non-orthodox in
        // both CPython and PyPy and have been removed.
        if is_instance(obj) {
            let w_type = w_instance_get_type(obj);
            if let Some(method) = lookup_in_type_where(w_type, "__iter__") {
                // descroperation.py:339-341 — explicit `__iter__ = None`
                // marks the type as non-iterable even though the lookup
                // succeeds.
                if is_none(method) {
                    return Err(PyError::type_error(format!(
                        "'{}' object is not iterable",
                        (*(*obj).ob_type).name
                    )));
                }
                let w_iter = crate::call::call_function_impl_result(method, &[obj])?;
                return iter_check_is_iterator(w_iter);
            }
            // descroperation.py:333-334 — `__getitem__` fallback only when
            // `space.type(w_obj).flag_map_or_seq != 'M'`.  Mapping types
            // without `__iter__` are reported as non-iterable.  Read off
            // the user `W_TypeObject` (typeobject.py:169) so heap-type
            // dict/list/tuple subclasses inherit the marker — see
            // `is_iterable` (this file) for the same pattern.
            let w_user_type = crate::typedef::r#type(obj).unwrap_or(std::ptr::null_mut());
            let is_mapping =
                pyre_object::typeobject::w_type_get_flag_map_or_seq(w_user_type) == b'M';
            // descroperation.py:333-334 — `space.lookup(w_obj, '__getitem__')`
            // is a type-MRO lookup; special-method resolution never consults
            // the instance dict, so an `obj.__getitem__ = f` instance attribute
            // does not enable sequence iteration.
            if !is_mapping && lookup_in_type_where(w_type, "__getitem__").is_some() {
                // descroperation.py:334 — `space.newseqiter(w_obj)` wraps the
                // live object in an index cursor (iterobject.py
                // W_SeqIterObject) that fetches each item lazily through
                // `space.getitem` in `next` and ends on IndexError.  The
                // sequence is not materialised and `__len__` does not bound
                // the walk: the cursor advances until `__getitem__` raises
                // IndexError, so an unbounded `__getitem__` iterates forever
                // exactly as the builtin sequence iterator does.
                return Ok(pyre_object::w_seq_iter_new(obj, 0));
            }
        }
        // Type object: check metaclass __iter__ (NOT the type's own MRO)
        // PyPy/CPython: iter(X) calls type(X).__iter__(X), not X.__iter__
        // For type objects, type(X) is the metaclass.
        if is_type(obj) {
            // baseobjspace.py:76 — metaclass from w_class
            let w_metaclass = {
                let w_class = (*obj).w_class;
                let w_type_type = crate::typedef::w_type();
                if !w_class.is_null() && !std::ptr::eq(w_class, w_type_type) {
                    Some(w_class)
                } else {
                    None
                }
            };
            if let Some(w_metaclass) = w_metaclass {
                if let Some(method) = lookup_in_type_where(w_metaclass, "__iter__") {
                    let w_iter = crate::call::call_function_impl_result(method, &[obj])?;
                    return iter_check_is_iterator(w_iter);
                }
            }
            // Fallback: check type type's MRO
            if let Some(w_type_type) = crate::typedef::gettypefor(&pyre_object::pyobject::TYPE_TYPE)
            {
                if let Some(method) = lookup_in_type_where(w_type_type, "__iter__") {
                    let w_iter = crate::call::call_function_impl_result(method, &[obj])?;
                    return iter_check_is_iterator(w_iter);
                }
            }
        }
    }
    Err(PyError::type_error(format!(
        "'{}' object is not iterable",
        unsafe { (*(*obj).ob_type).name }
    )))
}

/// `next(iterator)` — PyPy: space.next(w_iter)
pub fn next(obj: PyObjectRef) -> PyResult {
    let obj = unwrap_cell(obj);
    unsafe {
        // Seq iterator
        if is_seq_iter(obj) {
            // Read through a raw pointer rather than a long-lived `&mut`: the
            // generic branch below runs Python (which can relocate `obj`), so
            // its writes go through a re-read pointer instead.
            let iter_ptr = obj as *mut pyre_object::W_SeqIterObject;
            let seq = (*iter_ptr).seq;
            // iterobject.py W_SeqIterObject.descr_next — a None (null) seq
            // marks an iterator already exhausted by an earlier IndexError.
            if seq.is_null() {
                return Err(PyError::stop_iteration());
            }
            let idx = (*iter_ptr).index;
            let item = if is_list(seq) {
                pyre_object::w_list_getitem(seq, idx)
            } else if is_tuple(seq) {
                pyre_object::w_tuple_getitem(seq, idx)
            } else if is_str(seq) {
                // Box the idx-th code point as a one-character str,
                // reading the WTF-8 view so a lone surrogate is yielded
                // instead of panicking.
                let s = w_str_get_wtf8(seq);
                let mut found: Option<PyObjectRef> = None;
                let mut n = 0i64;
                for cp in s.code_points() {
                    if n == idx {
                        let mut one = Wtf8Buf::new();
                        one.push(cp);
                        found = Some(w_str_from_wtf8(one));
                        break;
                    }
                    n += 1;
                }
                found
            } else if pyre_object::interp_array::is_array(seq) {
                if (idx as usize) < pyre_object::interp_array::w_array_len(seq) {
                    Some(pyre_object::interp_array::w_array_unpack_item(
                        seq,
                        idx as usize,
                    ))
                } else {
                    None
                }
            } else {
                // Generic sequence-protocol object: fetch lazily through
                // `space.getitem`.  `iterobject.c iter_iternext` treats BOTH
                // IndexError and StopIteration as exhaustion (clearing the
                // sequence so a later next() short-circuits without
                // re-invoking __getitem__), and propagates any OTHER error
                // WITHOUT clearing the sequence, leaving the iterator
                // retryable.  This intentionally differs from PyPy's
                // `W_SeqIterObject.descr_next` (iterobject.py:75-79), which
                // catches only IndexError and clears `w_seq` on every error;
                // the observable behaviour target is 3.14.
                //
                // `getitem` runs Python (`__getitem__`), which can relocate
                // `obj`, so pin it and read the iterator state back through the
                // re-read pointer rather than the now-stale `iter` reference.
                let _roots = pyre_object::gc_roots::push_roots();
                pyre_object::gc_roots::pin_root(obj);
                let obj_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
                match getitem(seq, w_int_new(idx)) {
                    Ok(v) => {
                        let p = pyre_object::gc_roots::shadow_stack_get(obj_slot)
                            as *mut pyre_object::W_SeqIterObject;
                        (*p).index += 1;
                        return Ok(v);
                    }
                    Err(e)
                        if e.kind == crate::PyErrorKind::IndexError
                            || e.kind == crate::PyErrorKind::StopIteration =>
                    {
                        let p = pyre_object::gc_roots::shadow_stack_get(obj_slot)
                            as *mut pyre_object::W_SeqIterObject;
                        (*p).seq = std::ptr::null_mut();
                        return Err(PyError::stop_iteration());
                    }
                    Err(e) => return Err(e),
                }
            };
            if let Some(v) = item {
                (*iter_ptr).index += 1;
                return Ok(v);
            }
            // iterobject.py:90-98 — an exhausted cursor clears its sequence ref
            // (the generic arm above already did so on IndexError); the builtin
            // arms clear it here so a held iterator reports as exhausted.
            (*iter_ptr).seq = std::ptr::null_mut();
            return Err(PyError::stop_iteration());
        }
        // Range iterator
        if is_range_iter(obj) {
            return match pyre_object::w_range_iter_next(obj) {
                Some(v) => Ok(v),
                None => Err(PyError::stop_iteration()),
            };
        }
        // `functional.py W_LongRangeIterator.descr_next` — bignum-bound
        // range cursor (`start + index*step`).
        if pyre_object::is_long_range_iter(obj) {
            return match pyre_object::w_long_range_iter_next(obj) {
                Some(v) => Ok(v),
                None => Err(PyError::stop_iteration()),
            };
        }
        // Generator __next__ — PyPy: generator.py GeneratorIterator.next
        if pyre_object::generator::is_generator(obj) {
            return generator_next(obj);
        }
        // itertools.count.next_w — PyPy interp_itertools.py W_Count.next_w
        //
        //     def next_w(self):
        //         w_c = self.w_c
        //         self.w_c = self.space.add(w_c, self.w_step)
        //         return w_c
        if pyre_object::interp_itertools::is_count(obj) {
            let w_c = pyre_object::interp_itertools::w_count_get_c(obj);
            let w_step = pyre_object::interp_itertools::w_count_get_step(obj);
            let new_c = add(w_c, w_step)?;
            pyre_object::interp_itertools::w_count_set_c(obj, new_c);
            return Ok(w_c);
        }
        // itertools.repeat.next_w — PyPy interp_itertools.py W_Repeat.next_w
        //
        //     def next_w(self):
        //         if self.counting:
        //             if self.count <= 0:
        //                 raise OperationError(self.space.w_StopIteration, self.space.w_None)
        //             self.count -= 1
        //         return self.w_obj
        if pyre_object::interp_itertools::is_repeat(obj) {
            if pyre_object::interp_itertools::w_repeat_get_counting(obj) {
                if pyre_object::interp_itertools::w_repeat_get_count(obj) <= 0 {
                    return Err(PyError::stop_iteration());
                }
                pyre_object::interp_itertools::w_repeat_dec_count(obj);
            }
            return Ok(pyre_object::interp_itertools::w_repeat_get_obj(obj));
        }
        // itertools.takewhile — interp_itertools.py W_TakeWhile.next_w
        //
        //     def next_w(self):
        //         if self.stopped:
        //             raise OperationError(self.space.w_StopIteration, self.space.w_None)
        //         w_obj = self.space.next(self.w_iterable)  # may raise a w_StopIteration
        //         w_bool = self.space.call_function(self.w_predicate, w_obj)
        //         if not self.space.is_true(w_bool):
        //             self.stopped = True
        //             raise OperationError(self.space.w_StopIteration, self.space.w_None)
        //         return w_obj
        if pyre_object::interp_itertools::is_takewhile(obj) {
            let it = &mut *(obj as *mut pyre_object::interp_itertools::W_TakeWhile);
            if it.stopped {
                return Err(PyError::stop_iteration());
            }
            let w_obj = next(it.w_iterable)?;
            let w_bool = crate::call::call_function_impl_result(it.w_predicate, &[w_obj])?;
            if !is_true(w_bool)? {
                it.stopped = true;
                return Err(PyError::stop_iteration());
            }
            return Ok(w_obj);
        }
        // itertools.dropwhile — interp_itertools.py W_DropWhile.next_w
        //
        //     def next_w(self):
        //         if self.started:
        //             w_obj = self.space.next(self.w_iterable)  # may raise w_StopIter
        //         else:
        //             while True:
        //                 w_obj = self.space.next(self.w_iterable)  # may raise w_StopIter
        //                 w_bool = self.space.call_function(self.w_predicate, w_obj)
        //                 if not self.space.is_true(w_bool):
        //                     self.started = True
        //                     break
        //         return w_obj
        if pyre_object::interp_itertools::is_dropwhile(obj) {
            let it = &mut *(obj as *mut pyre_object::interp_itertools::W_DropWhile);
            let w_obj = if it.started {
                next(it.w_iterable)?
            } else {
                loop {
                    let w_obj = next(it.w_iterable)?;
                    let w_bool = crate::call::call_function_impl_result(it.w_predicate, &[w_obj])?;
                    if !is_true(w_bool)? {
                        it.started = true;
                        break w_obj;
                    }
                }
            };
            return Ok(w_obj);
        }
        // itertools.filterfalse — W_Filter.next_w (functional.py:930) with
        // reverse=True; the trailing _filter_jitdriver loop applies the
        // same predicate test until an element passes.
        //
        //     def next_w(self):
        //         w_obj = self.space.next(self.w_iterable)  # may raise w_StopIteration
        //         if self.w_predicate is None:
        //             pred = self.space.is_true(w_obj)
        //         else:
        //             w_pred = self.space.call_function(self.w_predicate, w_obj)
        //             pred = self.space.is_true(w_pred)
        //         if pred ^ self.reverse:
        //             return w_obj
        if pyre_object::interp_itertools::is_filterfalse(obj) {
            let it = &mut *(obj as *mut pyre_object::interp_itertools::W_FilterFalse);
            loop {
                let w_obj = next(it.w_iterable)?;
                let pred = if it.w_predicate.is_null() {
                    is_true(w_obj)?
                } else {
                    let w_pred = crate::call::call_function_impl_result(it.w_predicate, &[w_obj])?;
                    is_true(w_pred)?
                };
                if !pred {
                    return Ok(w_obj);
                }
            }
        }
        // `pypy/module/__builtin__/functional.py:930-942 W_Filter.next_w`
        // (reverse=False): pull from the iterator until the predicate (or
        // truthiness, when None) passes.
        //
        //     def next_w(self):
        //         w_obj = self.space.next(self.w_iterable)  # may raise w_StopIteration
        //         if self.w_predicate is None:
        //             pred = self.space.is_true(w_obj)
        //         else:
        //             w_pred = self.space.call_function(self.w_predicate, w_obj)
        //             pred = self.space.is_true(w_pred)
        //         if pred ^ self.reverse:
        //             return w_obj
        if pyre_object::functional::is_filter(obj) {
            // `next`, the predicate, and `is_true`/`__bool__` all run Python and
            // can move the filter and the yielded item; pin the filter and re-read
            // its fields after each call, and pin the item across the predicate.
            let _roots = pyre_object::gc_roots::push_roots();
            pyre_object::gc_roots::pin_root(obj);
            let obj_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            loop {
                let w_iterable = (*(pyre_object::gc_roots::shadow_stack_get(obj_slot)
                    as *const pyre_object::functional::W_Filter))
                    .w_iterable;
                let w_obj = next(w_iterable)?;
                let _r = pyre_object::gc_roots::push_roots();
                pyre_object::gc_roots::pin_root(w_obj);
                let w_obj_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
                let w_predicate = (*(pyre_object::gc_roots::shadow_stack_get(obj_slot)
                    as *const pyre_object::functional::W_Filter))
                    .w_predicate;
                let pred = if w_predicate.is_null() {
                    is_true(pyre_object::gc_roots::shadow_stack_get(w_obj_slot))?
                } else {
                    let w_pred = crate::call::call_function_impl_result(
                        w_predicate,
                        &[pyre_object::gc_roots::shadow_stack_get(w_obj_slot)],
                    )?;
                    is_true(w_pred)?
                };
                if pred {
                    return Ok(pyre_object::gc_roots::shadow_stack_get(w_obj_slot));
                }
            }
        }
        // `functional.py:849-863 W_Map.next_w` — pull one item from each
        // sub-iterator, then `call(w_fun, *items)`; stop at the shortest
        // (strict raises on mismatch).
        if pyre_object::functional::is_map(obj) {
            use pyre_object::functional as mo;
            let w_iterators = mo::w_map_get_iterators(obj);
            let strict = mo::w_map_get_strict(obj);
            return match pull_iterator_tuple(w_iterators, strict, "map")? {
                Some(items) => {
                    let w_fun = mo::w_map_get_fun(obj);
                    crate::call::call_function_impl_result(w_fun, &items)
                }
                None => Err(PyError::stop_iteration()),
            };
        }
        // `functional.py:1022-1057 W_Zip.next_w` — pull one item from each
        // sub-iterator into a tuple; stop at the shortest (strict raises on
        // mismatch).
        if pyre_object::functional::is_zip(obj) {
            use pyre_object::functional as zo;
            let w_iterators = zo::w_zip_get_iterators(obj);
            let strict = zo::w_zip_get_strict(obj);
            return match pull_iterator_tuple(w_iterators, strict, "zip")? {
                Some(items) => Ok(pyre_object::w_tuple_new(items)),
                None => Err(PyError::stop_iteration()),
            };
        }
        // itertools.pairwise — interp_itertools.py W_Pairwise.next_w
        //
        //     def next_w(self):
        //         space = self.space
        //         w_prev = self.w_prev
        //         if w_prev is None:
        //             w_prev = space.next(self.w_iterator)
        //             self.w_prev = w_prev  # set before fetching w_next to handle reentrancy
        //         w_next = space.next(self.w_iterator)
        //         self.w_prev = w_next
        //         return space.newtuple2(w_prev, w_next)
        if pyre_object::interp_itertools::is_pairwise(obj) {
            let it = &mut *(obj as *mut pyre_object::interp_itertools::W_Pairwise);
            let mut w_prev = it.w_prev;
            if w_prev.is_null() {
                w_prev = next(it.w_iterator)?;
                // set before fetching w_next to handle reentrancy
                it.w_prev = w_prev;
            }
            let w_next = next(it.w_iterator)?;
            it.w_prev = w_next;
            return Ok(pyre_object::w_tuple_new(vec![w_prev, w_next]));
        }
        // itertools.cycle — interp_itertools.py W_Cycle.next_w
        //
        //     def next_w(self):
        //         if self.index > 0:
        //             if not self.saved_w:
        //                 raise OperationError(self.space.w_StopIteration, ...)
        //             try:
        //                 w_obj = self.saved_w[self.index]
        //             except IndexError:
        //                 self.index = 1
        //                 w_obj = self.saved_w[0]
        //             else:
        //                 self.index += 1
        //         else:
        //             try:
        //                 w_obj = self.space.next(self.w_iterable)
        //             except OperationError as e:  # StopIteration
        //                 self.index = 1
        //                 if not self.saved_w:
        //                     raise
        //                 w_obj = self.saved_w[0]
        //             else:
        //                 self.saved_w.append(w_obj)
        //         return w_obj
        if pyre_object::interp_itertools::is_cycle(obj) {
            let it = &mut *(obj as *mut pyre_object::interp_itertools::W_Cycle);
            // Cycling pass (index > 0): replay `saved` after the source ended.
            if it.index > 0 {
                let n = pyre_object::w_list_len(it.saved) as i64;
                if n == 0 {
                    return Err(PyError::stop_iteration());
                }
                if it.index < n {
                    let w_obj = pyre_object::w_list_getitem(it.saved, it.index)
                        .expect("cycle saved index in range");
                    it.index += 1;
                    return Ok(w_obj);
                }
                // `IndexError` — wrap to the start; index left at 1 so the
                // next call reads `saved[1]`.
                it.index = 1;
                return Ok(pyre_object::w_list_getitem(it.saved, 0).expect("cycle saved non-empty"));
            }
            // First pass (index == 0): pull from the source, saving each.
            match next(it.w_iterable) {
                Ok(w_obj) => {
                    pyre_object::w_list_append(it.saved, w_obj);
                    return Ok(w_obj);
                }
                Err(e) if e.kind == PyErrorKind::StopIteration => {
                    it.index = 1;
                    if pyre_object::w_list_len(it.saved) == 0 {
                        return Err(PyError::stop_iteration());
                    }
                    return Ok(
                        pyre_object::w_list_getitem(it.saved, 0).expect("cycle saved non-empty")
                    );
                }
                Err(e) => return Err(e),
            }
        }
        // `pypy/objspace/std/dictmultiobject.py:809-845 _new_next`
        // line-by-line — two parity-mandated checks:
        //
        //     if self.len != self.w_dict.length():
        //         raise oefmt(space.w_RuntimeError,
        //                     "dictionary changed size during iteration")
        //     ...
        //     if self.strategy is self.w_dict.get_strategy():
        //         return result      # common case
        //     else:
        //         # obscure: strategy changed but length is the same
        //         if TP == 'key' or TP == 'value':
        //             return result
        //         w_key = result[0]
        //         w_value = self.w_dict.getitem(w_key)
        //         if w_value is None:
        //             raise "dictionary changed during iteration"
        //         return (w_key, w_value)
        if pyre_object::dictmultiobject::is_dict_view_iterator(obj) {
            use pyre_object::dictmultiobject as dv;
            let dict = dv::w_dict_view_iterator_get_dict(obj);
            let startlen = dv::w_dict_view_iterator_get_startlen(obj);
            let current_len = pyre_object::dictmultiobject::w_dict_len(dict);
            if startlen != current_len {
                return Err(PyError::new(
                    PyErrorKind::RuntimeError,
                    "dictionary changed size during iteration".to_string(),
                ));
            }
            let index = dv::w_dict_view_iterator_get_index(obj);
            let items = pyre_object::dictmultiobject::w_dict_items(dict);
            if index >= items.len() {
                return Err(PyError::stop_iteration());
            }
            let (k, mut v) = items[index];
            dv::w_dict_view_iterator_set_index(obj, index + 1);
            // `:829-841` strategy-transition handling.
            let start_strategy_id = dv::w_dict_view_iterator_get_start_strategy_id(obj);
            let current_strategy_id = pyre_object::dictmultiobject::w_dict_strategy_id(dict);
            let kind = dv::w_dict_view_iterator_get_kind(obj);
            if start_strategy_id != current_strategy_id {
                if matches!(kind, pyre_object::dictmultiobject::DictViewKind::Items) {
                    // `:837-841`: re-look-up the key on the new strategy;
                    // raise if it was removed during the transition.
                    match pyre_object::dictmultiobject::w_dict_lookup(dict, k) {
                        Some(fresh) => v = fresh,
                        None => {
                            return Err(PyError::new(
                                PyErrorKind::RuntimeError,
                                "dictionary changed during iteration".to_string(),
                            ));
                        }
                    }
                }
                // Keys / Values iterators return the cached entry as-is
                // (`:836 if TP == 'key' or TP == 'value': return result`).
            }
            return Ok(match kind {
                pyre_object::dictmultiobject::DictViewKind::Keys => k,
                pyre_object::dictmultiobject::DictViewKind::Values => v,
                pyre_object::dictmultiobject::DictViewKind::Items => {
                    pyre_object::w_tuple_new(vec![k, v])
                }
            });
        }
        // `pypy/module/__builtin__/functional.py:280-310 W_Enumerate.descr_next`
        // line-by-line port —
        //
        //     def descr_next(self, space):
        //         w_index = self.w_index
        //         w_iter_or_list = self.w_iter_or_list
        //         w_item = None
        //         if w_index is None:
        //             index = self.index
        //             if type(w_iter_or_list) is W_ListObject:
        //                 try:
        //                     w_item = w_iter_or_list.getitem(index)
        //                 except IndexError:
        //                     self.w_iter_or_list = None
        //                     raise OperationError(space.w_StopIteration, space.w_None)
        //                 self.index = index + 1
        //             elif w_iter_or_list is None:
        //                 raise OperationError(space.w_StopIteration, space.w_None)
        //             else:
        //                 try:
        //                     newval = rarithmetic.ovfcheck(index + 1)
        //                 except OverflowError:
        //                     w_index = space.newint(index)
        //                     self.w_index = space.add(w_index, space.newint(1))
        //                     self.index = -1
        //                 else:
        //                     self.index = newval
        //             w_index = space.newint(index)
        //         else:
        //             self.w_index = space.add(w_index, space.newint(1))
        //         if w_item is None:
        //             w_item = space.next(self.w_iter_or_list)
        //         return space.newtuple2(w_index, w_item)
        // `iter(callable, sentinel)` product
        // (`__builtin__/operation.py:128 _CallableIterator.__next__`):
        // invoke the zero-arg callable; stop when the result equals the
        // sentinel.  Once exhausted, `callable` is latched to `PY_NULL`
        // so further `next()` keeps raising.  The latch is also set when
        // the callable itself raises `StopIteration`, and a re-entrant
        // call that exhausts the iterator during `callable()` discards
        // this call's result.
        if pyre_object::operation::is_callable_iterator(obj) {
            use pyre_object::operation as ci;
            let callable = ci::w_callable_iterator_get_callable(obj);
            if callable.is_null() {
                return Err(PyError::stop_iteration());
            }
            let result = match crate::call::call_function_impl_result(callable, &[]) {
                Ok(r) => r,
                Err(e) if e.kind == crate::PyErrorKind::StopIteration => {
                    // `calliter_iternext`: when the callable itself raises
                    // `StopIteration`, latch `it_callable` to `PY_NULL` so
                    // further `next()` stays stopped, then re-raise.
                    ci::w_callable_iterator_set_callable(obj, pyre_object::PY_NULL);
                    return Err(e);
                }
                Err(e) => return Err(e),
            };
            // Re-check `it_callable` after the call: the callable may have
            // re-entered `next()` on this same iterator and latched it to
            // `PY_NULL`, exhausting it; discard the result and stay stopped
            // rather than comparing a stale value to the sentinel.
            if ci::w_callable_iterator_get_callable(obj).is_null() {
                return Err(PyError::stop_iteration());
            }
            let sentinel = ci::w_callable_iterator_get_sentinel(obj);
            if is_true(compare(result, sentinel, CompareOp::Eq)?)? {
                ci::w_callable_iterator_set_callable(obj, pyre_object::PY_NULL);
                return Err(PyError::stop_iteration());
            }
            return Ok(result);
        }
        if pyre_object::functional::is_enumerate(obj) {
            use pyre_object::functional as eo;
            let w_index_slot = eo::w_enumerate_get_w_index(obj);
            let mut w_iter_or_list = eo::w_enumerate_get_iter_or_list(obj);
            let mut w_item: PyObjectRef = pyre_object::PY_NULL;
            let w_index: PyObjectRef;
            if w_index_slot.is_null() {
                // i64 fast-path branch.
                let index = eo::w_enumerate_get_index(obj);
                if !w_iter_or_list.is_null() && pyre_object::is_list(w_iter_or_list) {
                    // `:289-294 W_ListObject` fast path — directly
                    // getitem; IndexError marks end-of-iteration and
                    // clears the slot.
                    let list_len = pyre_object::w_list_len(w_iter_or_list) as i64;
                    if index < 0 || index >= list_len {
                        eo::w_enumerate_set_iter_or_list(obj, pyre_object::PY_NULL);
                        return Err(PyError::stop_iteration());
                    }
                    w_item = pyre_object::w_list_getitem(w_iter_or_list, index).unwrap_or(PY_NULL);
                    eo::w_enumerate_set_index(obj, index + 1);
                } else if w_iter_or_list.is_null() {
                    // `:295-296` — slot cleared after a previous
                    // list-getitem stop.
                    return Err(PyError::stop_iteration());
                } else {
                    // General iterator path — `:297-303` ovfcheck.
                    match index.checked_add(1) {
                        Some(next) => eo::w_enumerate_set_index(obj, next),
                        None => {
                            // Promote to bigint slot per `:299-302`.
                            let w_idx =
                                pyre_object::w_long_new(::malachite_bigint::BigInt::from(index));
                            let one =
                                pyre_object::w_long_new(::malachite_bigint::BigInt::from(1i64));
                            let bumped = add(w_idx, one)?;
                            eo::w_enumerate_set_w_index(obj, bumped);
                            eo::w_enumerate_set_index(obj, -1);
                        }
                    }
                }
                w_index = pyre_object::w_int_new(index);
            } else {
                // Bigint slot active — bump via `space.add`.
                let one = pyre_object::w_int_new(1);
                let bumped = add(w_index_slot, one)?;
                eo::w_enumerate_set_w_index(obj, bumped);
                w_index = w_index_slot;
            }
            if w_item.is_null() {
                // Re-read slot — list fast-path already set w_item;
                // otherwise we need to pull from the iterator.
                w_iter_or_list = eo::w_enumerate_get_iter_or_list(obj);
                if w_iter_or_list.is_null() {
                    return Err(PyError::stop_iteration());
                }
                w_item = next(w_iter_or_list)?;
            }
            return Ok(pyre_object::w_tuple_new(vec![w_index, w_item]));
        }
        // `pypy/module/__builtin__/functional.py:385-405
        // W_ReversedIterator.descr_next` — `getitem(sequence, remaining)`
        // then decrement; IndexError / StopIteration ends the walk and
        // clears the slot.
        if pyre_object::functional::is_reversed(obj) {
            use pyre_object::functional as ro;
            let remaining = ro::w_reversed_get_remaining(obj);
            if remaining >= 0 {
                // `getitem` runs `__getitem__` (Python) and can move the reversed
                // iterator; pin it and write its state through the re-read pointer.
                let _roots = pyre_object::gc_roots::push_roots();
                pyre_object::gc_roots::pin_root(obj);
                let obj_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
                let seq = ro::w_reversed_get_sequence(obj);
                match getitem(seq, w_int_new(remaining)) {
                    Ok(w_item) => {
                        ro::w_reversed_set_remaining(
                            pyre_object::gc_roots::shadow_stack_get(obj_slot),
                            remaining - 1,
                        );
                        return Ok(w_item);
                    }
                    Err(e) => {
                        let obj = pyre_object::gc_roots::shadow_stack_get(obj_slot);
                        ro::w_reversed_set_remaining(obj, -1);
                        ro::w_reversed_set_sequence(obj, pyre_object::PY_NULL);
                        if e.kind == PyErrorKind::IndexError || e.kind == PyErrorKind::StopIteration
                        {
                            return Err(PyError::stop_iteration());
                        }
                        return Err(e);
                    }
                }
            }
            ro::w_reversed_set_remaining(obj, -1);
            ro::w_reversed_set_sequence(obj, pyre_object::PY_NULL);
            return Err(PyError::stop_iteration());
        }
        // `pypy/module/_sre/interp_sre.py:918 W_SRE_Scanner.next_w` —
        // search from the current position, yielding the match object.
        if pyre_object::interp_sre::is_sre_scanner(obj) {
            return crate::module::_sre::interp_sre::sre_scanner_next(obj);
        }
        // `interp_struct.py:195 W_UnpackIter.descr_next` — dispatch the
        // iterator's own `__next__` from its type.
        if crate::module::r#struct::is_unpack_iter(obj) {
            if let Some(w_type) = crate::typedef::r#type(obj) {
                if let Some(method) = lookup_in_type_where(w_type, "__next__") {
                    return crate::call::call_function_impl_result(method, &[obj]);
                }
            }
        }
        // Instance __next__
        if is_instance(obj) {
            let w_type = w_instance_get_type(obj);
            if let Some(method) = lookup_in_type_where(w_type, "__next__") {
                return crate::call::call_function_impl_result(method, &[obj]);
            }
        }
    }
    Err(PyError::type_error("not an iterator"))
}

/// `descriptor.py:256-273 W_Property._copy` — clone the property with
/// one accessor replaced.  A getter-inherited doc (`getter_doc`) does
/// not survive a getter replacement (descriptor.py:263-266); the
/// constructor's doc capture then re-derives it from the new getter.
unsafe fn property_copy(
    prop: PyObjectRef,
    w_getter: Option<PyObjectRef>,
    w_setter: Option<PyObjectRef>,
    w_deleter: Option<PyObjectRef>,
) -> PyResult {
    let w_none = pyre_object::w_none();
    let resolve = |slot: Option<PyObjectRef>, get: unsafe fn(PyObjectRef) -> PyObjectRef| {
        let v = slot.unwrap_or_else(|| unsafe { get(prop) });
        if v.is_null() { w_none } else { v }
    };
    let getter = resolve(w_getter, w_property_get_fget);
    let setter = resolve(w_setter, w_property_get_fset);
    let deleter = resolve(w_deleter, w_property_get_fdel);
    let getter_doc = (*(prop as *const pyre_object::descriptor::W_Property)).getter_doc;
    // descriptor.py:263-264 `if self.getter_doc and w_getter is not None`
    // — judged on the getter AFTER defaulting from `w_fget`, so
    // `.setter(s)` on a getter_doc property still passes doc=None and
    // re-derives it from the kept getter (getter_doc stays inherited).
    let w_doc = if getter_doc && !is_none(getter) {
        w_none
    } else {
        let d = pyre_object::descriptor::w_property_get_doc(prop);
        if d.is_null() { w_none } else { d }
    };
    // descriptor.py:267-269 `w_type = self.getclass(space); space.call_function(
    // w_type, w_getter, w_setter, w_deleter, w_doc)` — construct through the
    // instance's own type so a `property` subclass is preserved; the
    // constructor re-runs the doc capture.
    let w_type = crate::typedef::r#type(prop)
        .unwrap_or_else(|| crate::typedef::gettypeobject(&pyre_object::descriptor::PROPERTY_TYPE));
    let w_res = crate::call::call_function_impl_result(w_type, &[getter, setter, deleter, w_doc])?;
    // descriptor.py:270-271 `if isinstance(w_res, W_Property): w_res.w_name
    // = self.w_name` — the copy keeps the source's name.
    if is_property(w_res) {
        let w_name = pyre_object::descriptor::w_property_get_name(prop);
        if !w_name.is_null() {
            pyre_object::descriptor::w_property_set_name(w_res, w_name);
        }
    }
    Ok(w_res)
}

/// `descriptor.py:206-217 W_Property._properror` — AttributeError naming
/// the missing accessor and, when known, the property's `__name__`.
unsafe fn property_no_accessor(prop: PyObjectRef, obj: PyObjectRef, kind: &str) -> crate::PyError {
    let qualname = match crate::typedef::r#type(obj) {
        Some(w_type) => match getattr_str(w_type, "__qualname__") {
            Ok(q) if !q.is_null() && is_str(q) => pyre_object::w_str_get_value(q).to_string(),
            _ => pyre_object::w_type_get_name(w_type).to_string(),
        },
        None => (*(*obj).ob_type).name.to_string(),
    };
    let w_name = pyre_object::descriptor::w_property_get_name(prop);
    let msg = if !w_name.is_null() {
        let name_repr = crate::display::py_repr(w_name).unwrap_or_else(|_| "<name>".to_string());
        format!("property {name_repr} of '{qualname}' object has no {kind}")
    } else {
        format!("property of '{qualname}' object has no {kind}")
    };
    crate::PyError::attribute_error(msg)
}

fn property_set_name_impl(args: &[PyObjectRef]) -> PyResult {
    // descriptor.py:274-276 `set_name(self, w_type, w_name)` — the bound
    // method receives `[property, owner, name]`.
    if let Some(&w_name) = args.get(2) {
        unsafe { pyre_object::descriptor::w_property_set_name(args[0], w_name) };
    }
    Ok(pyre_object::w_none())
}

fn property_setter_impl(args: &[PyObjectRef]) -> PyResult {
    // descriptor.py:243-244 `setter` → `_copy(space, w_setter=w_setter)`
    unsafe { property_copy(args[0], None, args.get(1).copied(), None) }
}

fn property_getter_impl(args: &[PyObjectRef]) -> PyResult {
    // descriptor.py:240-241 `getter` → `_copy(space, w_getter=w_getter)`
    unsafe { property_copy(args[0], args.get(1).copied(), None, None) }
}

fn property_deleter_impl(args: &[PyObjectRef]) -> PyResult {
    // descriptor.py:246-247 `deleter` → `_copy(space, w_deleter=w_deleter)`
    unsafe { property_copy(args[0], None, None, args.get(1).copied()) }
}

/// `descriptor.py W_Property.get` as the type-dict `__get__` entry — args
/// are `[property, obj, objtype]`.  The implicit descriptor path special-
/// cases properties in `get()` above; this entry exists so explicit
/// `prop.__get__(...)`, `hasattr(prop, '__get__')`, and descriptor
/// introspection see it.
pub(crate) fn property_descr_get_impl(args: &[PyObjectRef]) -> PyResult {
    unsafe {
        let prop = args[0];
        let obj = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
        // `if space.is_w(w_obj, space.w_None): return self`
        if obj.is_null() || is_none(obj) {
            return Ok(prop);
        }
        let fget = w_property_get_fget(prop);
        if fget.is_null() || is_none(fget) {
            return Err(property_no_accessor(prop, obj, "getter"));
        }
        crate::call::call_function_impl_result(fget, &[obj])
    }
}

/// `descriptor.py W_Property.set` — args `[property, obj, value]`.
pub(crate) fn property_descr_set_impl(args: &[PyObjectRef]) -> PyResult {
    unsafe {
        let prop = args[0];
        let obj = args[1];
        let value = args.get(2).copied().unwrap_or(pyre_object::PY_NULL);
        let fset = w_property_get_fset(prop);
        if fset.is_null() || is_none(fset) {
            return Err(property_no_accessor(prop, obj, "setter"));
        }
        crate::call::call_function_impl_result(fset, &[obj, value])?;
        Ok(pyre_object::w_none())
    }
}

/// `descriptor.py W_Property.delete` — args `[property, obj]`.
pub(crate) fn property_descr_delete_impl(args: &[PyObjectRef]) -> PyResult {
    unsafe {
        let prop = args[0];
        let obj = args[1];
        let fdel = w_property_get_fdel(prop);
        if fdel.is_null() || is_none(fdel) {
            return Err(property_no_accessor(prop, obj, "deleter"));
        }
        crate::call::call_function_impl_result(fdel, &[obj])?;
        Ok(pyre_object::w_none())
    }
}

// ── Generator methods ────────────────────────────────────────────────
//
// PyPy: pypy/interpreter/generator.py GeneratorIterator
//
// send_ex(w_arg, operr) is the core resume path.
// - __next__() → send_ex(None, None)
// - send(v)    → send_ex(v, None)
// - throw(t,v) → send_ex(None, OperationError(t,v))
// - close()    → throw(GeneratorExit) then check result

/// PyPy: GeneratorIterator._send_ex(w_arg, operr)
///
/// Resume a generator frame: push w_arg (for send/next) or inject operr
/// (for throw), then run the frame until YIELD_VALUE or RETURN_VALUE.
fn generator_send_ex(gen_obj: PyObjectRef, w_arg: PyObjectRef, operr: Option<PyError>) -> PyResult {
    use pyre_object::generator::*;
    unsafe {
        if w_generator_is_running(gen_obj) {
            return Err(PyError::value_error("generator already executing"));
        }

        if w_generator_is_exhausted(gen_obj) {
            if let Some(err) = operr {
                return Err(err);
            }
            return Err(PyError::stop_iteration());
        }

        let frame_ptr = w_generator_get_frame(gen_obj) as *mut crate::pyframe::PyFrame;
        if frame_ptr.is_null() {
            w_generator_set_exhausted(gen_obj);
            if let Some(err) = operr {
                return Err(err);
            }
            return Err(PyError::stop_iteration());
        }
        let frame = &mut *frame_ptr;
        let already_started = w_generator_is_started(gen_obj);

        if !already_started {
            if operr.is_none() && !w_arg.is_null() && !is_none(w_arg) {
                return Err(PyError::type_error(
                    "can't send non-None value to a just-started generator",
                ));
            }
        }
        w_generator_set_started(gen_obj);
        w_generator_set_running(gen_obj, true);

        // generator.py:104 — w_result = frame.execute_frame(w_arg, operr)
        let w_inputvalue = if already_started && operr.is_none() {
            Some(w_arg)
        } else {
            None
        };
        let result = frame.execute_frame(w_inputvalue, operr);

        w_generator_set_running(gen_obj, false);

        match result {
            Ok(value) => {
                // generator.py:109-114 — if the frame marked itself finished,
                // it was RETURNed from; otherwise it YIELDed.
                if frame.frame_finished_execution {
                    w_generator_set_exhausted(gen_obj);
                    // generator.py:117-119 / pyopcode.py RETURN_VALUE in
                    // generator frames — `raise StopIteration(returnvalue)`
                    // so callers can pull the return value off `.value`.
                    // Wrap any non-None return into the exception's args
                    // tuple; bare `return` (or fallthrough → None) keeps
                    // an empty args tuple.
                    Err(stop_iteration_with_value(value))
                } else {
                    Ok(value)
                }
            }
            Err(e) => {
                w_generator_set_exhausted(gen_obj);
                // generator.py `_leak_stopiteration` (PEP 479) — a
                // StopIteration that escaped the body becomes RuntimeError;
                // any other error propagates unchanged.  The normal-return
                // StopIteration is built in the `Ok`/`frame_finished_execution`
                // arm above and never reaches here.
                if e.kind == crate::PyErrorKind::StopIteration {
                    Err(leak_stopiteration(e))
                } else {
                    Err(e)
                }
            }
        }
    }
}

/// Build a `StopIteration` carrying `value` on `.value` / `args[0]`.
/// `value == None` (or PY_NULL) keeps the args tuple empty so
/// `next(g)` outside a generator-return context still surfaces a bare
/// `StopIteration()`.
fn stop_iteration_with_value(value: PyObjectRef) -> PyError {
    use pyre_object::interp_exceptions::*;
    let exc = w_exception_new(ExcKind::StopIteration, "");
    if !value.is_null() && unsafe { !is_none(value) } {
        // `interp_exceptions.py:121-124 W_BaseException.descr_init`
        // stores `args_w` as a list; pyre matches the shape so that
        // `e.args` materialises a fresh tuple each read.
        let args_list = w_list_new(vec![value]);
        unsafe {
            w_exception_set_args(exc, args_list);
        }
    }
    unsafe { PyError::from_exc_object(exc) }
}

/// generator.py:131-138 `_invoke_execute_frame` / `_leak_stopiteration`
/// (PEP 479): a `StopIteration` that escapes the generator body — whether
/// raised explicitly or leaked from a `next()` inside the body — is replaced
/// by `RuntimeError("generator raised StopIteration")` chained from it
/// (`__cause__` and `__context__` both point at the leaked exception, and the
/// cause suppresses the context in display, mirroring
/// `chain_exceptions_from_cause`).  This is distinct from a normal generator
/// return, which surfaces through the `Ok`/`frame_finished_execution` path.
unsafe fn leak_stopiteration(e: PyError) -> PyError {
    use pyre_object::interp_exceptions::*;
    let w_stopiter = e.to_exc_object();
    let rt = w_exception_new(ExcKind::RuntimeError, "generator raised StopIteration");
    if pyre_object::is_exception(rt) && !w_stopiter.is_null() {
        w_exception_set_context(rt, w_stopiter);
        w_exception_set_cause(rt, w_stopiter);
        w_exception_set_suppress_context(rt, true);
    }
    PyError::from_exc_object(rt)
}

/// PyPy: GeneratorIterator.next() — equivalent to __next__
fn generator_next(gen_obj: PyObjectRef) -> PyResult {
    generator_send_ex(gen_obj, w_none(), None)
}

/// __next__ method wrapper
fn generator_next_method(args: &[PyObjectRef]) -> PyResult {
    let gen_obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
    generator_next(gen_obj)
}

/// Generic __next__ wrapper for iterators that delegate to `next()`.
/// Used for itertools count/repeat etc.
fn iter_next_method(args: &[PyObjectRef]) -> PyResult {
    let obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
    next(obj)
}

/// `__iter__` for an iterator — returns the iterator itself.
fn iter_self_method(args: &[PyObjectRef]) -> PyResult {
    Ok(args.first().copied().unwrap_or(pyre_object::PY_NULL))
}

/// `takewhile.__reduce__` — `interp_itertools.py W_TakeWhile.descr_reduce`:
/// `(type(self), (predicate, iterable), stopped)`.
fn takewhile_reduce_method(args: &[PyObjectRef]) -> PyResult {
    let it = unsafe { &*(args[0] as *const pyre_object::interp_itertools::W_TakeWhile) };
    let w_type = crate::typedef::r#type(args[0]).unwrap_or(PY_NULL);
    let state = w_tuple_new(vec![it.w_predicate, it.w_iterable]);
    Ok(w_tuple_new(vec![w_type, state, w_bool_from(it.stopped)]))
}

/// `takewhile.__setstate__` — `interp_itertools.py W_TakeWhile.descr_setstate`:
/// `self.stopped = space.bool_w(w_state)`.  `space.bool_w(w)` is
/// `bool(int_w(w))` (baseobjspace.py:1944): it unwraps an int and
/// rejects non-ints, NOT the general `is_true` truth test, so
/// `int_w(...)? != 0` is the exact equivalent (raises on a non-int
/// state just as `bool_w` does).
fn takewhile_setstate_method(args: &[PyObjectRef]) -> PyResult {
    let it = unsafe { &mut *(args[0] as *mut pyre_object::interp_itertools::W_TakeWhile) };
    it.stopped = int_w(args.get(1).copied().unwrap_or(w_none()))? != 0;
    Ok(w_none())
}

/// `dropwhile.__reduce__` — `interp_itertools.py W_DropWhile.descr_reduce`:
/// `(type(self), (predicate, iterable), started)`.
fn dropwhile_reduce_method(args: &[PyObjectRef]) -> PyResult {
    let it = unsafe { &*(args[0] as *const pyre_object::interp_itertools::W_DropWhile) };
    let w_type = crate::typedef::r#type(args[0]).unwrap_or(PY_NULL);
    let state = w_tuple_new(vec![it.w_predicate, it.w_iterable]);
    Ok(w_tuple_new(vec![w_type, state, w_bool_from(it.started)]))
}

/// `dropwhile.__setstate__` — `interp_itertools.py W_DropWhile.descr_setstate`:
/// `self.started = space.bool_w(w_state)` (= `bool(int_w(w))`; see
/// `takewhile_setstate_method` for the `int_w(...)? != 0` equivalence).
fn dropwhile_setstate_method(args: &[PyObjectRef]) -> PyResult {
    let it = unsafe { &mut *(args[0] as *mut pyre_object::interp_itertools::W_DropWhile) };
    it.started = int_w(args.get(1).copied().unwrap_or(w_none()))? != 0;
    Ok(w_none())
}

/// `filterfalse.__reduce__` — `interp_itertools.py W_FilterFalse.descr_reduce`:
/// `(type(self), (None-or-predicate, iterable))` — no state element.
fn filterfalse_reduce_method(args: &[PyObjectRef]) -> PyResult {
    let it = unsafe { &*(args[0] as *const pyre_object::interp_itertools::W_FilterFalse) };
    let w_type = crate::typedef::r#type(args[0]).unwrap_or(PY_NULL);
    let w_pred = if it.w_predicate.is_null() {
        w_none()
    } else {
        it.w_predicate
    };
    let state = w_tuple_new(vec![w_pred, it.w_iterable]);
    Ok(w_tuple_new(vec![w_type, state]))
}

/// `count.__reduce__` — `interp_itertools.py W_Count.reduce_w`:
/// `(gettypefor(W_Count), (c,))` when `step` is an int instance equal to 1
/// (`single_argument`), else `(gettypefor(W_Count), (c, step))`.  count has
/// no `__setstate__`.
fn count_reduce_method(args: &[PyObjectRef]) -> PyResult {
    // reduce_w pickles to the exact `itertools.count` builtin type
    // (`space.gettypefor(W_Count)`), not the receiver's subclass type.
    let w_type =
        crate::typedef::gettypefor(&pyre_object::interp_itertools::COUNT_TYPE).unwrap_or(PY_NULL);
    let w_c = unsafe { pyre_object::interp_itertools::w_count_get_c(args[0]) };
    let w_step = unsafe { pyre_object::interp_itertools::w_count_get_step(args[0]) };
    // single_argument(): `isinstance_w(w_step, w_int) and eq_w(w_step,
    // newint(1))` -- an int (or int subclass) whose object-space value is 1.
    let w_int_type = crate::typedef::gettypefor(&pyre_object::INT_TYPE).unwrap_or(PY_NULL);
    let single = unsafe { isinstance_w(w_step, w_int_type) } && eq_w(w_step, w_int_new(1))?;
    let state = if single {
        w_tuple_new(vec![w_c])
    } else {
        w_tuple_new(vec![w_c, w_step])
    };
    Ok(w_tuple_new(vec![w_type, state]))
}

/// `repeat.__reduce__` — `interp_itertools.py W_Repeat.descr_reduce`:
/// `(gettypefor(W_Repeat), (obj, count))` when counting, else
/// `(gettypefor(W_Repeat), (obj,))`.  repeat has no `__setstate__`.
fn repeat_reduce_method(args: &[PyObjectRef]) -> PyResult {
    // descr_reduce pickles to the exact `itertools.repeat` builtin type
    // (`space.gettypefor(W_Repeat)`), not the receiver's subclass type.
    let w_type =
        crate::typedef::gettypefor(&pyre_object::interp_itertools::REPEAT_TYPE).unwrap_or(PY_NULL);
    let w_obj = unsafe { pyre_object::interp_itertools::w_repeat_get_obj(args[0]) };
    let counting = unsafe { pyre_object::interp_itertools::w_repeat_get_counting(args[0]) };
    let count = unsafe { pyre_object::interp_itertools::w_repeat_get_count(args[0]) };
    let state = if counting {
        w_tuple_new(vec![w_obj, w_int_new(count)])
    } else {
        w_tuple_new(vec![w_obj])
    };
    Ok(w_tuple_new(vec![w_type, state]))
}

/// `cycle.__reduce__` — `interp_itertools.py W_Cycle.descr_reduce`:
/// `(type(self), (iterable,), (list(saved), index))`.  The saved buffer is
/// copied into a fresh list (`space.newlist(self.saved_w)`) so later
/// cycling cannot mutate the pickled state.
fn cycle_reduce_method(args: &[PyObjectRef]) -> PyResult {
    // Capture every field before any allocation (`w_list_new` /
    // `w_tuple_new` may collect): the saved elements go into a `Vec`
    // that `w_list_new` pins, and `w_iterable` / `index` are read up
    // front rather than across an allocation.
    let w_type = crate::typedef::r#type(args[0]).unwrap_or(PY_NULL);
    let it = unsafe { &*(args[0] as *const pyre_object::interp_itertools::W_Cycle) };
    let w_iterable = it.w_iterable;
    let index = it.index;
    let n = unsafe { pyre_object::w_list_len(it.saved) };
    let mut saved = Vec::with_capacity(n);
    for i in 0..n as i64 {
        saved.push(
            unsafe { pyre_object::w_list_getitem(it.saved, i) }
                .expect("cycle saved index in range"),
        );
    }
    let state = w_tuple_new(vec![w_list_new(saved), w_int_new(index)]);
    Ok(w_tuple_new(vec![
        w_type,
        w_tuple_new(vec![w_iterable]),
        state,
    ]))
}

/// `cycle.__setstate__` — `interp_itertools.py W_Cycle.descr_setstate`:
/// unpack `(saved, index)`, replace `saved_w` with a fresh list of the
/// unpacked elements, and restore `index`.  Reassigning the `saved`
/// pointer field requires the GC write barrier so an old→young edge is
/// recorded.
fn cycle_setstate_method(args: &[PyObjectRef]) -> PyResult {
    // `unpackiterable` iterates the pickled state and may collect; pin the
    // receiver and the state tuple so they (and, transitively, the saved
    // list reached through the tuple) survive each iteration.
    let _roots = pyre_object::gc_roots::push_roots();
    let w_self = args[0];
    let w_state = args.get(1).copied().unwrap_or(w_none());
    pyre_object::gc_roots::pin_root(w_self);
    pyre_object::gc_roots::pin_root(w_state);
    let state_w = unpackiterable(w_state, 2)?;
    let saved_w = unpackiterable(state_w[0], -1)?;
    let w_saved = w_list_new(saved_w);
    let index = int_w(state_w[1])?;
    let it = unsafe { &mut *(w_self as *mut pyre_object::interp_itertools::W_Cycle) };
    it.saved = w_saved;
    pyre_object::gc_hook::try_gc_write_barrier(w_self as *mut u8);
    it.index = index;
    Ok(w_none())
}

/// PyPy: GeneratorIterator.descr_send(w_arg)
fn generator_send_method(args: &[PyObjectRef]) -> PyResult {
    let gen_obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
    let value = args.get(1).copied().unwrap_or(w_none());
    generator_send_ex(gen_obj, value, None)
}

/// PyPy: GeneratorIterator.descr_throw(w_type, w_val=None, w_tb=None)
fn generator_throw_method(args: &[PyObjectRef]) -> PyResult {
    let gen_obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
    let w_type = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
    let w_val = args.get(2).copied().unwrap_or(pyre_object::PY_NULL);
    // w_tb (args[3]) ignored for now — traceback not yet supported

    let err = normalize_throw_args(w_type, w_val);
    generator_send_ex(gen_obj, w_none(), Some(err))
}

/// PyPy: GeneratorIterator.descr_close()
fn generator_close_method(args: &[PyObjectRef]) -> PyResult {
    let gen_obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
    unsafe {
        use pyre_object::generator::*;
        if w_generator_is_exhausted(gen_obj) {
            return Ok(w_none());
        }
        if !w_generator_is_started(gen_obj) {
            w_generator_set_exhausted(gen_obj);
            return Ok(w_none());
        }
    }
    let err = PyError::new(PyErrorKind::GeneratorExit, String::new());
    match generator_send_ex(gen_obj, w_none(), Some(err)) {
        Ok(_) => {
            // Generator yielded after GeneratorExit — RuntimeError
            Err(PyError::runtime_error("generator ignored GeneratorExit"))
        }
        Err(e) if e.kind == PyErrorKind::StopIteration || e.kind == PyErrorKind::GeneratorExit => {
            Ok(w_none())
        }
        Err(e) => Err(e),
    }
}

/// Normalize throw() arguments into a PyError.
///
/// PyPy: generator.py throw() → OperationError(w_type, w_val, tb) + normalize
///
/// Handles:
///   throw(TypeError)         — type → creates instance
///   throw(TypeError("msg"))  — instance → derives type
///   throw(TypeError, "msg")  — type + value → creates instance
fn normalize_throw_args(w_type: PyObjectRef, w_val: PyObjectRef) -> PyError {
    unsafe {
        // If w_type is an exception instance, use it directly
        if !w_type.is_null() && pyre_object::interp_exceptions::is_exception(w_type) {
            return PyError::from_exc_object(w_type);
        }

        // If w_type is a type (class), try to create exception from it
        if !w_type.is_null() && pyre_object::is_type(w_type) {
            let type_name = pyre_object::w_type_get_name(w_type);
            if let Some(kind) = pyre_object::interp_exceptions::exc_kind_from_name(type_name) {
                let msg = if w_val.is_null() || pyre_object::is_none(w_val) {
                    String::new()
                } else if pyre_object::is_str(w_val) {
                    pyre_object::w_str_get_value(w_val).to_string()
                } else {
                    String::new()
                };
                return PyError::new(PyError::kind_from_exc(kind), msg);
            }
        }

        // Fallback: TypeError
        PyError::type_error("exceptions must be classes or instances deriving from BaseException")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_setattr_getattr() {
        // PyPy raises AttributeError when setattr targets a non-hasdict
        // type. Use a hasdict instance: a W_ObjectObject of a fresh
        // user class created via type().
        let obj = make_user_instance();
        setattr_str(obj, "name", w_int_new(100)).unwrap();
        let result = getattr_str(obj, "name").unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 100) };
    }

    #[test]
    fn test_getattr_missing() {
        let obj = w_int_new(1);
        let err = getattr_str(obj, "missing").unwrap_err();
        assert!(matches!(err.kind, PyErrorKind::AttributeError));
    }

    #[test]
    fn test_setattr_overwrite() {
        let obj = make_user_instance();
        setattr_str(obj, "x", w_int_new(1)).unwrap();
        setattr_str(obj, "x", w_int_new(2)).unwrap();
        let result = getattr_str(obj, "x").unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 2) };
    }

    /// Helper for the setattr/getattr tests: build an instance of a fresh
    /// user class so the object has a live W_DictObject backing store
    /// (analogous to PyPy's `_getusercls` instances).
    fn make_user_instance() -> PyObjectRef {
        crate::typedef::init_typeobjects();
        use pyre_object::objectobject::w_instance_new;
        let cls = crate::typedef::make_builtin_type("TestUserClass", |_| {});
        unsafe { pyre_object::w_type_set_hasdict(cls, true) };
        w_instance_new(cls)
    }

    /// typeobject.py:293-301 — under the interpreter (`we_are_jitted()`
    /// is false in tests) the `version_tag()` gate and the
    /// `@elidable_promote` `_pure_version_tag` both read the live
    /// `_version_tag` field, matching the raw accessor.
    #[test]
    fn version_tag_gate_and_pure_read_the_field() {
        let t =
            pyre_object::w_type_new("VersionTagGate", pyre_object::PY_NULL, std::ptr::null_mut());
        unsafe {
            let raw = pyre_object::typeobject::w_type_get_version_tag(t);
            assert_ne!(raw, 0);
            assert_eq!(w_type_version_tag(t), raw);
            assert_eq!(_pure_version_tag(t), raw);
        }
    }

    /// typeobject.py:308/333 — once `uses_object_getattribute` /
    /// `uses_object_setattr` is set, the `*_if_not_from_object` fast path
    /// returns `None` (the object default) without an MRO walk;
    /// `mutated()` clears both flags (typeobject.py:275-276).
    #[test]
    fn if_not_from_object_respects_flag_and_mutated_reset() {
        // `make_builtin_type` reads the base type's layout, so the type system
        // must be bootstrapped first (matches `make_user_instance`); otherwise
        // the test aborts when run in isolation.
        crate::typedef::init_typeobjects();
        let t = crate::typedef::make_builtin_type("IfNotFromObject", |_| {});
        unsafe {
            pyre_object::typeobject::w_type_set_uses_object_getattribute(t, true);
            pyre_object::typeobject::w_type_set_uses_object_setattr(t, true);
            assert!(getattribute_if_not_from_object(t).is_none());
            assert!(setattr_if_not_from_object(t).is_none());
            mutated(t, None);
            assert!(!pyre_object::typeobject::w_type_get_uses_object_getattribute(t));
            assert!(!pyre_object::typeobject::w_type_get_uses_object_setattr(t));
        }
    }

    #[test]
    fn test_module_setattr_getattr() {
        let mut namespace = Box::new(crate::DictStorage::default());
        namespace.fix_ptr();
        let module =
            pyre_object::module::w_module_new("test_module", Box::into_raw(namespace) as *mut u8);

        setattr_str(module, "ps1", w_str_new("py> ")).unwrap();
        let result = getattr_str(module, "ps1").unwrap();
        unsafe { assert_eq!(w_str_get_value(result), "py> ") };
    }

    #[test]
    fn test_module_delattr() {
        let mut namespace = Box::new(crate::DictStorage::default());
        namespace.fix_ptr();
        let module =
            pyre_object::module::w_module_new("test_module", Box::into_raw(namespace) as *mut u8);

        setattr_str(module, "ps1", w_str_new("py> ")).unwrap();
        delattr_str(module, "ps1").unwrap();
        let err = getattr_str(module, "ps1").unwrap_err();
        assert!(matches!(err.kind, PyErrorKind::AttributeError));
    }

    /// `pypy/interpreter/module.py:77 Module.getdict()` parity invariant:
    /// every call to `dict_storage_to_dict(storage)` must return the
    /// **same** `W_DictObject` (single canonical) for a given storage.
    /// This is the foundation of `f.__globals__ is m.__dict__` and
    /// `globals() is __main__.__dict__` — pyre's split entries Vec /
    /// DictStorage no longer creates fresh snapshot wrappers per call.
    ///
    /// The first call lazy-allocates a W_DictObject and registers it as
    /// the storage's `mirror_target`; subsequent calls retrieve that
    /// same dict via `mirror_target` lookup.
    #[test]
    fn test_dict_storage_to_dict_returns_canonical_w_dict() {
        let mut namespace = Box::new(crate::DictStorage::default());
        namespace.fix_ptr();
        crate::dict_storage_store(&mut namespace, "alpha", w_int_new(7));
        let ns_ptr: *const crate::DictStorage = &*namespace;

        let first = super::dict_storage_to_dict(ns_ptr);
        let second = super::dict_storage_to_dict(ns_ptr);
        assert!(
            std::ptr::eq(first, second),
            "dict_storage_to_dict must return the canonical W_DictObject \
             (storage's mirror_target), not a fresh snapshot",
        );

        // Storage-side write after the canonical has been allocated
        // surfaces in the same W_DictObject via the back-mirror.
        crate::dict_storage_store(&mut namespace, "beta", w_int_new(11));
        unsafe {
            assert_eq!(
                w_int_get_value(pyre_object::w_dict_lookup(first, w_str_new("beta")).unwrap()),
                11,
                "post-canonicalization storage write must mirror into the canonical W_DictObject's entries Vec",
            );
        }
    }

    /// `pypy/interpreter/module.py:77 Module.getdict()` parity invariant
    /// for the new module-creation pattern (canonical W_DictObject reuse
    /// via `dict_storage_to_dict` + `w_module_new_aliasing_dict`):
    /// `module.w_dict` IS the storage's canonical W_DictObject, and
    /// `dict_storage_to_dict(module.dict_storage)` returns that same
    /// object on every call.
    #[test]
    fn test_module_w_dict_is_canonical_for_storage() {
        let mut namespace = Box::new(crate::DictStorage::default());
        namespace.fix_ptr();
        crate::dict_storage_store(&mut namespace, "x", w_int_new(42));
        let ns_ptr = Box::into_raw(namespace);

        // Pattern matches the production module-creation path
        // (executioncontext::get_builtin / importing::init_builtin_module
        //  / pyrex::run_source __main__).
        let canonical = super::dict_storage_to_dict(ns_ptr);
        let module = pyre_object::module::w_module_new_aliasing_dict(
            "test_canonical",
            ns_ptr as *mut u8,
            canonical,
        );

        // Module's w_dict identity equals the canonical lazy-paired
        // W_DictObject — `f.__globals__ is m.__dict__` invariant.
        let module_w_dict = unsafe { pyre_object::w_module_get_w_dict(module) };
        assert!(
            std::ptr::eq(module_w_dict, canonical),
            "Module.w_dict must alias the storage's canonical W_DictObject",
        );

        // Repeat lookup of the canonical (e.g. function.__globals__
        // path in production) returns the same object.
        let again = super::dict_storage_to_dict(ns_ptr);
        assert!(
            std::ptr::eq(again, canonical),
            "subsequent dict_storage_to_dict on the same storage must return the same W_DictObject",
        );

        // `module.__dict__["x"]` (resolved via the canonical W_DictObject)
        // sees the storage-pre-populated entry.
        unsafe {
            assert_eq!(
                w_int_get_value(pyre_object::w_dict_lookup(module_w_dict, w_str_new("x")).unwrap()),
                42,
                "canonical W_DictObject must surface storage-side entries that pre-date canonicalization",
            );
        }
    }

    #[test]
    fn test_py_contains_manual_list() {
        let list = w_list_new(vec![w_int_new(1), w_int_new(2), w_int_new(3)]);
        let needle = w_int_new(1);
        unsafe {
            assert!(
                is_list(list),
                "should be list, got type: {}",
                (*(*list).ob_type).name
            );
        }
        let result = super::contains(list, needle).expect("contains failed");
        assert!(result, "1 should be in [1, 2, 3]");
    }

    /// abstractinst.py:53-72 — `isinstance(5, 6)` must raise TypeError
    /// from `check_class()`, not silently return False from a naive
    /// `isinstance_w` walk. PyPy test: `test_builtin.py:605`.
    #[test]
    fn test_isinstance_non_class_arg2_raises_typeerror() {
        crate::typedef::init_typeobjects();
        let err = super::isinstance(w_int_new(5), w_int_new(6)).unwrap_err();
        assert!(matches!(err.kind, PyErrorKind::TypeError));
        assert!(err.message.contains("isinstance() arg 2"));
    }

    /// abstractinst.py:108-114 + 53-72 — when one tuple element is not a
    /// class the recursion must surface the TypeError from `check_class`.
    #[test]
    fn test_isinstance_tuple_with_non_class_raises_typeerror() {
        crate::typedef::init_typeobjects();
        let float_type = crate::typedef::r#type(w_float_new(0.0)).unwrap();
        let bad = w_tuple_new(vec![float_type, w_int_new(6)]);
        let err = super::isinstance(w_int_new(5), bad).unwrap_err();
        assert!(matches!(err.kind, PyErrorKind::TypeError));
    }

    /// abstractinst.py:150-169 — `issubclass(5, int)` must raise
    /// TypeError because the first argument is not a class.
    #[test]
    fn test_issubclass_non_class_arg1_raises_typeerror() {
        crate::typedef::init_typeobjects();
        let int_type = crate::typedef::r#type(w_int_new(0)).unwrap();
        let err = super::issubclass(w_int_new(5), int_type).unwrap_err();
        assert!(matches!(err.kind, PyErrorKind::TypeError));
        assert!(err.message.contains("issubclass() arg 1"));
    }

    /// abstractinst.py:150-169 — `issubclass(int, 6)` must raise
    /// TypeError because the second argument is not a class.
    #[test]
    fn test_issubclass_non_class_arg2_raises_typeerror() {
        crate::typedef::init_typeobjects();
        let int_type = crate::typedef::r#type(w_int_new(0)).unwrap();
        let err = super::issubclass(int_type, w_int_new(6)).unwrap_err();
        assert!(matches!(err.kind, PyErrorKind::TypeError));
        assert!(err.message.contains("issubclass() arg 2"));
    }

    /// abstractinst.py:127-147 — `p_abstract_issubclass_w` must walk
    /// `__bases__` for pseudo-classes (any object that exposes a tuple
    /// `__bases__` attribute), not just real type objects. We construct
    /// `outer` whose `__bases__` is `(inner,)` and `inner` whose
    /// `__bases__` is the empty tuple, then verify
    /// `issubclass(outer, inner)` returns True via the abstract walk.
    #[test]
    fn test_issubclass_pseudo_class_via_bases() {
        crate::typedef::init_typeobjects();
        let inner_type = crate::typedef::make_builtin_type("PseudoInner", |ns| {
            crate::dict_storage_store(ns, "__bases__", w_tuple_new(vec![]));
        });
        let inner = pyre_object::objectobject::w_instance_new(inner_type);
        let outer_type = crate::typedef::make_builtin_type("PseudoOuter", |_ns| {
            // closure capture is fine — make_builtin_type runs init eagerly.
        });
        // Stash __bases__ on outer's type dict pointing at the inner instance.
        crate::dict_storage_store(
            unsafe {
                &mut *(pyre_object::w_type_get_dict_ptr(outer_type) as *mut crate::DictStorage)
            },
            "__bases__",
            w_tuple_new(vec![inner]),
        );
        let outer = pyre_object::objectobject::w_instance_new(outer_type);
        let yes = super::issubclass(outer, inner).expect("issubclass should succeed");
        assert!(yes);
    }

    /// pypy/interpreter/baseobjspace.py:983 `unpackiterable` known-length:
    /// `[1, 2, 3]` with expected_length=3 yields the unpacked items.
    #[test]
    fn unpackiterable_known_length_match() {
        let lst = w_list_new(vec![w_int_new(1), w_int_new(2), w_int_new(3)]);
        let items = unpackiterable(lst, 3).expect("unpack should succeed");
        assert_eq!(items.len(), 3);
        unsafe {
            assert_eq!(w_int_get_value(items[0]), 1);
            assert_eq!(w_int_get_value(items[1]), 2);
            assert_eq!(w_int_get_value(items[2]), 3);
        }
    }

    /// pypy/interpreter/baseobjspace.py:1049-1052 — `not enough values
    /// to unpack` ValueError when iterator yields fewer items than
    /// expected.
    #[test]
    fn unpackiterable_too_few() {
        let lst = w_list_new(vec![w_int_new(1)]);
        let err = unpackiterable(lst, 3).expect_err("expected ValueError");
        assert_eq!(err.kind, crate::PyErrorKind::ValueError);
        assert!(err.message.contains("not enough values"));
    }

    /// pypy/interpreter/baseobjspace.py:1043-1046 — `too many values
    /// to unpack` ValueError when iterator yields more items than
    /// expected.
    #[test]
    fn unpackiterable_too_many() {
        let lst = w_list_new(vec![w_int_new(1), w_int_new(2), w_int_new(3), w_int_new(4)]);
        let err = unpackiterable(lst, 3).expect_err("expected ValueError");
        assert_eq!(err.kind, crate::PyErrorKind::ValueError);
        assert!(err.message.contains("too many values"));
    }

    /// pypy/interpreter/baseobjspace.py:983-994 — expected_length=-1
    /// accepts any length without validation.
    #[test]
    fn unpackiterable_unknown_length_accepts_any() {
        let lst = w_list_new(vec![w_int_new(10), w_int_new(20)]);
        let items = unpackiterable(lst, -1).expect("unpack should succeed");
        assert_eq!(items.len(), 2);
    }

    /// pypy/interpreter/baseobjspace.py:1110-1116 `fixedview` is a
    /// thin wrapper over `unpackiterable`; verify it dispatches.
    #[test]
    fn fixedview_delegates_to_unpackiterable() {
        let lst = w_list_new(vec![w_int_new(7), w_int_new(8)]);
        let items = fixedview(lst, 2).expect("fixedview should succeed");
        assert_eq!(items.len(), 2);
        unsafe {
            assert_eq!(w_int_get_value(items[0]), 7);
            assert_eq!(w_int_get_value(items[1]), 8);
        }
    }

    /// pypy/objspace/descroperation.py:319-326 `is_iterable`:
    /// list / tuple / dict / str return true via builtin shortcuts.
    #[test]
    fn is_iterable_true_for_builtin_types() {
        assert!(is_iterable(w_list_new(vec![])));
        assert!(is_iterable(pyre_object::w_tuple_new(vec![])));
        assert!(is_iterable(pyre_object::w_str_new("hello")));
    }

    /// pypy/objspace/descroperation.py:319-326 `is_iterable`:
    /// scalar types (int) without `__iter__` / `__getitem__` return false.
    #[test]
    fn is_iterable_false_for_scalar() {
        assert!(!is_iterable(w_int_new(42)));
        assert!(!is_iterable(w_none()));
    }

    /// pypy/objspace/std/objspace.py:609-617 + dictmultiobject.py:307
    /// — exact `dict` with all string keys takes the
    /// strategy-specific fast path and returns parallel
    /// `(Some(keys), Some(values))`.  An empty exact dict goes
    /// through the same path and returns `(Some([]), Some([]))`.
    #[test]
    fn view_as_kwargs_empty_dict_returns_some_empty() {
        let d = pyre_object::dictmultiobject::w_dict_new();
        let (names, values) = view_as_kwargs(d);
        assert_eq!(names.as_ref().map(|v| v.len()), Some(0));
        assert_eq!(values.as_ref().map(|v| v.len()), Some(0));
    }

    /// pypy/objspace/std/dictmultiobject.py:1325 — kwargs strategy
    /// only succeeds when every key is a unicode string; the base
    /// `(None, None)` is returned for non-string keys (e.g. int).
    #[test]
    fn view_as_kwargs_int_key_returns_none() {
        unsafe {
            let d = pyre_object::dictmultiobject::w_dict_new();
            pyre_object::dictmultiobject::w_dict_store(d, w_int_new(1), w_int_new(2));
            let (names, values) = view_as_kwargs(d);
            assert!(names.is_none());
            assert!(values.is_none());
        }
    }

    /// pypy/objspace/std/objspace.py:615 `isinstance(w_dict,
    /// W_DictObject)` — non-dict (e.g. `int`) returns the base
    /// `(None, None)`.
    #[test]
    fn view_as_kwargs_non_dict_returns_none() {
        let (names, values) = view_as_kwargs(w_int_new(42));
        assert!(names.is_none());
        assert!(values.is_none());
    }

    /// pypy/interpreter/baseobjspace.py:2137-2140 `object_functionstr`
    /// fallback path: scalars without `__qualname__` go through
    /// `space.str(w_function)`, which dispatches to the type's
    /// `__str__` slot via `lookup`.  Pyre's `lookup` walks the
    /// W_TypeObject MRO, which is only populated after
    /// `init_typeobjects()` runs.
    #[test]
    fn object_functionstr_scalar_fallback() {
        crate::typedef::init_typeobjects();
        let s = object_functionstr(w_int_new(42)).expect("scalar fallback never propagates async");
        assert_eq!(s, "42");
    }
}

/// `in` operator: check if `needle` is in `haystack`.
/// PyPy: space.contains_w(haystack, needle)
pub fn contains(haystack: PyObjectRef, needle: PyObjectRef) -> Result<bool, PyError> {
    use pyre_object::*;
    // A builtin container subclass overriding `__contains__` dispatches the
    // override; exact instances and non-overriding subclasses fall through to
    // the by-layout membership slot, which gives the inherited builtin scan
    // without re-entering override dispatch.  Override targets are never a
    // dict-proxy / dict-view / range, so this precedes the unwrap below.
    unsafe {
        if is_list(haystack)
            || is_tuple(haystack)
            || is_str(haystack)
            || is_dict(haystack)
            || pyre_object::is_set_or_frozenset(haystack)
        {
            if let Some((method, w_type)) = subclass_special_override(haystack, "__contains__") {
                let result = get_and_call_function(method, haystack, w_type, &[needle])?;
                return Ok(is_true(result)?);
            }
        }
    }
    contains_slot(haystack, needle)
}

/// The builtin `__contains__` slot body: membership dispatch by concrete
/// layout.  Reached from the operator [`contains`] for exact instances and
/// non-overriding subclasses, and bound directly as the `list`/`str`/`tuple`
/// `__contains__` slot so a subclass override's `super().__contains__`
/// resolves to the inherited builtin scan instead of re-entering override
/// dispatch (which would recurse).
pub(crate) fn contains_slot(haystack: PyObjectRef, needle: PyObjectRef) -> Result<bool, PyError> {
    use pyre_object::*;
    // `pypy/objspace/std/dictproxyobject.py:38 descr_contains` →
    // `space.contains(self.w_mapping, w_key)`.
    let haystack = unsafe {
        if pyre_object::is_dict_proxy(haystack) {
            pyre_object::w_dict_proxy_get_mapping(haystack)
        } else {
            haystack
        }
    };
    // `pypy/objspace/std/dictmultiobject.py`
    // `W_DictViewKeysObject.descr_contains` →
    // `space.contains(self.w_dict, w_key)`.
    // `W_DictViewItemsObject.descr_contains` matches a (k, v)
    // tuple via dict lookup + value equality.  `W_DictViewValuesObject`
    // has no `__contains__` slot in PyPy — pyre delegates the
    // fall-through to the standard `iter`-based scan further down so
    // `v in d.values()` still works (as in PyPy where the missing
    // slot triggers the iter fallback).
    unsafe {
        if pyre_object::dictmultiobject::is_dict_view(haystack) {
            let kind = pyre_object::dictmultiobject::w_dict_view_get_kind(haystack);
            let dict = pyre_object::dictmultiobject::w_dict_view_get_dict(haystack);
            if dict.is_null() {
                return Ok(false);
            }
            match kind {
                pyre_object::dictmultiobject::DictViewKind::Keys => {
                    return match unsafe {
                        pyre_object::dictmultiobject::w_dict_lookup_checked(dict, needle)
                    } {
                        Ok(v) => Ok(v.is_some()),
                        Err(_) => Err(take_pending_hash_error()),
                    };
                }
                pyre_object::dictmultiobject::DictViewKind::Items => {
                    if !is_tuple(needle) || w_tuple_len(needle) != 2 {
                        return Ok(false);
                    }
                    let k = match w_tuple_getitem(needle, 0) {
                        Some(k) => k,
                        None => return Ok(false),
                    };
                    let want = match w_tuple_getitem(needle, 1) {
                        Some(v) => v,
                        None => return Ok(false),
                    };
                    return match unsafe {
                        pyre_object::dictmultiobject::w_dict_lookup_checked(dict, k)
                    } {
                        Ok(Some(have)) => eq_w(have, want),
                        Ok(None) => Ok(false),
                        Err(_) => Err(take_pending_hash_error()),
                    };
                }
                pyre_object::dictmultiobject::DictViewKind::Values => {
                    // values view: PyPy uses iter-based scan.
                    for (_, v) in pyre_object::w_dict_items(dict) {
                        if eq_w(v, needle)? {
                            return Ok(true);
                        }
                    }
                    return Ok(false);
                }
            }
        }
    }
    // `functional.py W_Range.descr_contains` — O(1) membership for
    // an int/long needle; any other type falls back to an elementwise scan.
    unsafe {
        if pyre_object::is_w_range(haystack) {
            if is_int(needle) || is_long(needle) {
                let item = pyre_object::range_obj_to_bigint(needle);
                return Ok(pyre_object::w_range_contains_bigint(haystack, &item));
            }
            // `space.sequence_contains` — elementwise scan.
            let it = iter(haystack)?;
            loop {
                match next(it) {
                    Ok(item) => {
                        if is_true(compare(item, needle, CompareOp::Eq)?)? {
                            return Ok(true);
                        }
                    }
                    Err(e) if e.kind == PyErrorKind::StopIteration => break,
                    Err(e) => return Err(e),
                }
            }
            return Ok(false);
        }
    }
    unsafe {
        if is_list(haystack) {
            let len = w_list_len(haystack);
            for i in 0..len {
                if let Some(item) = w_list_getitem(haystack, i as i64) {
                    if eq_w(item, needle)? {
                        return Ok(true);
                    }
                }
            }
            return Ok(false);
        }
        if is_tuple(haystack) {
            let len = w_tuple_len(haystack);
            for i in 0..len {
                if let Some(item) = w_tuple_getitem(haystack, i as i64) {
                    if eq_w(item, needle)? {
                        return Ok(true);
                    }
                }
            }
            return Ok(false);
        }
        if is_str(haystack) && is_str(needle) {
            // Substring test over the WTF-8 bytes: the encoding is
            // self-synchronizing, so a byte-level match coincides with a
            // codepoint-level match and lone surrogates compare correctly.
            let h = pyre_object::w_str_get_wtf8(haystack).as_bytes();
            let n = pyre_object::w_str_get_wtf8(needle).as_bytes();
            return Ok(n.is_empty() || h.windows(n.len()).any(|w| w == n));
        }
        // bytes / bytearray: stringmethods.py descr_contains via
        // `_op_val(allow_char=True)` — an int needle is a single byte value
        // (range-checked), a bytes-like needle is matched as a substring (an
        // empty needle is always present), and any other type is a TypeError.
        // `_op_val` falls back to `buffer_w(BUF_SIMPLE)`, so any buffer-protocol
        // object (e.g. a memoryview) is also accepted as the needle.
        if pyre_object::bytesobject::is_bytes_like(haystack) {
            let hay = pyre_object::bytesobject::bytes_like_data(haystack);
            if is_int(needle) || is_long(needle) {
                // `_single_char`: `int_w` then `0 <= c < 256`; a bignum is
                // necessarily out of range (its `int_w` overflows upstream).
                let v = if is_int(needle) {
                    pyre_object::w_int_get_value(needle)
                } else {
                    -1
                };
                if !(0..=255).contains(&v) {
                    return Err(PyError::value_error("byte must be in range(0, 256)"));
                }
                return Ok(hay.contains(&(v as u8)));
            }
            if let Some(src) = crate::typedef::buffer_as_bytes_like(needle)? {
                let sub = pyre_object::bytesobject::bytes_like_data(src);
                return Ok(sub.is_empty() || hay.windows(sub.len()).any(|w| w == sub));
            }
            let tname = match crate::typedef::r#type(needle) {
                Some(tp) => pyre_object::w_type_get_name(tp).to_string(),
                None => "object".to_string(),
            };
            return Err(PyError::type_error(format!(
                "a bytes-like object is required, not '{tname}'"
            )));
        }
        // dict: key containment (dictmultiobject.py __contains__)
        if is_dict(haystack) {
            return match pyre_object::dictmultiobject::w_dict_lookup_checked(haystack, needle) {
                Ok(v) => Ok(v.is_some()),
                Err(_) => Err(take_pending_hash_error()),
            };
        }
        // set / frozenset (setobject.py W_BaseSetObject.descr_contains)
        if pyre_object::is_set_or_frozenset(haystack) {
            return Ok(pyre_object::w_set_contains(haystack, needle));
        }
    }
    // Instance __contains__ — PyPy: descroperation.py contains_w
    unsafe {
        if is_instance(haystack) {
            let w_type = w_instance_get_type(haystack);
            if let Some(method) = lookup_in_type_where(w_type, "__contains__") {
                let result = crate::builtins::call_and_check(method, &[haystack, needle])?;
                return Ok(is_true(result)?);
            }
            // Also check per-instance attributes
            if let Ok(method) = getattr_str(haystack, "__contains__") {
                let result = crate::builtins::call_and_check(method, &[haystack, needle])?;
                return Ok(is_true(result)?);
            }
        }
    }
    // A `__contains__` resolved on the receiver's dynamic type applies
    // before the getitem scan: covers a metaclass `__contains__` when the
    // haystack is a class (`x in Color` → `type(Color).__contains__(Color,
    // x)`; `type` defines none, so an ordinary class falls through) and a
    // builtin-leaf subclass instance (`x in flag` where the flag's ob_type
    // is the int storage type but its w_class carries `__contains__`).  The
    // is_instance receivers are already handled above.
    unsafe {
        if let Some(w_type) = crate::typedef::r#type(haystack) {
            if let Some(method) = lookup_in_type_where(w_type, "__contains__") {
                let result = crate::builtins::call_and_check(method, &[haystack, needle])?;
                return Ok(is_true(result)?);
            }
        }
    }
    // Fallback: `space.sequence_contains` — scan via getitem(obj, i) for
    // i = 0, 1, ….  An `IndexError` ends the scan (not found); any other
    // error (e.g. a released/non-contiguous memoryview) propagates, matching
    // `PySequence_Contains`.
    let mut i = 0i64;
    loop {
        match getitem(haystack, pyre_object::w_int_new(i)) {
            Ok(item) => {
                if eq_w(item, needle)? {
                    return Ok(true);
                }
                i += 1;
            }
            Err(e) if e.kind == PyErrorKind::IndexError => return Ok(false),
            Err(e) => return Err(e),
        }
    }
}

/// `pypy/interpreter/baseobjspace.py:840-845 W_ObjectSpace.hash_w` —
/// returns the `__hash__` digest as `i64`.  Routes through pyre's
/// existing `builtins::hash_value`, which already covers
/// int/long/bool/float/str/tuple/frozenset/None plus user
/// `__hash__` dispatch through `lookup_in_type`.  Returns `0` for
/// non-hashable types (PyPy raises; pyre surfaces the same
/// hash-not-available signal by returning `0` and letting the dict
/// dispatcher fall through).
pub fn hash_w(obj: PyObjectRef) -> i64 {
    crate::builtins::hash_value(obj)
}

/// `pypy/objspace/descroperation.py:553-580 hash_w` — strict variant
/// that raises `TypeError: unhashable type: '<typename>'` instead of
/// silently returning a sentinel hash.  Built-in mutable containers
/// (dict / list / set / bytearray / dict view) are explicit
/// unhashables per `dictmultiobject.py:1431` + `listobject.py` +
/// `setobject.py`; everything else routes through `hash_value`'s
/// hashable type ladder.  Mirrors the entry-point gate already in
/// `builtins::builtin_hash` so callers that need to surface PyPy's
/// `TypeError` directly (EmptyDictStrategy `getitem` /
/// ObjectDictStrategy lookups per `dictmultiobject.py:738-743`) can
/// reuse the same dispatch without duplicating the type ladder.
pub fn hash_w_strict(obj: PyObjectRef) -> Result<i64, PyError> {
    if obj.is_null() {
        return Err(PyError::type_error("hash() argument is null"));
    }
    unsafe {
        let kind = if pyre_object::is_dict(obj) {
            Some("dict")
        } else if pyre_object::is_list(obj) {
            Some("list")
        } else if pyre_object::is_set(obj) {
            Some("set")
        } else if pyre_object::is_bytearray(obj) {
            Some("bytearray")
        } else if pyre_object::dictmultiobject::is_dict_view(obj) {
            Some("dict view")
        } else if pyre_object::sliceobject::is_slice(obj) {
            Some("slice")
        } else {
            None
        };
        if let Some(name) = kind {
            return Err(PyError::type_error(format!("unhashable type: '{}'", name)));
        }
        // A released or writable memoryview is unhashable; route through the
        // fallible hasher so it raises the proper ValueError instead of an
        // infallible identity hash (`memoryobject.py descr_hash`).
        if pyre_object::memoryview::is_w_memoryview(obj) {
            return crate::builtins::try_hash_value(obj);
        }
    }
    Ok(crate::builtins::hash_value(obj))
}

/// Compare two objects for equality (returns bool, not PyObjectRef).
/// baseobjspace.py:823-825 `eq_w`:
///   `self.is_w(w_obj1, w_obj2) or self.is_true(self.eq(w_obj1, w_obj2))`.
/// A raising `__eq__` or a raising `__bool__` on its result propagates.
pub fn eq_w(a: PyObjectRef, b: PyObjectRef) -> Result<bool, PyError> {
    if a == b {
        return Ok(true);
    }
    unsafe {
        use pyre_object::*;
        // The by-value fast paths assume exact builtin operands; a subclass
        // overriding `__eq__` must dispatch through `compare` instead.
        if is_exact_builtin_instance(a) && is_exact_builtin_instance(b) {
            if (is_int(a) || is_bool(a)) && (is_int(b) || is_bool(b)) {
                let av = if is_bool(a) {
                    w_bool_get_value(a) as i64
                } else {
                    w_int_get_value(a)
                };
                let bv = if is_bool(b) {
                    w_bool_get_value(b) as i64
                } else {
                    w_int_get_value(b)
                };
                return Ok(av == bv);
            }
            if is_str(a) && is_str(b) {
                // Compare WTF-8 bytes so lone-surrogate strings compare by
                // content instead of panicking in `w_str_get_value`.
                return Ok(pyre_object::w_str_get_wtf8(a).as_bytes()
                    == pyre_object::w_str_get_wtf8(b).as_bytes());
            }
        }
    }
    Ok(is_true(compare(a, b, CompareOp::Eq)?)?)
}

/// `baseobjspace.py:933 ObjSpace._side_effects_ok`.
///
/// Reverse debugging is not ported (`reverse_debugging` is set from
/// `config.translation.reverse_debugger` at `baseobjspace.py:441`), so the
/// `if self.reverse_debugging: return self._revdb_standard_code()` branch is
/// unreachable and this always returns `True`, matching the non-revdb path.
/// The JIT does not trace this cache write because the cache lookup lives in
/// the `@jit.dont_look_inside` `find_map_attr_cache` and the JIT calls
/// `compute_find_map_attr` directly (`mapdict.py:100-103`).
pub fn side_effects_ok() -> bool {
    true
}

/// Delete item: `del obj[index]`
///
/// PyPy: descroperation.py delitem → dispatches to type-specific __delitem__.
pub fn delitem(obj: PyObjectRef, index: PyObjectRef) -> Result<(), PyError> {
    use pyre_object::*;
    unsafe {
        // `pypy/objspace/std/dictproxyobject.py` exposes no
        // `__delitem__`, so `space.delitem` on a mappingproxy raises
        // `TypeError: 'mappingproxy' object does not support item
        // deletion`.
        if pyre_object::is_dict_proxy(obj) {
            return Err(PyError::type_error(
                "'mappingproxy' object does not support item deletion",
            ));
        }
        // A builtin sequence subclass overriding `__delitem__` dispatches the
        // override; exact instances and non-overriding subclasses fall through
        // to the by-layout deletion slot below.
        if is_list(obj) || pyre_object::bytearrayobject::is_bytearray(obj) {
            if let Some((method, w_type)) = subclass_special_override(obj, "__delitem__") {
                get_and_call_function(method, obj, w_type, &[index])?;
                return Ok(());
            }
        }
    }
    delitem_slot(obj, index)
}

/// The builtin `__delitem__` slot body: item-deletion dispatch by concrete
/// layout.  Reached from the operator [`delitem`] for exact instances and
/// non-overriding subclasses, and bound directly as the `list` `__delitem__`
/// slot so a subclass override's `super().__delitem__` resolves to the
/// inherited builtin deletion instead of re-entering override dispatch
/// (which would recurse).
pub(crate) fn delitem_slot(obj: PyObjectRef, index: PyObjectRef) -> Result<(), PyError> {
    use pyre_object::*;
    unsafe {
        if is_list(obj) {
            if is_int(index) {
                let i = w_int_get_value(index);
                let len = w_list_len(obj) as i64;
                let idx = if i < 0 { len + i } else { i };
                if idx >= 0 && idx < len {
                    w_list_pop(obj, idx);
                    return Ok(());
                }
                return Err(PyError::type_error("list index out of range"));
            }
            if is_slice(index) {
                let len = w_list_len(obj) as i64;
                let (start, stop, step) = normalize_slice(index, len)?;
                if step == 1 {
                    w_list_delslice(obj, start.max(0) as usize, stop.max(start) as usize);
                    return Ok(());
                }
                // Extended-slice delete: gather the selected indices, then
                // pop them in descending order so earlier removals do not
                // shift the positions of later targets.
                let mut indices: Vec<i64> = Vec::new();
                let mut i = start;
                if step > 0 {
                    while i < stop {
                        indices.push(i);
                        i += step;
                    }
                } else {
                    while i > stop {
                        indices.push(i);
                        i += step;
                    }
                }
                indices.sort_unstable_by(|a, b| b.cmp(a));
                for idx in indices {
                    if idx >= 0 && idx < w_list_len(obj) as i64 {
                        w_list_pop(obj, idx);
                    }
                }
                return Ok(());
            }
        }
        if is_dict(obj) {
            return dict_delitem(obj, index);
        }
        // `bytearrayobject.py` ass_subscript with a NULL value deletes like a
        // list: a single index removes one byte, a slice removes its selected
        // bytes (contiguous via drain, extended-step descending).  `descr_delitem`
        // runs `_unpack_slice` — the slice's `__index__` is evaluated before the
        // length is read, so a mutation during index evaluation is reflected in
        // the bounds.
        if pyre_object::bytearrayobject::is_bytearray(obj) {
            if is_slice(index) {
                let (rs, rp, st) = crate::sliceobject::slice_unpack(
                    w_slice_get_start(index),
                    w_slice_get_stop(index),
                    w_slice_get_step(index),
                )?;
                let len = pyre_object::bytearrayobject::w_bytearray_len(obj) as i64;
                let (start, stop, step, _) =
                    crate::sliceobject::slice_adjust_indices(rs, rp, st, len);
                let vec = pyre_object::bytearrayobject::w_bytearray_vec_mut(obj);
                if step == 1 {
                    let s = start.max(0) as usize;
                    let e = stop.max(start).min(vec.len() as i64) as usize;
                    vec.drain(s..e);
                    return Ok(());
                }
                let mut indices: Vec<i64> = Vec::new();
                let mut i = start;
                if step > 0 {
                    while i < stop {
                        indices.push(i);
                        i += step;
                    }
                } else {
                    while i > stop {
                        indices.push(i);
                        i += step;
                    }
                }
                indices.sort_unstable_by(|a, b| b.cmp(a));
                for idx in indices {
                    if idx >= 0 && idx < vec.len() as i64 {
                        vec.remove(idx as usize);
                    }
                }
                return Ok(());
            }
            let i = bytearray_index(index)?;
            let len = pyre_object::bytearrayobject::w_bytearray_len(obj) as i64;
            let idx = if i < 0 { len + i } else { i };
            if idx >= 0 && idx < len {
                pyre_object::bytearrayobject::w_bytearray_vec_mut(obj).remove(idx as usize);
                return Ok(());
            }
            return Err(PyError::new(
                PyErrorKind::IndexError,
                "bytearray index out of range",
            ));
        }
        // memoryview never supports deletion; `memoryview_delitem` reports the
        // released / read-only / "cannot delete memory" error in order.
        if pyre_object::memoryview::is_w_memoryview(obj) {
            crate::builtins::memoryview_delitem(&[obj, index])?;
            return Ok(());
        }
    }
    // Instance __delitem__ — PyPy: descroperation.py delitem.  Errors from
    // user `__delitem__` propagate (PyPy `space.delitem` raises directly);
    // pyre's `call_function` stashes errors as PY_NULL so use
    // `call_and_check` to recover them.
    unsafe {
        if pyre_object::is_instance(obj) {
            if let Some(method) =
                lookup_in_type_where(pyre_object::w_instance_get_type(obj), "__delitem__")
            {
                crate::builtins::call_and_check(method, &[obj, index])?;
                return Ok(());
            }
        }
    }
    Err(PyError::type_error("object does not support item deletion"))
}

/// Delete item from dict by key.  `pypy/objspace/std/dictmultiobject.py:177
/// W_DictMultiObject.descr_delitem` routes `self.delitem(w_key)` through
/// the strategy slot, so both module and regular dicts get typed-storage
/// dispatch (IntDictStrategy / BytesDictStrategy / KwargsDictStrategy
/// etc. each own their layout — the previous raw
/// `Vec<(PyObjectRef, PyObjectRef)>` cast assumed ObjectDictStrategy).
/// `ObjectDictStrategy::delitem` + `ModuleDictStrategy::delitem` both
/// honour the W_DictObject `dict_storage_proxy` back-mirror via
/// `w_dict_delitem_object_strategy` / `w_module_dict_delitem_inner`.
fn dict_delitem(obj: PyObjectRef, key: PyObjectRef) -> Result<(), PyError> {
    unsafe {
        match pyre_object::dictmultiobject::w_dict_delitem_checked(obj, key) {
            Ok(true) => Ok(()),
            Ok(false) => Err(PyError::key_error_with_key(key)),
            Err(_) => Err(take_pending_hash_error()),
        }
    }
}

// py_str and py_repr are defined in display.rs (with __str__/__repr__ dispatch).
// Re-exported via crate::display::*.
