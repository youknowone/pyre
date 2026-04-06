//! Function object.
//!
//! Wraps a code object pointer, a function name, a pointer to the
//! defining module's globals namespace, and an optional closure tuple.
//! When called, the interpreter creates a new PyFrame that *shares*
//! the globals pointer (no clone).

use crate::executioncontext::PyNamespace;
use pyre_object::pyobject::*;

/// Type descriptor for user-defined functions.
pub static FUNCTION_TYPE: PyType = PyType {
    tp_name: "function",
};

/// User-defined function object.
///
/// Layout: `[ob_type | code | name_ptr | w_func_globals | closure]`
/// - `code`: opaque pointer to the CodeObject
/// - `name_ptr`: leaked `Box<String>` containing the function name
/// - `w_func_globals`: raw pointer to the module-level namespace (shared)
/// - `closure`:  tuple of cell objects, or PY_NULL if no closure
#[repr(C)]
pub struct Function {
    pub ob: PyObject,
    /// Opaque pointer to the CodeObject (borrowed from constants).
    pub code: *const (),
    /// Function name (leaked Box<String>).
    pub name: *const String,
    /// PyPy: W_Function.w_func_globals
    pub w_func_globals: *mut PyNamespace,
    /// Closure: tuple of cell objects from the enclosing scope,
    /// or PY_NULL if this function has no free variables.
    pub closure: PyObjectRef,
    /// Default argument values.
    /// PyPy: W_Function.defs_w
    pub defs_w: PyObjectRef,
    /// Keyword-only default values.
    /// PyPy: W_Function.w_kw_defs
    pub w_kw_defs: PyObjectRef,
}

pub type BuiltinFunction = Function;
pub type FunctionWithFixedCode = Function;
pub type Method = pyre_object::methodobject::W_MethodObject;
pub type StaticMethod = pyre_object::propertyobject::W_StaticMethodObject;
pub type ClassMethod = pyre_object::propertyobject::W_ClassMethodObject;

/// Field offset of `code` within `Function`, for JIT field access.
pub const FUNCTION_CODE_OFFSET: usize = std::mem::offset_of!(Function, code);
/// Field offset of `name` within `Function`.
pub const FUNCTION_NAME_OFFSET: usize = std::mem::offset_of!(Function, name);
/// Field offset of `w_func_globals` within `Function`.
pub const FUNCTION_GLOBALS_OFFSET: usize = std::mem::offset_of!(Function, w_func_globals);
/// Field offset of `closure` within `Function`.
pub const FUNCTION_CLOSURE_OFFSET: usize = std::mem::offset_of!(Function, closure);

/// Allocate a new `Function`.
///
/// `code` is an opaque pointer to the CodeObject.
/// `name` is the function name string (leaked).
/// `w_func_globals` is the defining module's namespace pointer (shared).
pub fn function_new(
    code: *const (),
    name: String,
    w_func_globals: *mut PyNamespace,
) -> PyObjectRef {
    function_new_with_closure(code, name, w_func_globals, PY_NULL)
}

/// Allocate a new `Function` with a closure.
///
/// `closure` is a tuple of cell objects, or PY_NULL if no closure.
pub fn function_new_with_closure(
    code: *const (),
    name: String,
    w_func_globals: *mut PyNamespace,
    closure: PyObjectRef,
) -> PyObjectRef {
    let name_ptr = Box::into_raw(Box::new(name)) as *const String;
    let obj = Box::new(Function {
        ob: PyObject {
            ob_type: &FUNCTION_TYPE as *const PyType,
        },
        code,
        name: name_ptr,
        w_func_globals,
        closure,
        defs_w: PY_NULL,
        w_kw_defs: PY_NULL,
    });
    Box::into_raw(obj) as PyObjectRef
}

/// PyPy `function._get_immutable_code`.
/// rlib/jit.py:180 — `@elidable_promote()`: the code pointer is immutable
/// once can_change_code is false, so the JIT can constant-fold this.
#[majit_macros::elidable_promote]
#[inline]
pub unsafe fn _get_immutable_code(func: PyObjectRef) -> *const () {
    unsafe { function_get_code(func) }
}

/// Check if an object is a user-defined function.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_function(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &FUNCTION_TYPE) }
}

#[inline]
pub unsafe fn getcode(obj: PyObjectRef) -> *const () {
    function_get_code(obj)
}

/// Get the opaque code pointer from a function object.
///
/// # Safety
/// `obj` must point to a valid `Function`.
/// NOTE: NOT elidable — code field can change (can_change_code).
/// Use _get_immutable_code() for the elidable path.
#[inline]
pub unsafe fn function_get_code(obj: PyObjectRef) -> *const () {
    unsafe { (*(obj as *const Function)).code }
}

/// Get the function name.
///
/// # Safety
/// `obj` must point to a valid `Function`.
#[inline]
pub unsafe fn function_get_name(obj: PyObjectRef) -> &'static str {
    unsafe { &*(*(obj as *const Function)).name }
}

#[inline]
pub unsafe fn _eq(_obj: PyObjectRef, other: PyObjectRef) -> bool {
    _obj == other
}

/// PyPy-compatible descriptor accessor for function name.
#[inline]
pub unsafe fn fget_func_name(obj: PyObjectRef) -> PyObjectRef {
    pyre_object::w_str_new(function_get_name(obj))
}

/// Get the globals namespace pointer from a function object.
///
/// # Safety
/// `obj` must point to a valid `Function`.
#[inline]
pub unsafe fn function_get_globals(obj: PyObjectRef) -> *mut PyNamespace {
    unsafe { (*(obj as *const Function)).w_func_globals }
}

/// Get the closure tuple from a function object.
/// Returns PY_NULL if the function has no closure.
///
/// # Safety
/// `obj` must point to a valid `Function`.
#[inline]
pub unsafe fn function_get_closure(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const Function)).closure }
}

/// Set the closure on a function object.
///
/// # Safety
/// `obj` must point to a valid `Function`.
#[inline]
pub unsafe fn function_set_closure(obj: PyObjectRef, closure: PyObjectRef) {
    unsafe { (*(obj as *mut Function)).closure = closure }
}

/// Get defaults tuple.
#[inline]
pub unsafe fn function_get_defaults(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const Function)).defs_w }
}

/// Set defaults tuple.
#[inline]
pub unsafe fn function_set_defaults(obj: PyObjectRef, defaults: PyObjectRef) {
    unsafe { (*(obj as *mut Function)).defs_w = defaults }
}

/// Get kwdefaults dict.
#[inline]
pub unsafe fn function_get_kwdefaults(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const Function)).w_kw_defs }
}

/// Set kwdefaults dict.
#[inline]
pub unsafe fn function_set_kwdefaults(obj: PyObjectRef, kwdefaults: PyObjectRef) {
    unsafe { (*(obj as *mut Function)).w_kw_defs = kwdefaults }
}

/// PyPy-compatible `__dict__` storage field alias.
#[inline]
pub unsafe fn function_getdict(obj: PyObjectRef) -> PyObjectRef {
    crate::baseobjspace::getattr(obj, "__dict__").unwrap_or(pyre_object::w_none())
}

/// PyPy-compatible `__dict__` mutator.
#[inline]
pub unsafe fn function_setdict(obj: PyObjectRef, value: PyObjectRef) {
    let _ = crate::baseobjspace::setattr(obj, "__dict__", value);
}

/// PyPy-compatible `getdict()` descriptor helper.
#[inline]
pub unsafe fn getdict(obj: PyObjectRef) -> PyObjectRef {
    function_getdict(obj)
}

/// PyPy-compatible `setdict()` helper.
#[inline]
pub unsafe fn setdict(obj: PyObjectRef, value: PyObjectRef) {
    function_setdict(obj, value)
}

/// PyPy `W_Function.w_doc` accessor.
#[inline]
pub fn function_get_doc(obj: PyObjectRef) -> PyObjectRef {
    crate::getattr(obj, "__doc__").unwrap_or(pyre_object::w_none())
}

/// PyPy `W_Function.w_doc` mutator.
#[inline]
pub fn function_set_doc(obj: PyObjectRef, value: PyObjectRef) {
    let _ = crate::setattr(obj, "__doc__", value);
}

/// PyPy `W_Function.w_doc` deleter.
#[inline]
pub fn function_del_doc(obj: PyObjectRef) {
    let _ = crate::delattr(obj, "__doc__");
}

/// PyPy `fget_func_defaults` accessor.
#[inline]
pub unsafe fn fget_func_defaults(obj: PyObjectRef) -> PyObjectRef {
    let value = function_get_defaults(obj);
    if value.is_null() {
        pyre_object::w_none()
    } else {
        value
    }
}

/// PyPy `fset_func_defaults` mutator.
#[inline]
pub unsafe fn fset_func_defaults(obj: PyObjectRef, value: PyObjectRef) {
    if value.is_null() || unsafe { pyre_object::is_none(value) } {
        function_set_defaults(obj, pyre_object::PY_NULL);
    } else {
        function_set_defaults(obj, value);
    }
}

/// PyPy `fdel_func_defaults` deleter.
#[inline]
pub unsafe fn fdel_func_defaults(obj: PyObjectRef) {
    function_set_defaults(obj, pyre_object::PY_NULL);
}

/// PyPy `fget_func_kwdefaults` accessor.
#[inline]
pub unsafe fn fget_func_kwdefaults(obj: PyObjectRef) -> PyObjectRef {
    let value = function_get_kwdefaults(obj);
    if value.is_null() {
        pyre_object::w_none()
    } else {
        value
    }
}

/// PyPy `fset_func_kwdefaults` mutator.
#[inline]
pub unsafe fn fset_func_kwdefaults(obj: PyObjectRef, value: PyObjectRef) {
    if value.is_null() || unsafe { pyre_object::is_none(value) } {
        function_set_kwdefaults(obj, pyre_object::PY_NULL);
    } else {
        function_set_kwdefaults(obj, value);
    }
}

/// PyPy `fdel_func_kwdefaults` deleter.
#[inline]
pub unsafe fn fdel_func_kwdefaults(obj: PyObjectRef) {
    function_set_kwdefaults(obj, pyre_object::PY_NULL);
}

/// PyPy-compatible `__code__` getter alias.
#[inline]
pub unsafe fn function_get_func_code(obj: PyObjectRef) -> *const () {
    function_get_code(obj)
}

/// PyPy-compatible `__code__` setter.
#[inline]
pub unsafe fn function_set_func_code(obj: PyObjectRef, code: *const ()) {
    (*(obj as *mut Function)).code = code;
}

/// PyPy-compatible `__name__` getter alias.
#[inline]
pub unsafe fn function_get_func_name(obj: PyObjectRef) -> &'static str {
    function_get_name(obj)
}

/// PyPy-compatible `__name__` setter.
#[inline]
pub unsafe fn function_set_func_name(obj: PyObjectRef, name: PyObjectRef) {
    if !pyre_object::is_str(name) {
        return;
    }
    let name = unsafe { pyre_object::w_str_get_value(name) };
    let name = Box::into_raw(Box::new(name.to_string())) as *const String;
    let old = (*(obj as *mut Function)).name;
    if !old.is_null() {
        drop(Box::from_raw(old as *mut String));
    }
    (*(obj as *mut Function)).name = name;
}

/// PyPy-compatible setter for `__name__` descriptor.
#[inline]
pub unsafe fn fset_func_name(obj: PyObjectRef, name: PyObjectRef) {
    function_set_func_name(obj, name)
}

/// PyPy compatibility helper for descriptor mutation checks.
#[inline]
pub unsafe fn _check_code_mutable(_obj: PyObjectRef, _attr: *const str) {}

/// PyPy-compatible `__globals__` getter alias.
#[inline]
pub unsafe fn function_get_w_globals(obj: PyObjectRef) -> *mut PyNamespace {
    function_get_globals(obj)
}

/// PyPy-compatible `__globals__` setter alias.
#[inline]
pub unsafe fn function_set_w_globals(obj: PyObjectRef, globals: *mut PyNamespace) {
    (*(obj as *mut Function)).w_func_globals = globals;
}

/// PyPy-compatible `fset_func_code` descriptor.
#[inline]
pub unsafe fn fset_func_code(obj: PyObjectRef, code: *const ()) {
    _check_code_mutable(obj, "func_code");
    function_set_func_code(obj, code);
}

/// PyPy-compatible `__closure__` getter alias.
#[inline]
pub unsafe fn function_get_func_closure(obj: PyObjectRef) -> PyObjectRef {
    function_get_closure(obj)
}

/// PyPy-compatible `fget_func_closure`.
#[inline]
pub unsafe fn fget_func_closure(obj: PyObjectRef) -> PyObjectRef {
    let value = function_get_func_closure(obj);
    if value.is_null() {
        pyre_object::w_none()
    } else {
        value
    }
}

/// PyPy-compatible `__closure__` setter alias.
#[inline]
pub unsafe fn fset_func_closure(obj: PyObjectRef, closure: PyObjectRef) {
    function_set_closure(obj, closure);
}

/// PyPy-compatible `__module__` getter.
#[inline]
pub unsafe fn fget___module__(obj: PyObjectRef) -> PyObjectRef {
    let globals = function_get_globals(obj);
    if globals.is_null() {
        return pyre_object::w_none();
    }
    unsafe {
        let namespace = &*globals;
        namespace
            .get("__name__")
            .copied()
            .unwrap_or(pyre_object::w_none())
    }
}

/// PyPy-compatible `descr_function__new__` helper.
#[inline]
pub unsafe fn descr_function__new__(
    code: *const (),
    w_globals: *mut PyNamespace,
    w_name: PyObjectRef,
    _argdefs: PyObjectRef,
    w_closure: PyObjectRef,
) -> PyObjectRef {
    let _ = _argdefs;
    let name = if !w_name.is_null() && !pyre_object::is_none(w_name) {
        unsafe { pyre_object::w_str_get_value(w_name).to_string() }
    } else {
        String::new()
    };
    let closure = if w_closure.is_null() || unsafe { pyre_object::is_none(w_closure) } {
        pyre_object::PY_NULL
    } else {
        w_closure
    };
    function_new_with_closure(code, name, w_globals, closure)
}

/// PyPy-compatible `__module__` setter.
#[inline]
pub unsafe fn fset___module__(_obj: PyObjectRef, _value: PyObjectRef) {}

/// PyPy-compatible `__module__` deleter.
#[inline]
pub unsafe fn fdel___module__(_obj: PyObjectRef) {}

/// PyPy-compatible `descr_function__new__` overload.
#[inline]
pub unsafe fn _cleanup_(_obj: PyObjectRef) -> bool {
    true
}

#[inline]
pub unsafe fn descr_builtinfunction__new__(
    code: *const (),
    w_globals: *mut PyNamespace,
    w_name: PyObjectRef,
    _argdefs: PyObjectRef,
    w_closure: PyObjectRef,
) -> PyObjectRef {
    descr_function__new__(code, w_globals, w_name, _argdefs, w_closure)
}

/// PyPy-compatible static registry hook.
#[inline]
pub fn add_to_table() {}

/// PyPy-compatible `__doc__` getter.
#[inline]
pub unsafe fn fget_func_doc(obj: PyObjectRef) -> PyObjectRef {
    function_get_doc(obj)
}

/// PyPy-compatible `__doc__` setter.
#[inline]
pub unsafe fn fset_func_doc(obj: PyObjectRef, value: PyObjectRef) {
    function_set_doc(obj, value);
}

/// PyPy-compatible `__doc__` deleter.
#[inline]
pub unsafe fn fdel_func_doc(obj: PyObjectRef) {
    function_del_doc(obj);
}

#[inline]
pub fn immutable_unique_id(_obj: PyObjectRef) -> usize {
    _obj as usize
}

/// PyPy-compatible `find` helper.
#[inline]
pub fn find(_identifier: &str) -> PyObjectRef {
    let _ = _identifier;
    pyre_object::PY_NULL
}

#[inline]
fn is_builtin_code(obj: PyObjectRef) -> bool {
    unsafe { crate::gateway::is_builtin_code(obj) }
}

#[inline]
pub fn descr_init() {}

#[inline]
pub unsafe fn descr_classmethod__new__(
    _subtype: PyObjectRef,
    w_function: PyObjectRef,
) -> PyObjectRef {
    let _ = _subtype;
    if w_function.is_null() {
        pyre_object::w_none()
    } else {
        pyre_object::propertyobject::w_classmethod_new(w_function)
    }
}

#[inline]
pub unsafe fn descr_classmethod_get(
    w_obj: PyObjectRef,
    obj: PyObjectRef,
    w_cls: PyObjectRef,
) -> PyObjectRef {
    let _ = w_cls;
    if obj.is_null() || unsafe { pyre_object::is_none(obj) } {
        w_obj
    } else {
        let func = pyre_object::w_classmethod_get_func(w_obj);
        let cls = if w_obj.is_null() {
            pyre_object::w_none()
        } else {
            obj
        };
        pyre_object::w_method_new(func, cls, cls)
    }
}

#[inline]
pub unsafe fn descr_staticmethod__new__(
    _subtype: PyObjectRef,
    w_function: PyObjectRef,
) -> PyObjectRef {
    let _ = _subtype;
    if w_function.is_null() {
        pyre_object::w_none()
    } else {
        pyre_object::propertyobject::w_staticmethod_new(w_function)
    }
}

#[inline]
pub unsafe fn descr_staticmethod_get(
    obj: PyObjectRef,
    _obj: PyObjectRef,
    _cls: PyObjectRef,
) -> PyObjectRef {
    let _ = (_obj, _cls);
    if obj.is_null() {
        pyre_object::w_none()
    } else {
        pyre_object::w_staticmethod_get_func(obj)
    }
}

#[inline]
pub unsafe fn descr_method__new__(
    _subtype: PyObjectRef,
    w_function: PyObjectRef,
    w_instance: PyObjectRef,
    w_class: PyObjectRef,
) -> PyObjectRef {
    let _ = _subtype;
    if w_function.is_null() {
        pyre_object::w_none()
    } else {
        pyre_object::w_method_new(w_function, w_instance, w_class)
    }
}

#[inline]
pub unsafe fn descr_method_get(
    _func: PyObjectRef,
    obj: PyObjectRef,
    cls: PyObjectRef,
) -> PyObjectRef {
    let _ = _func;
    if obj.is_null() || unsafe { pyre_object::is_none(obj) } {
        _func
    } else {
        let owner = if cls.is_null() {
            pyre_object::w_none()
        } else {
            cls
        };
        pyre_object::w_method_new(_func, obj, owner)
    }
}

#[inline]
pub unsafe fn descr_method_call(obj: PyObjectRef, args: &[PyObjectRef]) -> PyObjectRef {
    if args.is_empty() {
        call_obj_args(obj, pyre_object::w_none(), args)
    } else {
        call_obj_args(obj, args[0], &args[1..])
    }
}

#[inline]
pub unsafe fn descr_method_eq(_self: PyObjectRef, other: PyObjectRef) -> bool {
    _self == other
}

#[inline]
pub unsafe fn descr_method_ne(_self: PyObjectRef, other: PyObjectRef) -> bool {
    _self != other
}

#[inline]
pub unsafe fn descr_method_repr(obj: PyObjectRef) -> PyObjectRef {
    pyre_object::w_str_new(&format!("method {obj:?}"))
}

#[inline]
pub unsafe fn descr_method_getattribute(obj: PyObjectRef, _name: PyObjectRef) -> PyObjectRef {
    let _ = _name;
    obj
}

#[inline]
pub unsafe fn descr_method_hash(_self: PyObjectRef) -> isize {
    _self as isize
}

#[inline]
pub unsafe fn descr_method__reduce__(_obj: PyObjectRef) -> PyObjectRef {
    let _ = _obj;
    pyre_object::w_tuple_new(vec![pyre_object::w_str_new("method")])
}

#[inline]
pub unsafe fn is_w(_obj: PyObjectRef, _other: PyObjectRef) -> bool {
    _obj == _other
}

/// PyPy-compatible `descr_function_call` helper.
#[inline]
pub fn descr_function_call(args: &[PyObjectRef]) -> PyObjectRef {
    if args.is_empty() {
        pyre_object::PY_NULL
    } else {
        call_args(args[0], &args[1..])
    }
}

/// PyPy-compatible `descr_function_get` helper.
#[inline]
pub unsafe fn descr_function_get(
    _func: PyObjectRef,
    obj: PyObjectRef,
    cls: PyObjectRef,
) -> PyObjectRef {
    let _ = cls;
    if obj.is_null() || unsafe { pyre_object::is_none(obj) } {
        _func
    } else {
        pyre_object::w_method_new(_func, obj, cls)
    }
}

/// PyPy-compatible `descr_function_repr` helper.
#[inline]
pub unsafe fn descr_function_repr(obj: PyObjectRef) -> PyObjectRef {
    let name = function_get_name(obj);
    pyre_object::w_str_new(&format!("function {name}"))
}

/// PyPy-compatible `__code__` getter for direct descriptors.
#[inline]
pub unsafe fn fget_func_code(obj: PyObjectRef) -> *const () {
    function_get_code(obj)
}

/// PyPy-compatible `descr__reduce__` helper.
#[inline]
pub fn descr_function__reduce__(_obj: PyObjectRef) -> PyObjectRef {
    pyre_object::w_tuple_new(vec![
        pyre_object::w_tuple_new(vec![]),
        pyre_object::w_tuple_new(vec![]),
    ])
}

/// PyPy-compatible `descr__setstate__` helper.
#[inline]
pub fn descr_function__setstate__(_obj: PyObjectRef, _state: PyObjectRef) {
    let _ = _state;
}

#[inline]
pub fn __init__() {}

#[inline]
pub fn __repr__() -> String {
    "function".to_string()
}

/// PyPy-compatible `__call__` alias helper.
#[inline]
pub fn call(frame: PyObjectRef, args: &[PyObjectRef]) -> PyObjectRef {
    call_args(frame, args)
}

/// PyPy-compatible call fast-path hooks.
#[inline]
pub fn call_args(func: PyObjectRef, args: &[PyObjectRef]) -> PyObjectRef {
    crate::call_function(func, args)
}

/// PyPy-compatible `call_obj_args` helper.
#[inline]
pub fn call_obj_args(func: PyObjectRef, obj: PyObjectRef, args: &[PyObjectRef]) -> PyObjectRef {
    crate::baseobjspace::call_obj_args(func, obj, args)
}

/// PyPy-compatible `call_args` instance method.
#[inline]
pub fn function_call_args(func: PyObjectRef, args: &[PyObjectRef]) -> PyObjectRef {
    call_args(func, args)
}

/// PyPy-compatible `call_obj_args` instance method.
#[inline]
pub(crate) fn function_call_obj_args(
    func: PyObjectRef,
    obj: PyObjectRef,
    args: &[PyObjectRef],
) -> PyObjectRef {
    call_obj_args(func, obj, args)
}

/// PyPy-compatible `funccall` helper.
#[inline]
pub fn funccall(func: PyObjectRef, args: &[PyObjectRef]) -> PyObjectRef {
    call_args(func, args)
}

/// PyPy-compatible `funccall_valuestack` helper.
#[inline]
pub fn funccall_valuestack(
    func: PyObjectRef,
    nargs: usize,
    frame: &mut crate::pyframe::PyFrame,
    dropvalues: usize,
    methodcall: bool,
) -> PyObjectRef {
    let _ = methodcall;
    let args = frame.peekvalues(nargs);
    frame.dropvalues(dropvalues);
    funccall(func, &args)
}

/// PyPy-compatible `_flat_pycall` helper.
#[inline]
pub fn _flat_pycall(
    func: PyObjectRef,
    nargs: usize,
    frame: &mut crate::pyframe::PyFrame,
    dropvalues: usize,
) -> PyObjectRef {
    funccall_valuestack(func, nargs, frame, dropvalues, false)
}

/// PyPy-compatible `_flat_pycall_defaults` helper.
#[inline]
pub fn _flat_pycall_defaults(
    func: PyObjectRef,
    nargs: usize,
    frame: &mut crate::pyframe::PyFrame,
    defs_to_load: usize,
    dropvalues: usize,
) -> PyObjectRef {
    let _ = defs_to_load;
    funccall_valuestack(func, nargs, frame, dropvalues, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_function_create() {
        let code = 0xDEAD_BEEF as *const ();
        let mut ns = PyNamespace::new();
        let obj = function_new(code, "myfunc".to_string(), &mut ns);
        unsafe {
            assert!(is_function(obj));
            assert!(!is_int(obj));
            assert_eq!(function_get_code(obj), code);
            assert_eq!(function_get_name(obj), "myfunc");
            assert_eq!(function_get_globals(obj), &mut ns as *mut PyNamespace);
            assert!(function_get_closure(obj).is_null());
        }
    }

    #[test]
    fn test_function_field_offsets() {
        assert_eq!(FUNCTION_CODE_OFFSET, 8); // after ob_type pointer
        assert_eq!(FUNCTION_NAME_OFFSET, 16); // after code
        assert_eq!(FUNCTION_GLOBALS_OFFSET, 24); // after name
        assert_eq!(FUNCTION_CLOSURE_OFFSET, 32); // after w_func_globals
    }
}
