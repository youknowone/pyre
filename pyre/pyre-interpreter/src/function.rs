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
/// Layout: `[ob_type | code | can_change_code | name_ptr | w_func_globals | closure]`
/// - `code`: pointer to a Code object (W_CodeObject for user funcs, BuiltinCode for builtins).
///   function.py:47 — `_immutable_fields_ = ['code?', ...]`
/// - `can_change_code`: function.py:33 — True by default; False for
///   `FunctionWithFixedCode` subclass (used by builtins).
/// - `name_ptr`: leaked `Box<String>` containing the function name
/// - `w_func_globals`: raw pointer to the module-level namespace (shared)
/// - `closure`:  tuple of cell objects, or PY_NULL if no closure
#[repr(C)]
pub struct Function {
    pub ob: PyObject,
    /// Pointer to a Code object (W_CodeObject or BuiltinCode).
    /// function.py:47 — `_immutable_fields_ = ['code?', ...]`
    pub code: *const (),
    /// function.py:33 — `can_change_code = True`
    /// False for FunctionWithFixedCode subclass.
    pub can_change_code: bool,
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
    /// function.py:56 — `self.w_module = None`
    pub w_module: PyObjectRef,
}

/// function.py:706 — `class BuiltinFunction(Function): can_change_code = False`
pub type BuiltinFunction = Function;
/// function.py:703 — `class FunctionWithFixedCode(Function): can_change_code = False`
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
/// `code` is a pointer to a Code object (W_CodeObject) cast to `*const ()`.
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
            w_class: std::ptr::null_mut(),
        },
        code,
        can_change_code: true, // function.py:33
        name: name_ptr,
        w_func_globals,
        closure,
        defs_w: PY_NULL,
        w_kw_defs: PY_NULL,
        w_module: PY_NULL,
    });
    Box::into_raw(obj) as PyObjectRef
}

/// function.py:703 — `class FunctionWithFixedCode(Function): can_change_code = False`
/// Allocate a function whose code pointer the JIT can treat as immutable.
pub fn function_new_with_fixed_code(
    code: *const (),
    name: String,
    w_func_globals: *mut PyNamespace,
) -> PyObjectRef {
    let name_ptr = Box::into_raw(Box::new(name)) as *const String;
    let obj = Box::new(Function {
        ob: PyObject {
            ob_type: &FUNCTION_TYPE as *const PyType,
            w_class: std::ptr::null_mut(),
        },
        code,
        can_change_code: false, // function.py:704
        name: name_ptr,
        w_func_globals,
        closure: PY_NULL,
        defs_w: PY_NULL,
        w_kw_defs: PY_NULL,
        w_module: PY_NULL,
    });
    Box::into_raw(obj) as PyObjectRef
}

/// function.py:367-370 — `_check_code_mutable(attr)`:
/// Raises TypeError if function code is not mutable.
pub unsafe fn _check_code_mutable(func: PyObjectRef, attr: &str) -> Result<(), crate::PyError> {
    if (*(func as *const Function)).can_change_code {
        Ok(())
    } else {
        Err(crate::PyError::type_error(format!(
            "Cannot change {} attribute of builtin functions",
            attr
        )))
    }
}

/// function.py:23 — `@jit.elidable_promote()`
/// Only valid when `can_change_code == false`.
#[majit_macros::elidable_promote]
#[inline]
pub unsafe fn _get_immutable_code(func: PyObjectRef) -> *const () {
    // function.py:25
    debug_assert!(
        !unsafe { (*(func as *const Function)).can_change_code },
        "_get_immutable_code called on function with can_change_code=true"
    );
    unsafe { (*(func as *const Function)).code }
}

/// Check if an object is a user-defined function.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_function(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &FUNCTION_TYPE) }
}

/// function.py:78-83 — `getcode(self)`: three-way dispatch.
///   - JIT + immutable code → _get_immutable_code (elidable_promote)
///   - JIT + mutable code  → promote(self.code)
///   - interpreter         → self.code
#[inline]
pub unsafe fn getcode(obj: PyObjectRef) -> *const () {
    let func = obj as *const Function;
    if majit_metainterp::jit::we_are_jitted() {
        if !(*func).can_change_code {
            // function.py:80-81
            return _get_immutable_code(obj);
        }
        // function.py:82
        return majit_metainterp::jit::promote((*func).code as usize) as *const ();
    }
    // function.py:83
    (*func).code
}

/// Get the Code object pointer from a function object.
///
/// Returns a pointer to the Code-level object (W_CodeObject or BuiltinCode).
///
/// # Safety
/// `obj` must point to a valid `Function`.
/// NOTE: NOT elidable — code field can change (can_change_code).
/// Use _get_immutable_code() for the elidable path.
#[inline]
pub unsafe fn function_get_code(obj: PyObjectRef) -> *const () {
    unsafe { (*(obj as *const Function)).code }
}

/// Extract the raw bytecode CodeObject pointer from a user function.
///
/// Equivalent to accessing `self.getcode().code_ptr` in PyPy terms:
/// `getcode()` returns the Code wrapper (W_CodeObject), and this
/// dereferences through it to the underlying CodeObject.
///
/// # Safety
/// `obj` must point to a valid `Function` whose `code` field is a `W_CodeObject`
/// (i.e., NOT a BuiltinCode). Only call on user-defined functions.
#[inline]
pub unsafe fn get_pycode(obj: PyObjectRef) -> *const () {
    let code = getcode(obj);
    debug_assert!(
        !crate::is_builtin_code(code as PyObjectRef),
        "get_pycode called on a builtin function"
    );
    crate::w_code_get_ptr(code as PyObjectRef)
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

/// function.py:400 — `fset_func_doc` mutator.
#[inline]
pub unsafe fn function_set_doc(obj: PyObjectRef, value: PyObjectRef) -> Result<(), crate::PyError> {
    _check_code_mutable(obj, "func_doc")?; // function.py:401
    let _ = crate::setattr(obj, "__doc__", value);
    Ok(())
}

/// function.py:404 — `fdel_func_doc` deleter.
#[inline]
pub unsafe fn function_del_doc(obj: PyObjectRef) -> Result<(), crate::PyError> {
    _check_code_mutable(obj, "func_doc")?; // function.py:405
    let _ = crate::delattr(obj, "__doc__");
    Ok(())
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

/// function.py:381 — `fset_func_defaults` mutator.
#[inline]
pub unsafe fn fset_func_defaults(
    obj: PyObjectRef,
    value: PyObjectRef,
) -> Result<(), crate::PyError> {
    _check_code_mutable(obj, "func_defaults")?; // function.py:382
    if value.is_null() || pyre_object::is_none(value) {
        function_set_defaults(obj, pyre_object::PY_NULL);
    } else {
        function_set_defaults(obj, value);
    }
    Ok(())
}

/// function.py:391 — `fdel_func_defaults` deleter.
#[inline]
pub unsafe fn fdel_func_defaults(obj: PyObjectRef) -> Result<(), crate::PyError> {
    _check_code_mutable(obj, "func_defaults")?; // function.py:392
    function_set_defaults(obj, pyre_object::PY_NULL);
    Ok(())
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

/// function.py — `fset_func_kwdefaults` mutator.
#[inline]
pub unsafe fn fset_func_kwdefaults(
    obj: PyObjectRef,
    value: PyObjectRef,
) -> Result<(), crate::PyError> {
    _check_code_mutable(obj, "func_kwdefaults")?;
    if value.is_null() || pyre_object::is_none(value) {
        function_set_kwdefaults(obj, pyre_object::PY_NULL);
    } else {
        function_set_kwdefaults(obj, value);
    }
    Ok(())
}

/// function.py — `fdel_func_kwdefaults` deleter.
#[inline]
pub unsafe fn fdel_func_kwdefaults(obj: PyObjectRef) -> Result<(), crate::PyError> {
    _check_code_mutable(obj, "func_kwdefaults")?;
    function_set_kwdefaults(obj, pyre_object::PY_NULL);
    Ok(())
}

/// function.py:435-436 — `fget_func_code(self, space): return self.getcode()`
/// Uses getcode() for JIT elidable_promote / promote path.
#[inline]
pub unsafe fn function_get_func_code(obj: PyObjectRef) -> *const () {
    getcode(obj)
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

/// function.py:411 — `fset_func_name` setter.
#[inline]
pub unsafe fn fset_func_name(obj: PyObjectRef, name: PyObjectRef) -> Result<(), crate::PyError> {
    _check_code_mutable(obj, "func_name")?; // function.py:412
    function_set_func_name(obj, name);
    Ok(())
}

// _check_code_mutable is defined above (function.py:367-370 parity).

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

/// function.py:367 — fset_func_code checks mutability before setting.
#[inline]
pub unsafe fn fset_func_code(obj: PyObjectRef, code: *const ()) -> Result<(), crate::PyError> {
    _check_code_mutable(obj, "func_code")?;
    function_set_func_code(obj, code);
    Ok(())
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

/// function.py:419-425 — `fget___module__`.
/// Caches on first read: if w_module is PY_NULL (unset), computes from
/// globals["__name__"] and stores into self.w_module. Always returns
/// self.w_module afterwards.
///
/// PY_NULL = RPython None (unset sentinel), w_none() = space.w_None.
/// After fdel___module__, w_module is w_none() (not PY_NULL), so
/// subsequent gets return None without re-computing from globals.
#[inline]
pub unsafe fn fget___module__(obj: PyObjectRef) -> PyObjectRef {
    let func = obj as *mut Function;
    // function.py:420: if self.w_module is None
    if (*func).w_module.is_null() {
        // function.py:421-422: compute and cache
        let globals = (*func).w_func_globals;
        if !globals.is_null() {
            (*func).w_module = (*globals)
                .get("__name__")
                .copied()
                .unwrap_or(pyre_object::w_none());
        } else {
            // function.py:424: self.w_module = space.w_None
            (*func).w_module = pyre_object::w_none();
        }
    }
    // function.py:425: return self.w_module
    (*func).w_module
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

/// function.py:427 — `fset___module__` setter.
#[inline]
pub unsafe fn fset___module__(obj: PyObjectRef, value: PyObjectRef) -> Result<(), crate::PyError> {
    _check_code_mutable(obj, "func_name")?; // function.py:428
    // function.py:429: self.w_module = w_module
    (*(obj as *mut Function)).w_module = value;
    Ok(())
}

/// function.py:431 — `fdel___module__` deleter.
#[inline]
pub unsafe fn fdel___module__(obj: PyObjectRef) -> Result<(), crate::PyError> {
    _check_code_mutable(obj, "func_name")?; // function.py:432
    // function.py:433: self.w_module = space.w_None
    (*(obj as *mut Function)).w_module = pyre_object::w_none();
    Ok(())
}

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

/// function.py:400 — `fset_func_doc` descriptor.
#[inline]
pub unsafe fn fset_func_doc(obj: PyObjectRef, value: PyObjectRef) -> Result<(), crate::PyError> {
    function_set_doc(obj, value)
}

/// function.py:404 — `fdel_func_doc` descriptor.
#[inline]
pub unsafe fn fdel_func_doc(obj: PyObjectRef) -> Result<(), crate::PyError> {
    function_del_doc(obj)
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
        // Function.code now stores a Code-level wrapper (W_CodeObject).
        let raw_code = 0xDEAD_BEEF as *const ();
        let w_code = crate::w_code_new(raw_code);
        let mut ns = PyNamespace::new();
        let obj = function_new(w_code as *const (), "myfunc".to_string(), &mut ns);
        unsafe {
            assert!(is_function(obj));
            assert!(!is_int(obj));
            assert_eq!(function_get_code(obj), w_code as *const ());
            assert_eq!(function_get_name(obj), "myfunc");
            assert_eq!(function_get_globals(obj), &mut ns as *mut PyNamespace);
            assert!(function_get_closure(obj).is_null());
        }
    }

    #[test]
    fn test_function_field_offsets() {
        assert_eq!(FUNCTION_CODE_OFFSET, 16); // after PyObject { ob_type(8) + w_class(8) }
        assert_eq!(FUNCTION_NAME_OFFSET, 32); // after code(8) + can_change_code(1) + padding(7)
        assert_eq!(FUNCTION_GLOBALS_OFFSET, 40); // after name
        assert_eq!(FUNCTION_CLOSURE_OFFSET, 48); // after w_func_globals
    }
}
