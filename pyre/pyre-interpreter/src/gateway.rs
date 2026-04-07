//! Built-in function objects.
//!
//! A `BuiltinCode` wraps a Rust function pointer that implements
//! a Python builtin like `print`, `len`, etc.

use pyre_object::pyobject::*;

#[derive(Debug, Clone, Default)]
pub struct SignatureBuilder {
    pub name: &'static str,
    pub argnames: Vec<&'static str>,
    pub varargname: Option<&'static str>,
    pub kwargname: Option<&'static str>,
}

impl SignatureBuilder {
    pub fn append(&mut self, argname: &'static str) {
        self.argnames.push(argname);
    }

    pub fn signature(&self) -> Signature {
        Signature {
            argnames: self.argnames.clone(),
            varargname: self.varargname,
            kwargname: self.kwargname,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Signature {
    pub argnames: Vec<&'static str>,
    pub varargname: Option<&'static str>,
    pub kwargname: Option<&'static str>,
}

#[derive(Debug, Clone)]
pub struct Unwrapper;

impl Unwrapper {
    pub fn unwrap(&self, _space: PyObjectRef, _value: PyObjectRef) -> PyObjectRef {
        let _ = (_space, _value);
        std::ptr::null_mut()
    }
}

#[derive(Debug, Clone)]
pub struct UnwrapSpecRecipe {
    pub miniglobals: Vec<PyObjectRef>,
}

impl UnwrapSpecRecipe {
    pub fn dispatch<T>(&self, _el: T, _args: &mut Vec<PyObjectRef>) {
        let _ = (&self.miniglobals, _el, _args);
    }

    pub fn apply_over(&self, _unwrap_spec: &[&str], _space: PyObjectRef, _name: &str) {
        let _ = (_unwrap_spec, _space, _name);
    }
}

#[derive(Debug, Clone)]
pub struct UnwrapSpecEmit;

impl UnwrapSpecEmit {
    pub fn new() -> Self {
        Self
    }

    pub fn succ(&mut self) -> usize {
        0
    }

    pub fn use_name(&mut self, obj: &'static str) -> &'static str {
        obj
    }
}

#[derive(Debug, Clone)]
pub struct UnwrapSpec_Check {
    pub func: PyObjectRef,
}

impl UnwrapSpec_Check {
    pub fn new(func: PyObjectRef, _argnames: &[&'static str]) -> Self {
        Self { func }
    }
}

#[derive(Debug, Clone)]
pub struct UnwrapSpec_EmitRun;

#[derive(Debug, Clone)]
pub struct UnwrapSpec_EmitShortcut;

#[derive(Debug, Clone)]
pub struct UnwrapSpec_FastFunc_Unwrap;

#[derive(Debug, Clone)]
pub struct FastFuncNotSupported;

#[derive(Debug, Clone)]
pub struct BuiltinActivation;

#[derive(Debug, Clone)]
pub struct GatewayCache;

#[derive(Debug, Clone)]
pub struct BuiltinCodePassThroughArguments0 {
    pub code: PyObjectRef,
}

#[derive(Debug, Clone)]
pub struct BuiltinCodePassThroughArguments1 {
    pub code: PyObjectRef,
}

#[derive(Debug, Clone)]
pub struct BuiltinCode0 {
    pub code: PyObjectRef,
}

#[derive(Debug, Clone)]
pub struct BuiltinCode1 {
    pub code: PyObjectRef,
}

#[derive(Debug, Clone)]
pub struct BuiltinCode2 {
    pub code: PyObjectRef,
}

#[derive(Debug, Clone)]
pub struct BuiltinCode3 {
    pub code: PyObjectRef,
}

#[derive(Debug, Clone)]
pub struct BuiltinCode4 {
    pub code: PyObjectRef,
}

#[derive(Debug, Clone)]
pub struct WrappedDefault;

#[derive(Debug, Clone)]
pub struct ApplevelClass {
    pub source: Option<PyObjectRef>,
}

#[derive(Debug, Clone)]
pub struct ApplevelCache {
    pub base: GatewayCache,
}

#[allow(non_camel_case_types)]
pub type interp2app = BuiltinCode;

#[allow(non_camel_case_types)]
pub type interp2app_temp = interp2app;

#[allow(non_camel_case_types)]
pub type applevel_temp = ApplevelClass;

pub fn build_applevel_dict(_space: PyObjectRef) -> PyObjectRef {
    std::ptr::null_mut()
}

pub fn build_unwrap_spec(
    _func: PyObjectRef,
    _argnames: &[&str],
    _self_type: Option<&str>,
) -> UnwrapSpecRecipe {
    let _ = (_func, _argnames, _self_type);
    UnwrapSpecRecipe {
        miniglobals: Vec::new(),
    }
}

pub fn int_unwrapping_space_method<T>(_typ: T) -> &'static str {
    let _ = _typ;
    "int"
}

pub fn interp2app(func: PyObjectRef) -> PyObjectRef {
    let _ = func;
    make_builtin_function("interp2app", |_| Ok(std::ptr::null_mut()))
}

pub fn interp2app_temp(func: PyObjectRef) -> PyObjectRef {
    interp2app(func)
}

pub fn interpindirect2app(
    unbound_meth: PyObjectRef,
    _unwrap_spec: Option<&UnwrapSpecRecipe>,
) -> PyObjectRef {
    let _ = _unwrap_spec;
    interp2app(unbound_meth)
}

pub fn unwrap_spec(_spec: &[&'static str]) -> PyObjectRef {
    let _ = _spec;
    make_builtin_function("unwrap", |_| Ok(std::ptr::null_mut()))
}

pub fn appdef(
    source: &'static str,
    _applevel: ApplevelClass,
    _filename: Option<&str>,
) -> PyObjectRef {
    let _ = (source, _filename);
    std::ptr::null_mut()
}

pub fn app2interp_temp(func: PyObjectRef, _filename: Option<&str>) -> PyObjectRef {
    let _ = _filename;
    interp2app(func)
}

pub fn app2interp(func: PyObjectRef, _filename: Option<&str>) -> PyObjectRef {
    app2interp_temp(func, _filename)
}

pub fn applevel_temp(_func: PyObjectRef, _filename: Option<&str>) -> PyObjectRef {
    let _ = _filename;
    std::ptr::null_mut()
}

impl UnwrapSpec_FastFunc_Unwrap {
    pub fn visit_nonnegint(&mut self) {}
}

impl UnwrapSpec_EmitShortcut {
    pub fn handle(self) {}
}

/// Type descriptor for built-in functions.
pub static BUILTIN_CODE_TYPE: PyType = PyType {
    tp_name: "builtin_function_or_method",
};

/// Signature of a built-in function.
///
/// PyPy: all interp-level functions can raise OperationError.
/// pyre equivalent: returns Result so errors propagate through the call stack.
pub type BuiltinCodeFn = fn(&[PyObjectRef]) -> Result<PyObjectRef, crate::PyError>;

/// A built-in function object.
#[repr(C)]
pub struct BuiltinCode {
    pub ob: PyObject,
    pub name: &'static str,
    pub func: BuiltinCodeFn,
}

/// Allocate a new `BuiltinCode`.
pub fn builtin_code_new(name: &'static str, func: BuiltinCodeFn) -> PyObjectRef {
    let obj = Box::new(BuiltinCode {
        ob: PyObject {
            ob_type: &BUILTIN_CODE_TYPE,
        },
        name,
        func,
    });
    Box::into_raw(obj) as PyObjectRef
}

/// Check if an object is a built-in function.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_builtin_code(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &BUILTIN_CODE_TYPE) }
}

/// Get the function pointer from a built-in function object.
///
/// # Safety
/// `obj` must point to a valid `BuiltinCode`.
#[inline]
pub unsafe fn builtin_code_get(obj: PyObjectRef) -> BuiltinCodeFn {
    let func_obj = obj as *const BuiltinCode;
    unsafe { (*func_obj).func }
}

/// Get the name of a built-in function.
///
/// # Safety
/// `obj` must point to a valid `BuiltinCode`.
#[inline]
pub unsafe fn builtin_code_name(obj: PyObjectRef) -> &'static str {
    let func_obj = obj as *const BuiltinCode;
    unsafe { (*func_obj).name }
}

/// gateway.py GatewayCache.build() parity — wrap a BuiltinCodeFn as FunctionWithFixedCode.
///
/// Creates a BuiltinCode (Code object) and wraps it in a Function with
/// `can_change_code = false`, matching PyPy's:
///   `fn = FunctionWithFixedCode(space, code, None, defs, forcename=gateway.name)`
pub fn make_builtin_function(name: &'static str, func: BuiltinCodeFn) -> PyObjectRef {
    let code = builtin_code_new(name, func);
    crate::function_new_with_fixed_code(code as *const (), name.to_string(), std::ptr::null_mut())
}
