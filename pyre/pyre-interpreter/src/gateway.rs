//! Built-in function objects.
#![allow(non_camel_case_types)]
//!
//! A `BuiltinCode` wraps a Rust function pointer that implements
//! a Python builtin like `print`, `len`, etc.

use pyre_object::pyobject::*;

/// pypy/interpreter/gateway.py:36-73 `class SignatureBuilder`.
///
/// ```python
/// class SignatureBuilder(object):
///     def __init__(self, ...):
///         ...
///         self.posonlyargcount = 0
///         self.kwonlystartindex = -1
///
///     def append(self, argname):
///         self.argnames.append(argname)
///
///     def marker_posonly(self):
///         assert self.posonlyargcount == 0
///         assert self.kwonlystartindex == -1
///         self.posonlyargcount = len(self.argnames)
///
///     def marker_kwonly(self):
///         assert self.kwonlystartindex == -1
///         self.kwonlystartindex = len(self.argnames)
///
///     def signature(self):
///         if self.kwonlystartindex == -1:
///             kwonlyargcount = 0
///         else:
///             kwonlyargcount = len(self.argnames) - self.kwonlystartindex
///         return Signature(self.argnames,
///                          self.varargname, self.kwargname,
///                          kwonlyargcount, self.posonlyargcount)
/// ```
///
/// Pyre carries `kwonlystartindex` rather than `kwonlyargcount`
/// directly so the marker-driven build (`marker_kwonly()` records
/// where the kw-only tail starts; `signature()` derives the count at
/// build time) matches PyPy 1:1.  The `-1` sentinel encodes "no
/// `marker_kwonly` call seen yet".
#[derive(Debug, Clone)]
pub struct SignatureBuilder {
    pub name: &'static str,
    pub argnames: Vec<&'static str>,
    pub varargname: Option<&'static str>,
    pub kwargname: Option<&'static str>,
    pub posonlyargcount: usize,
    pub kwonlystartindex: isize,
}

impl Default for SignatureBuilder {
    fn default() -> Self {
        Self {
            name: "",
            argnames: Vec::new(),
            varargname: None,
            kwargname: None,
            posonlyargcount: 0,
            kwonlystartindex: -1,
        }
    }
}

impl SignatureBuilder {
    /// gateway.py:54-55 `append`.
    pub fn append(&mut self, argname: &'static str) {
        self.argnames.push(argname);
    }

    /// gateway.py:57-60 `marker_posonly`.  PyPy asserts the marker is
    /// emitted at most once and before `marker_kwonly`.
    pub fn marker_posonly(&mut self) {
        assert!(self.posonlyargcount == 0);
        assert!(self.kwonlystartindex == -1);
        self.posonlyargcount = self.argnames.len();
    }

    /// gateway.py:62-64 `marker_kwonly`.  PyPy asserts the marker is
    /// emitted at most once.
    pub fn marker_kwonly(&mut self) {
        assert!(self.kwonlystartindex == -1);
        self.kwonlystartindex = self.argnames.len() as isize;
    }

    /// gateway.py:66-73 `signature`.  Derives `kwonlyargcount` from
    /// the argname list length minus the `marker_kwonly` index, or 0
    /// if the marker never fired.
    pub fn signature(&self) -> Signature {
        let kwonlyargcount = if self.kwonlystartindex == -1 {
            0
        } else {
            self.argnames.len() - self.kwonlystartindex as usize
        };
        Signature {
            argnames: self.argnames.clone(),
            varargname: self.varargname,
            kwargname: self.kwargname,
            kwonlyargcount,
            posonlyargcount: self.posonlyargcount,
        }
    }
}

/// pypy/interpreter/signature.py:3-78 `class Signature`.
///
/// ```python
/// class Signature(object):
///     _immutable_ = True
///     _immutable_fields_ = ["argnames[*]"]
///     __slots__ = ("argnames", "posonlyargcount", "kwonlyargcount",
///                  "varargname", "kwargname")
/// ```
///
/// `argnames` contains both the positional-only and the positional
/// arguments; the count of positional-only arguments is
/// `posonlyargcount`.  Keyword-only argument names live at the tail
/// of `argnames` and are counted by `kwonlyargcount`.
#[derive(Debug, Clone, PartialEq)]
pub struct Signature {
    pub argnames: Vec<&'static str>,
    pub varargname: Option<&'static str>,
    pub kwargname: Option<&'static str>,
    pub posonlyargcount: usize,
    pub kwonlyargcount: usize,
}

impl Signature {
    /// pypy/interpreter/signature.py:8-16 `Signature.__init__`.
    pub fn new(
        argnames: Vec<&'static str>,
        varargname: Option<&'static str>,
        kwargname: Option<&'static str>,
        kwonlyargcount: usize,
        posonlyargcount: usize,
    ) -> Self {
        Self {
            argnames,
            varargname,
            kwargname,
            posonlyargcount,
            kwonlyargcount,
        }
    }

    /// pypy/interpreter/signature.py:18-24 `find_argname`:
    /// ```python
    /// @jit.elidable
    /// def find_argname(self, name):
    ///     try:
    ///         return self.argnames.index(name)
    ///     except ValueError:
    ///         pass
    ///     return -1
    /// ```
    pub fn find_argname(&self, name: &str) -> isize {
        for (i, arg) in self.argnames.iter().enumerate() {
            if *arg == name {
                return i as isize;
            }
        }
        -1
    }

    /// pypy/interpreter/signature.py:26-31 `find_w_argname`:
    /// ```python
    /// @jit.elidable
    /// def find_w_argname(self, w_name):
    ///     for i, name in enumerate(self.argnames):
    ///         if w_name.eq_unwrapped(name):
    ///             return i
    ///     return -1
    /// ```
    ///
    /// `w_name.eq_unwrapped(name)` compares the wrapped string with a
    /// raw `&str`; pyre delegates to `find_argname` after unwrapping the
    /// PyObject via `w_str_get_value`.  Non-string `w_name` returns `-1`
    /// (matches PyPy's RPython unwrap-or-fail semantics for strings).
    pub fn find_w_argname(&self, w_name: PyObjectRef) -> isize {
        if w_name.is_null() {
            return -1;
        }
        unsafe {
            if !pyre_object::is_str(w_name) {
                return -1;
            }
            let name = pyre_object::w_str_get_value(w_name);
            self.find_argname(name)
        }
    }

    /// pypy/interpreter/signature.py:33-34 `num_argnames`:
    /// ```python
    /// def num_argnames(self):
    ///     return len(self.argnames) - self.kwonlyargcount
    /// ```
    pub fn num_argnames(&self) -> usize {
        self.argnames.len() - self.kwonlyargcount
    }

    /// pypy/interpreter/signature.py:36-37 `num_posonlyargnames`:
    /// ```python
    /// def num_posonlyargnames(self):
    ///     return self.posonlyargcount
    /// ```
    pub fn num_posonlyargnames(&self) -> usize {
        self.posonlyargcount
    }

    /// pypy/interpreter/signature.py:39-40 `num_kwonlyargnames`:
    /// ```python
    /// def num_kwonlyargnames(self):
    ///     return self.kwonlyargcount
    /// ```
    pub fn num_kwonlyargnames(&self) -> usize {
        self.kwonlyargcount
    }

    /// pypy/interpreter/signature.py:42-43 `has_vararg`:
    /// ```python
    /// def has_vararg(self):
    ///     return self.varargname is not None
    /// ```
    pub fn has_vararg(&self) -> bool {
        self.varargname.is_some()
    }

    /// pypy/interpreter/signature.py:45-46 `has_kwarg`:
    /// ```python
    /// def has_kwarg(self):
    ///     return self.kwargname is not None
    /// ```
    pub fn has_kwarg(&self) -> bool {
        self.kwargname.is_some()
    }

    /// pypy/interpreter/signature.py:48-52 `scope_length`:
    /// ```python
    /// def scope_length(self):
    ///     scopelen = len(self.argnames)
    ///     scopelen += self.has_vararg()
    ///     scopelen += self.has_kwarg()
    ///     return scopelen
    /// ```
    pub fn scope_length(&self) -> usize {
        let mut scopelen = self.argnames.len();
        if self.has_vararg() {
            scopelen += 1;
        }
        if self.has_kwarg() {
            scopelen += 1;
        }
        scopelen
    }

    /// pypy/interpreter/signature.py:54-60 `getallvarnames`:
    /// ```python
    /// def getallvarnames(self):
    ///     argnames = self.argnames
    ///     if self.varargname is not None:
    ///         argnames = argnames + [self.varargname]
    ///     if self.kwargname is not None:
    ///         argnames = argnames + [self.kwargname]
    ///     return argnames
    /// ```
    pub fn getallvarnames(&self) -> Vec<&'static str> {
        let mut argnames = self.argnames.clone();
        if let Some(name) = self.varargname {
            argnames.push(name);
        }
        if let Some(name) = self.kwargname {
            argnames.push(name);
        }
        argnames
    }
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

impl Default for UnwrapSpecEmit {
    fn default() -> Self {
        Self::new()
    }
}

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

/// Type descriptor for built-in code objects.
///
/// PyPy typedef.py: BuiltinCode.typedef = TypeDef('builtin-code', ...)
pub static BUILTIN_CODE_TYPE: PyType = pyre_object::pyobject::new_pytype("builtin-code");

/// GC type id assigned to `BuiltinCode` at JitDriver init time. Held
/// as a constant alongside the struct (rather than runtime-queried) so
/// the allocation hook can reach it without a back-channel, mirroring
/// `W_INT_GC_TYPE_ID` / `W_FLOAT_GC_TYPE_ID`. `pyre/pyre-jit/src/eval.rs`
/// asserts the same id is returned by `gc.register_type(...)` so any
/// drift panics on startup.
pub const BUILTIN_CODE_GC_TYPE_ID: u32 = 13;

/// Signature of a built-in function.
///
/// PyPy: all interp-level functions can raise OperationError.
/// pyre equivalent: returns Result so errors propagate through the call stack.
pub type BuiltinCodeFn = fn(&[PyObjectRef]) -> Result<PyObjectRef, crate::PyError>;

/// Cold interp2app arity-error formatter.
///
/// RPython's gateway constructs this exception only on the rejected-call
/// branch.  Keep formatting out of generated wrapper JitCodes; valid calls
/// retain the argument-count guards but never execute this residual helper.
#[majit_macros::dont_look_inside]
pub fn method_arity_failure(
    name: &str,
    expected: &str,
    given: usize,
) -> Result<PyObjectRef, crate::PyError> {
    Err(crate::PyError::type_error(format!(
        "{name}() takes {expected} ({given} given)"
    )))
}

/// Cold zero-user-argument gateway failure.
///
/// Valid generated wrappers guard their exact total arity before entering
/// this helper.  The rejected branch still distinguishes CALL_KW's trailing
/// marker from surplus positional arguments, preserving the public gateway
/// error while keeping keyword classification out of the hot JitCode.
#[majit_macros::dont_look_inside]
pub fn method_noarg_failure(
    args: &[PyObjectRef],
    name: &str,
    receiver_slots: usize,
) -> Result<PyObjectRef, crate::PyError> {
    if crate::builtins::has_builtin_kwargs(args) {
        Err(crate::PyError::type_error(format!(
            "{name}() takes no keyword arguments"
        )))
    } else {
        method_arity_failure(
            name,
            "no arguments",
            args.len().saturating_sub(receiver_slots),
        )
    }
}

/// Cold gateway failure for a missing required argument.
///
/// A wrapper whose optional trailing parameters turn the accepted count into a
/// range reports the PyArg_UnpackTuple `expected at least N arguments, got M`
/// form.  The wrapper keeps the count guard; the message is built only here.
#[majit_macros::dont_look_inside]
pub fn method_min_arity_failure(
    name: &str,
    min: usize,
    given: usize,
) -> Result<PyObjectRef, crate::PyError> {
    Err(crate::PyError::type_error(format!(
        "{name} expected at least {min} argument{}, got {given}",
        if min == 1 { "" } else { "s" },
    )))
}

/// Translation-visible registry of generated interp2app gateway bodies.
///
/// RPython's `BuiltinCode.func` is a PBC whose possible function values are
/// discovered by the annotator and become the candidate graph list on the
/// eventual `indirect_call`.  Rust erases that family to the bare
/// [`BuiltinCodeFn`] type, so `#[pyre_methods]` publishes the same set here.
/// The registry is process-global, matching the immutable generated function
/// objects; it is not runtime interpreter state.
#[derive(Clone, Copy)]
pub struct BuiltinWrapperDescriptor {
    pub path: &'static str,
    pub func: BuiltinCodeFn,
}

#[cfg(not(target_arch = "wasm32"))]
#[linkme::distributed_slice]
pub static BUILTIN_WRAPPER_DESCRIPTORS: [BuiltinWrapperDescriptor];

/// The type a method descriptor belongs to, and the layout test its receiver
/// must satisfy — `PyDescrObject.d_type`, and the `self` entry of PyPy's
/// `interp2app` unwrap_spec.
///
/// An unbound descriptor validates its receiver against this before the
/// implementation runs, so a foreign object can never reach an accessor that
/// reads the receiver's payload at the owning type's layout.  Without it,
/// `str.split(x)` on an `x` whose `__class__` merely claims to be `str`
/// reaches `w_str_get_wtf8` and dereferences arbitrary memory.
pub struct MethodOwner {
    /// The owning type's name as it appears in the mismatch message.
    pub type_name: &'static str,
    /// Layout membership, true for instances of subclasses as well.  `None`
    /// for a type with no layout test written yet: the receiver still has to
    /// be *present*, and the owner still names the descriptor in every error
    /// it raises, but a foreign object of the right shape is not rejected.
    pub is_instance: Option<fn(PyObjectRef) -> bool>,
}

/// A built-in function object.
///
/// `docstring` mirrors PyPy `BuiltinCode.docstring` (gateway.py:673
/// `self.docstring = doc or func.__doc__`). It is consulted by
/// `BuiltinCode::getdocstring`, which is the lazy fallback used by
/// `Function.fget_func_doc` (function.py:395-398).
#[repr(C)]
pub struct BuiltinCode {
    pub ob: PyObject,
    pub name: &'static str,
    pub func: BuiltinCodeFn,
    pub docstring: Option<&'static str>,
    /// eval.py:16-23 — `fast_natural_arity`. For builtins with fixed
    /// positional arity 0-4, this equals the arity directly. Builtins
    /// with optional/variadic args use HOPELESS (0x400).
    pub fast_natural_arity: u16,
    /// gateway.py:743 `BuiltinCode.sig` — the argument `Signature`
    /// (named params, `*args`/`**kwargs`, kw-only tail) used to bind
    /// keyword arguments into positional order before the function runs.
    /// `null` means "no declared signature" (positional-only), which is
    /// every builtin today.  The pointee is leaked to `'static` so the
    /// raw pointer carries no Drop obligation and is not a GC pointer,
    /// matching the `func` function-pointer convention.
    pub sig: *const Signature,
    /// The type this code object is a descriptor of, or null for a builtin
    /// with no receiver to check (a module-level function, a `__new__`
    /// carrier, a class or static method).  `stamp_method_owners` fills it
    /// in once a type's namespace is fully populated, mirroring
    /// `PyType_Ready` handing each `PyMethodDef` to `PyDescr_NewMethod`.
    /// The pointee is `'static`, so it carries no Drop obligation and is not
    /// a GC pointer.
    pub owner: *const MethodOwner,
    /// The module this function is defined in — `PyCFunctionObject.m_module`.
    /// Empty for a builtin that is not a module-level function, and for
    /// `builtins` itself, which is the module `_PyObject_FunctionStr` leaves
    /// off (`len()`, not `builtins.len()`).  `'static` like `name`, so it is
    /// neither a Drop obligation nor a GC pointer.
    pub module: &'static str,
}

/// Fixed payload size used by `gct_fv_gc_malloc`'s `c_size`
/// (`framework.py:811`). The payload has no inline GC pointers (`name`
/// / `docstring` are `'static` slices, `func` is a function pointer,
/// `ob.w_class` follows the existing W_IntObject / W_FloatObject
/// convention of leaving typeptr fixups out of `gc_ptr_offsets`).
pub const BUILTIN_CODE_OBJECT_SIZE: usize = std::mem::size_of::<BuiltinCode>();

impl pyre_object::lltype::GcType for BuiltinCode {
    fn type_id() -> u32 {
        BUILTIN_CODE_GC_TYPE_ID
    }
    const SIZE: usize = BUILTIN_CODE_OBJECT_SIZE;
}

/// eval.py:16 — `FLATPYCALL = 0x100`.
pub const FLATPYCALL: u16 = 0x100;
/// eval.py:17 — `PASSTHROUGHARGS1 = 0x200`.
pub const PASSTHROUGHARGS1: u16 = 0x200;
/// eval.py:18 — `HOPELESS = 0x400`. Default for code that cannot fast-path.
pub const HOPELESS: u16 = 0x400;

/// Allocate a new `BuiltinCode` with no docstring.
/// `fast_natural_arity` defaults to HOPELESS (no fast path).
pub fn builtin_code_new(name: &'static str, func: BuiltinCodeFn) -> PyObjectRef {
    builtin_code_new_with_doc(name, func, None)
}

/// Allocate a new `BuiltinCode` with known fixed arity (0-4).
/// gateway.py:843 — `self.__class__ = globals()['BuiltinCode%d' % arity]`
pub fn builtin_code_new_with_arity(
    name: &'static str,
    func: BuiltinCodeFn,
    arity: u16,
) -> PyObjectRef {
    debug_assert!(
        arity <= 4,
        "builtin arity {arity} for {name} exceeds fast-path max 4"
    );
    builtin_code_new_full(name, func, None, arity, std::ptr::null())
}

/// Allocate a new `BuiltinCode` with an explicit docstring.
///
/// PyPy gateway.py:673 — `self.docstring = doc or func.__doc__`. Pyre has
/// no introspection of `func.__doc__`, so callers must pass the docstring
/// explicitly when registering a builtin.
pub fn builtin_code_new_with_doc(
    name: &'static str,
    func: BuiltinCodeFn,
    docstring: Option<&'static str>,
) -> PyObjectRef {
    builtin_code_new_full(name, func, docstring, HOPELESS, std::ptr::null())
}

/// Allocate a new `BuiltinCode` with `fast_natural_arity = PASSTHROUGHARGS1`.
///
/// PyPy gateway.py — picks `BuiltinCodePassThroughArguments1` when the
/// `unwrap_spec` is `[W_Root, Arguments]`. `funcrun_obj` then receives the
/// first positional unwrapped (`w_obj`) and the rest as an `Arguments`
/// object. Pyre's single `BuiltinCodeFn` signature already takes a flat
/// slice, so the same closure shape works — the dispatch path in
/// `function.rs:funccall_valuestack` peeks `args[0]` separately to mirror
/// `function.py:194-199`, but the closure still receives `[w_obj, ...rest]`.
pub fn builtin_code_new_passthrough_args1(name: &'static str, func: BuiltinCodeFn) -> PyObjectRef {
    builtin_code_new_full(name, func, None, PASSTHROUGHARGS1, std::ptr::null())
}

/// Full constructor for `BuiltinCode`.  `sig` is a `*const Signature`
/// (null for positional-only builtins); the pointee must outlive the
/// object — callers leak it to `'static`.
///
/// `#[dont_look_inside]` (`@jit.dont_look_inside`, `rlib/jit.py:139`), the
/// `w_dict_new` twin: the body boxes a `BuiltinCode` through the non-numeric
/// `malloc_typed` (`fuse_boxing_alloc` fuses only the numeric boxes), so
/// tracing into it carries the unported `malloc->new` lowering into the caller.
/// Residualise the whole constructor — the JIT models it by signature as a
/// plain `PyObjectRef` GCREF and emits a residual call. Every
/// `builtin_code_new*` / `make_builtin_function_with_arity` head bottoms out
/// here.
#[majit_macros::dont_look_inside]
fn builtin_code_new_full(
    name: &'static str,
    func: BuiltinCodeFn,
    docstring: Option<&'static str>,
    fast_natural_arity: u16,
    sig: *const Signature,
) -> PyObjectRef {
    pyre_object::lltype::malloc_typed(BuiltinCode {
        ob: PyObject {
            ob_type: &BUILTIN_CODE_TYPE,
            w_class: pyre_object::pyobject::get_instantiate(&BUILTIN_CODE_TYPE),
        },
        name,
        func,
        docstring,
        fast_natural_arity,
        sig,
        owner: std::ptr::null(),
        module: "",
    }) as PyObjectRef
}

/// Record the module a function object was defined in and hand the object
/// back, so a registration table can wrap its constructor in place.  A
/// non-`BuiltinCode` callable (an app-level def installed by the same table)
/// passes through untouched.
///
/// The first writer wins: a module registered under two names, or one whose
/// namespace is swept after the table already stamped it, keeps the module
/// that defined the function rather than the one that re-exported it.
pub fn with_module(module: &'static str, func: PyObjectRef) -> PyObjectRef {
    unsafe {
        // A module namespace holds every kind of value, so the callable check
        // has to come before `getcode` reads a `Function` field off it.
        if func.is_null() || !crate::function::is_function(func) {
            return func;
        }
        let code = crate::function::getcode(func) as PyObjectRef;
        if !code.is_null() && is_builtin_code(code) {
            let code = code as *mut BuiltinCode;
            if std::ptr::read(&raw const (*code).module).is_empty() {
                (*code).module = module;
            }
        }
    }
    func
}

/// Allocate a `BuiltinCode` carrying an argument `Signature`.  The
/// signature is leaked to `'static` so the raw pointer stored on the
/// object has no Drop obligation, matching the `func`/`name` convention.
/// `fast_natural_arity` is `HOPELESS`: a builtin with a declared
/// signature takes the keyword-binding slow path, not the fixed-arity
/// fast path.
pub fn builtin_code_new_with_signature(
    name: &'static str,
    func: BuiltinCodeFn,
    docstring: Option<&'static str>,
    signature: Signature,
) -> PyObjectRef {
    let sig: *const Signature = Box::into_raw(Box::new(signature));
    builtin_code_new_full(name, func, docstring, HOPELESS, sig)
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

/// Whether two builtin implementations are the same function.
///
/// `std::ptr::fn_addr_eq` is bound on the `core::marker::FnPtr` lang item,
/// which Charon does not model — every call site it appears in resolves to
/// `Error during trait resolution` and leaves a hole in the extracted body.
/// Both operands are already the same fn-pointer type here, so their
/// addresses answer the question without the trait.
#[inline]
pub fn builtin_code_fn_eq(a: BuiltinCodeFn, b: BuiltinCodeFn) -> bool {
    a as usize == b as usize
}

/// Record the type whose namespace this code object was installed in, so its
/// receiver is checked before the implementation runs.
///
/// # Safety
/// `obj` must point to a valid `BuiltinCode`.
pub unsafe fn builtin_code_set_owner(obj: PyObjectRef, owner: &'static MethodOwner) {
    unsafe { (*(obj as *mut BuiltinCode)).owner = owner };
}

/// Invoke a `BuiltinCode` after applying the receiver and arity checks its
/// registration requires.  Every dispatch path — the interpreter's call paths,
/// the JIT's residual call, and the bound-method fast paths — goes through
/// here, so a call can never reach the implementation with an unchecked
/// receiver or a slice the implementation is not written for.
///
/// Roots nothing of its own: `args` is a native slice the collector does not
/// update, and the implementation it dispatches to runs Python.  Every caller
/// therefore has to publish `[obj, args...]` on the shadow stack and read the
/// slice back from it, so the set stays live for the whole body and the slice
/// is current at entry.  `dropvalues()` retires the frame slots that would
/// otherwise root a peeked argument, so a frame dispatch has to publish before
/// it drops, not after.
///
/// # Safety
/// `obj` must point to a valid `BuiltinCode`.
#[inline]
pub unsafe fn builtin_code_call(
    obj: PyObjectRef,
    args: &[PyObjectRef],
) -> Result<PyObjectRef, crate::PyError> {
    let code = obj as *const BuiltinCode;
    // The trailing marker dict is not an argument, so every check below reads
    // the positional slice: counting it as the receiver would report the
    // keyword dict as the object the descriptor was called on.
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    let receiver = positional.first().copied();
    let owner = unsafe { (*code).owner };
    if !owner.is_null() {
        let owner = unsafe { &*owner };
        let accepted = matches!(receiver, Some(receiver)
            if owner.is_instance.is_none_or(|is_instance| is_instance(receiver)));
        if !accepted {
            return Err(receiver_mismatch(
                owner,
                unsafe { (*code).name },
                positional,
            ));
        }
    }
    // eval.py:16-23 — a `fast_natural_arity` of 0..=4 is the exact positional
    // count the implementation was registered for; `HOPELESS`,
    // `PASSTHROUGHARGS1` and the `FLATPYCALL` bit all exceed 4.  A body with a
    // fixed arity indexes its slice directly, so a call that supplies a
    // different number of arguments is rejected before the body runs.  The
    // trailing marker dict occupies a slot of its own, so the keyword one
    // argument short of the arity leaves the slice at exactly the declared
    // length — the split has to happen before the count is read, not only once
    // a length already mismatched.
    let arity = unsafe { (*code).fast_natural_arity } as usize;
    if arity <= 4 {
        // A builtin registered by arity alone declares no parameter names
        // (`sig` is null), so it is positional-only and a keyword cannot name
        // anything it accepts.  Checked ahead of the count, which is the order
        // the messages come in (`','.join('a', 'b', x=1)` reports the keyword).
        if unsafe { (*code).sig.is_null() } && crate::builtins::has_real_kwargs(kwargs) {
            return Err(no_keyword_arguments(unsafe { &*code }, receiver));
        }
        if positional.len() != arity {
            return Err(arity_mismatch(
                unsafe { &*code },
                receiver,
                arity,
                positional.len(),
            ));
        }
    }
    unsafe { ((*code).func)(args) }
}

/// Wording for a keyword passed to a positional-only builtin.  A slot wrapper
/// reports itself as `wrapper NAME()`; everything else uses its qualified
/// name, the same split [`arity_mismatch`] draws.
#[cold]
#[inline(never)]
fn no_keyword_arguments(code: &BuiltinCode, receiver: Option<PyObjectRef>) -> crate::PyError {
    let (owner, qualname) = builtin_names(code, receiver);
    let name = code.name;
    let subject = match owner {
        Some(owner) if is_slot_wrapper(owner.type_name, name) => format!("wrapper {name}"),
        _ => qualname,
    };
    crate::PyError::type_error(format!("{subject}() takes no keyword arguments"))
}

/// The name a builtin reports itself under: a module-level builtin
/// `module.name` — bare for `builtins`, whose module prefix is left off — and
/// a descriptor `TYPE.name`.
///
/// TYPE is the receiver itself when the receiver IS a type: `int.mro` is a
/// bound `builtin_function_or_method` over a method of `type`, and a bound
/// builtin reports the qualname of the class it is bound to
/// (`int.mro() takes no arguments`).
///
/// Otherwise TYPE is the class that *declares* the method, which is what a
/// `method_descriptor` reports — `MyList([1]).append()` is `list.append()`,
/// naming `list` and not the receiver's own class.  Binding a method to a
/// name first (`f = MyList([1]).append`) produces the other callable kind and
/// reports `MyList.append()`; pyre hands out one callable kind and so takes
/// the declaring class, the form both call syntaxes reach.
fn builtin_names(
    code: &BuiltinCode,
    receiver: Option<PyObjectRef>,
) -> (Option<&'static MethodOwner>, String) {
    let owner = unsafe { code.owner.as_ref() };
    let qualname = match owner {
        None if code.module.is_empty() || code.module == "builtins" => code.name.to_string(),
        None => format!("{}.{}", code.module, code.name),
        Some(owner) => {
            let ty = match receiver {
                Some(r) if unsafe { pyre_object::typeobject::is_type(r) } => unsafe {
                    pyre_object::w_type_get_name(r)
                },
                _ => owner.type_name,
            };
            format!("{ty}.{}", code.name)
        }
    };
    (owner, qualname)
}

/// Wording for a call whose positional count does not match the
/// implementation's.  A builtin is reported under the convention its C
/// counterpart is written in, which for a fixed-arity implementation follows
/// from the argument count alone once the receiver is discounted:
///
/// - two or more arguments is argument-clinic's `_PyArg_CheckPositional`:
///   `NAME expected N arguments, got M`, under the BARE name — this is the
///   form for slot wrappers too (`__setitem__ expected 2 arguments, got 0`);
/// - one argument on a slot wrapper drops the name entirely
///   (`expected 1 argument, got 2`), because the wrapper stands in front of
///   every type's slot;
/// - otherwise no arguments is `METH_NOARGS`
///   (`NAME() takes no arguments (M given)`) and exactly one is `METH_O`
///   (`NAME() takes exactly one argument (M given)`).
///
/// The `at least` / `at most` variants belong to builtins with optional
/// arguments; those carry no fixed arity, so they check their own counts and
/// never reach here.
#[cold]
#[inline(never)]
fn arity_mismatch(
    code: &BuiltinCode,
    receiver: Option<PyObjectRef>,
    expected: usize,
    given: usize,
) -> crate::PyError {
    let (owner, qualname) = builtin_names(code, receiver);
    // A descriptor's slice leads with the receiver; every wording below counts
    // the arguments after it.
    let (declared, supplied) = match owner {
        Some(_) => (expected.saturating_sub(1), given.saturating_sub(1)),
        None => (expected, given),
    };
    let name = code.name;
    let slot = owner.is_some_and(|owner| is_slot_wrapper(owner.type_name, name));
    let message = if declared >= 2 {
        format!("{name} expected {declared} arguments, got {supplied}")
    } else if slot {
        format!(
            "expected {declared} argument{}, got {supplied}",
            if declared == 1 { "" } else { "s" }
        )
    } else if declared == 0 {
        format!("{qualname}() takes no arguments ({supplied} given)")
    } else {
        format!("{qualname}() takes exactly one argument ({supplied} given)")
    };
    crate::PyError::type_error(message)
}

/// Build the keyword-rejection error for a fixed-count BuiltinCode, for a
/// caller that rejects the keyword before it reaches [`builtin_code_call`].
/// `receiver` is the call's first positional argument, if any.
///
/// # Safety
/// `obj` must point to a valid `BuiltinCode`.
#[inline]
pub unsafe fn builtin_code_no_keyword_arguments(
    obj: PyObjectRef,
    receiver: Option<PyObjectRef>,
) -> crate::PyError {
    unsafe { no_keyword_arguments(&*(obj as *const BuiltinCode), receiver) }
}

/// Python 3.14 fills a type's slots with `wrapper_descriptor`s and its
/// `tp_methods` with `method_descriptor`s, and the two kinds word their
/// receiver errors differently, so only the slot names need listing — every
/// other name is an ordinary method.  `__contains__` and `__getitem__` are
/// slots everywhere except on the types that expose them through `tp_methods`.
pub(crate) fn is_slot_wrapper(type_name: &str, name: &str) -> bool {
    match name {
        "__contains__" => !matches!(type_name, "dict" | "set" | "frozenset" | "FrameLocalsProxy"),
        "__getitem__" => !matches!(type_name, "dict" | "list" | "FrameLocalsProxy"),
        _ => matches!(
            name,
            "__abs__"
                | "__add__"
                | "__aiter__"
                | "__and__"
                | "__anext__"
                | "__await__"
                | "__bool__"
                | "__buffer__"
                | "__call__"
                | "__del__"
                | "__delattr__"
                | "__delete__"
                | "__delitem__"
                | "__divmod__"
                | "__eq__"
                | "__float__"
                | "__floordiv__"
                | "__ge__"
                | "__get__"
                | "__getattribute__"
                | "__gt__"
                | "__hash__"
                | "__iadd__"
                | "__iand__"
                | "__ifloordiv__"
                | "__ilshift__"
                | "__imatmul__"
                | "__imod__"
                | "__imul__"
                | "__index__"
                | "__init__"
                | "__int__"
                | "__invert__"
                | "__ior__"
                | "__ipow__"
                | "__irshift__"
                | "__isub__"
                | "__iter__"
                | "__itruediv__"
                | "__ixor__"
                | "__le__"
                | "__len__"
                | "__lshift__"
                | "__lt__"
                | "__matmul__"
                | "__mod__"
                | "__mul__"
                | "__ne__"
                | "__neg__"
                | "__next__"
                | "__or__"
                | "__pos__"
                | "__pow__"
                | "__radd__"
                | "__rand__"
                | "__rdivmod__"
                | "__release_buffer__"
                | "__repr__"
                | "__rfloordiv__"
                | "__rlshift__"
                | "__rmatmul__"
                | "__rmod__"
                | "__rmul__"
                | "__ror__"
                | "__rpow__"
                | "__rrshift__"
                | "__rshift__"
                | "__rsub__"
                | "__rtruediv__"
                | "__rxor__"
                | "__set__"
                | "__setattr__"
                | "__setitem__"
                | "__str__"
                | "__sub__"
                | "__truediv__"
                | "__xor__"
        ),
    }
}

/// Build the TypeError an unbound descriptor raises for a missing or
/// foreign receiver.  Cold: it runs only on the failing call.
#[cold]
#[inline(never)]
fn receiver_mismatch(owner: &MethodOwner, name: &str, args: &[PyObjectRef]) -> crate::PyError {
    let ty = owner.type_name;
    let slot_wrapper = is_slot_wrapper(ty, name);
    let message = match args.first() {
        None if slot_wrapper => format!("descriptor '{name}' of '{ty}' object needs an argument"),
        None => format!("unbound method {ty}.{name}() needs an argument"),
        Some(&receiver) => {
            let received = crate::baseobjspace::object_functionstr_type_name(receiver);
            if slot_wrapper {
                format!("descriptor '{name}' requires a '{ty}' object but received a '{received}'")
            } else {
                format!(
                    "descriptor '{name}' for '{ty}' objects doesn't apply to a '{received}' object"
                )
            }
        }
    };
    crate::PyError::type_error(message)
}

/// eval.py:16-23 — read `fast_natural_arity` from a BuiltinCode.
///
/// # Safety
/// `obj` must point to a valid `BuiltinCode`.
#[inline]
pub unsafe fn builtin_code_get_fast_natural_arity(obj: PyObjectRef) -> u16 {
    unsafe { (*(obj as *const BuiltinCode)).fast_natural_arity }
}

/// Read the declared argument `Signature` from a BuiltinCode, or `None`
/// when the builtin is positional-only (`sig` is null — every builtin
/// today).  The pointee is leaked to `'static`, so the borrow is sound.
///
/// # Safety
/// `obj` must point to a valid `BuiltinCode`.
#[inline]
pub unsafe fn builtin_code_get_signature(obj: PyObjectRef) -> Option<&'static Signature> {
    unsafe { (*(obj as *const BuiltinCode)).sig.as_ref() }
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

/// gateway.py:777 BuiltinCode.getdocstring — return the stored docstring
/// wrapped as a `str`, or `None` if no docstring was attached.
///
/// # Safety
/// `obj` must point to a valid `BuiltinCode`.
#[inline]
pub unsafe fn builtin_code_get_docstring(obj: PyObjectRef) -> PyObjectRef {
    let func_obj = obj as *const BuiltinCode;
    match unsafe { (*func_obj).docstring } {
        Some(s) => pyre_object::w_str_new(s),
        None => pyre_object::w_none(),
    }
}

/// gateway.py GatewayCache.build() parity — wrap a BuiltinCodeFn as FunctionWithFixedCode.
///
/// Creates a BuiltinCode (Code object) and wraps it in a Function with
/// `can_change_code = false`, matching PyPy's:
///   `fn = FunctionWithFixedCode(space, code, None, defs, forcename=gateway.name)`
pub fn make_builtin_function(name: &'static str, func: BuiltinCodeFn) -> PyObjectRef {
    let code = builtin_code_new(name, func);
    crate::function_new_with_fixed_code(code as *const (), name.to_string(), pyre_object::PY_NULL)
}

/// `make_builtin_function` carrying the interpreter-level function's
/// docstring.  PyPy's `interp2app` reads this from `func.__doc__`; Rust
/// functions have no app-visible function object, so typedef registrations
/// pass the upstream literal explicitly.
pub fn make_builtin_function_with_doc(
    name: &'static str,
    func: BuiltinCodeFn,
    docstring: &'static str,
) -> PyObjectRef {
    let code = builtin_code_new_with_doc(name, func, Some(docstring));
    crate::function_new_with_fixed_code(code as *const (), name.to_string(), pyre_object::PY_NULL)
}

/// GatewayCache.build parity for an interp2app carrying the text signature
/// produced by `interp2app._generate_text_signature`.
pub fn make_builtin_function_with_text_signature(
    name: &'static str,
    func: BuiltinCodeFn,
    text_signature: &'static str,
) -> PyObjectRef {
    let function = make_builtin_function(name, func);
    unsafe {
        crate::function::fset_func_text_signature(function, pyre_object::w_str_new(text_signature));
    }
    function
}

/// `make_builtin_function_with_text_signature` that also records an argument
/// `Signature` (`Some` routes through the keyword-binding constructor; `None`
/// falls back to the positional-only one).  Used by the `#[pyre_methods]`
/// all-required instance-method arm, which wants both the generated
/// `__text_signature__` for introspection and by-name keyword binding.
pub fn make_builtin_function_with_text_signature_and_sig(
    name: &'static str,
    func: BuiltinCodeFn,
    text_signature: &'static str,
    signature: Option<Signature>,
) -> PyObjectRef {
    let function = make_builtin_function_maybe_sig(name, func, signature);
    unsafe {
        crate::function::fset_func_text_signature(function, pyre_object::w_str_new(text_signature));
    }
    function
}

/// Fixed-arity twin of `make_builtin_function_with_text_signature`.
pub fn make_builtin_function_with_arity_and_text_signature(
    name: &'static str,
    func: BuiltinCodeFn,
    arity: u16,
    text_signature: &'static str,
) -> PyObjectRef {
    let function = make_builtin_function_with_arity(name, func, arity);
    unsafe {
        crate::function::fset_func_text_signature(function, pyre_object::w_str_new(text_signature));
    }
    function
}

/// Like `make_builtin_function` but tagged as `BuiltinFunction`
/// (the `builtin_function` type) rather than `FunctionWithFixedCode`.
/// Used for builtin `__new__` carriers so `type(int.__new__)` is the
/// builtin-function type — distinct from a user `def`'s `function` — so
/// `copyreg._reduce_ex`'s `isinstance(new, type(int.__new__))` matches
/// only C-level `tp_new` wrappers, mirroring the
/// `builtin_function_or_method` type.
pub fn make_builtin_function_as_builtin(name: &'static str, func: BuiltinCodeFn) -> PyObjectRef {
    let code = builtin_code_new(name, func);
    crate::function_new_builtin(code as *const (), name.to_string(), pyre_object::PY_NULL)
}

/// Signature-aware [`make_builtin_function_as_builtin`].  Builtin `__new__`
/// descriptors need the builtin-function carrier for `copyreg` parity while
/// still routing keyword-only arguments through the gateway binder.
pub fn make_builtin_function_as_builtin_with_signature(
    name: &'static str,
    func: BuiltinCodeFn,
    signature: Signature,
) -> PyObjectRef {
    let code = builtin_code_new_with_signature(name, func, None, signature);
    crate::function_new_builtin(code as *const (), name.to_string(), pyre_object::PY_NULL)
}

/// `make_builtin_function` for a builtin with a declared argument
/// `Signature`, so the call path binds keyword arguments into positional
/// order before the function runs (see `call::bind_kwargs_to_signature`).
pub fn make_builtin_function_with_signature(
    name: &'static str,
    func: BuiltinCodeFn,
    signature: Signature,
) -> PyObjectRef {
    let code = builtin_code_new_with_signature(name, func, None, signature);
    crate::function_new_with_fixed_code(code as *const (), name.to_string(), pyre_object::PY_NULL)
}

/// `make_builtin_function`, optionally carrying a `Signature`.  `Some`
/// routes through `make_builtin_function_with_signature` (keyword-aware
/// binding); `None` falls back to the positional-only constructor.  Used
/// by the `#[pyre_function]`-derived `<name>_pyre_sig()` companion so a
/// builtin that declares keyword / kw-only parameters binds them by name.
pub fn make_builtin_function_maybe_sig(
    name: &'static str,
    func: BuiltinCodeFn,
    signature: Option<Signature>,
) -> PyObjectRef {
    match signature {
        Some(signature) => make_builtin_function_with_signature(name, func, signature),
        None => make_builtin_function(name, func),
    }
}

/// `make_builtin_function` recording both a fixed `fast_natural_arity`
/// and an optional argument `Signature`.  The arity preserves the fast
/// positional dispatch path; the `Some` signature additionally lets the
/// keyword-call path bind arguments by name.  Used by the
/// `inline_functions:` arm so a `#[pyre_function]` builtin keeps its
/// derived arity while gaining keyword / kw-only binding.
///
/// A `*args` / `**kwargs` / keyword-only parameter makes the passed
/// fixed `arity` meaningless (the raw param count over-counts the
/// variadic slots and the kw-only tail cannot be filled positionally),
/// so any such signature is demoted to `HOPELESS` — the call always
/// takes the keyword-binding slow path.
pub fn make_builtin_function_with_arity_and_maybe_sig(
    name: &'static str,
    func: BuiltinCodeFn,
    arity: u16,
    signature: Option<Signature>,
) -> PyObjectRef {
    let arity = match &signature {
        Some(s) if s.has_vararg() || s.has_kwarg() || s.num_kwonlyargnames() > 0 => HOPELESS,
        _ => arity,
    };
    let sig: *const Signature = match signature {
        Some(signature) => Box::into_raw(Box::new(signature)),
        None => std::ptr::null(),
    };
    let code = builtin_code_new_full(name, func, None, arity, sig);
    crate::function_new_with_fixed_code(code as *const (), name.to_string(), pyre_object::PY_NULL)
}

/// `gateway.py BuiltinCode.funcrun` reaches the unwrapped body through
/// `Arguments.parse_obj`, which rejects a call whose positional count does not
/// fit the signature.  A builtin registered with a declared count carries no
/// `Signature` to parse against and its body indexes those slots directly, so
/// the positional count is checked before the body runs.
pub fn check_declared_arity(name: &str, arity: usize, given: usize) -> Result<(), crate::PyError> {
    if given == arity {
        return Ok(());
    }
    let message = match arity {
        0 => format!("{name}() takes no arguments ({given} given)"),
        1 => format!("{name}() takes exactly one argument ({given} given)"),
        n => format!("{name} expected {n} arguments, got {given}"),
    };
    Err(crate::PyError::type_error(message))
}

/// Check the declared arity as positional-only. Keyword calls carry a trailing
/// marker dict in the raw builtin slice; the callee still receives that raw
/// slice after the arity guard.
pub fn check_declared_positional_arity(
    name: &str,
    arity: usize,
    args: &[PyObjectRef],
) -> Result<(), crate::PyError> {
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    if crate::builtins::has_real_kwargs(kwargs) {
        return Err(crate::PyError::type_error(format!(
            "{name}() takes no keyword arguments"
        )));
    }
    check_declared_arity(name, arity, positional.len())
}

/// `make_builtin_function` with known fixed arity for fast-path dispatch.
pub fn make_builtin_function_with_arity(
    name: &'static str,
    func: BuiltinCodeFn,
    arity: u16,
) -> PyObjectRef {
    let code = builtin_code_new_with_arity(name, func, arity);
    crate::function_new_with_fixed_code(code as *const (), name.to_string(), pyre_object::PY_NULL)
}

/// Fixed-arity builtin carrying the app-visible docstring that PyPy normally
/// obtains from the wrapped interpreter-level function's `__doc__`.
pub fn make_builtin_function_with_arity_and_doc(
    name: &'static str,
    func: BuiltinCodeFn,
    arity: u16,
    docstring: &'static str,
) -> PyObjectRef {
    debug_assert!(arity <= 4);
    let code = builtin_code_new_full(name, func, Some(docstring), arity, std::ptr::null());
    crate::function_new_with_fixed_code(code as *const (), name.to_string(), pyre_object::PY_NULL)
}

/// Build a CPython-compatible slot-wrapper descriptor with a fixed arity.
pub fn make_slot_wrapper_with_arity(
    name: &'static str,
    func: BuiltinCodeFn,
    arity: u16,
) -> PyObjectRef {
    let code = builtin_code_new_with_arity(name, func, arity);
    crate::function_new_slot_wrapper(code as *const (), name.to_string())
}

/// Build a CPython-compatible variadic slot-wrapper descriptor.
pub fn make_slot_wrapper(name: &'static str, func: BuiltinCodeFn) -> PyObjectRef {
    let code = builtin_code_new(name, func);
    crate::function_new_slot_wrapper(code as *const (), name.to_string())
}

/// Build a CPython-compatible ordinary method descriptor with fixed arity.
pub fn make_method_descriptor_with_arity(
    name: &'static str,
    func: BuiltinCodeFn,
    arity: u16,
) -> PyObjectRef {
    let code = builtin_code_new_with_arity(name, func, arity);
    crate::function_new_method_descriptor(code as *const (), name.to_string())
}

/// `make_builtin_function` with `fast_natural_arity = PASSTHROUGHARGS1` —
/// PyPy `BuiltinCodePassThroughArguments1` registration shape.
pub fn make_builtin_function_passthrough_args1(
    name: &'static str,
    func: BuiltinCodeFn,
) -> PyObjectRef {
    let code = builtin_code_new_passthrough_args1(name, func);
    crate::function_new_with_fixed_code(code as *const (), name.to_string(), pyre_object::PY_NULL)
}

/// mixedmodule.py:116 parity — wrap a BuiltinCodeFn as BuiltinFunction.
///
/// Module-level builtins are not descriptors: storing them on a user class
/// must not synthesize a bound method.
pub fn make_module_builtin_function(name: &'static str, func: BuiltinCodeFn) -> PyObjectRef {
    let code = builtin_code_new(name, func);
    crate::function_new_builtin(code as *const (), name.to_string(), pyre_object::PY_NULL)
}

/// `make_module_builtin_function` carrying PyPy `BuiltinCode.docstring`.
///
/// PyPy's gateway derives this from the wrapped interpreter function's
/// `__doc__`; Rust functions have no runtime doc attribute, so line-by-line
/// ports pass the upstream literal at registration.
pub fn make_module_builtin_function_with_doc(
    name: &'static str,
    func: BuiltinCodeFn,
    docstring: &'static str,
) -> PyObjectRef {
    let code = builtin_code_new_with_doc(name, func, Some(docstring));
    crate::function_new_builtin(code as *const (), name.to_string(), pyre_object::PY_NULL)
}

/// `make_module_builtin_function` with known fixed arity for fast-path dispatch.
pub fn make_module_builtin_function_with_arity(
    name: &'static str,
    func: BuiltinCodeFn,
    arity: u16,
) -> PyObjectRef {
    let code = builtin_code_new_with_arity(name, func, arity);
    crate::function_new_builtin(code as *const (), name.to_string(), pyre_object::PY_NULL)
}

/// `make_module_builtin_function_with_arity` additionally carrying an argument
/// `Signature` so the keyword-call path binds arguments by name at the call
/// site — the builtin then receives only positional slots and never the
/// `__pyre_kw__` marker dict. A `*args`/`**kwargs`/kw-only signature is demoted
/// to `HOPELESS` (as in `make_builtin_function_with_arity_and_maybe_sig`).
pub fn make_module_builtin_function_with_arity_and_sig(
    name: &'static str,
    func: BuiltinCodeFn,
    arity: u16,
    signature: Signature,
) -> PyObjectRef {
    let arity =
        if signature.has_vararg() || signature.has_kwarg() || signature.num_kwonlyargnames() > 0 {
            HOPELESS
        } else {
            arity
        };
    let sig: *const Signature = Box::into_raw(Box::new(signature));
    let code = builtin_code_new_full(name, func, None, arity, sig);
    crate::function_new_builtin(code as *const (), name.to_string(), pyre_object::PY_NULL)
}

/// Non-binding (`BuiltinFunction`) twin of
/// `make_builtin_function_with_arity_and_maybe_sig`, used by the
/// `py_module!` `inline_functions:` arm.  A module-level `#[pyre_function]`
/// builtin must be a `BuiltinFunction` (typedef omits `__get__`) rather than
/// a `FunctionWithFixedCode`: `mixedmodule.py:_load_lazily` gives every
/// function directly in a mixed-module a builtin type without `__get__`, so
/// storing one on a user class does not synthesize a bound method.  `None`
/// keeps the positional-only fast path; `Some` records the argument
/// `Signature` for keyword binding and demotes a `*args`/`**kwargs`/kw-only
/// signature to `HOPELESS`.
pub fn make_module_builtin_function_with_arity_and_maybe_sig(
    name: &'static str,
    func: BuiltinCodeFn,
    arity: u16,
    signature: Option<Signature>,
) -> PyObjectRef {
    let arity = match &signature {
        Some(s) if s.has_vararg() || s.has_kwarg() || s.num_kwonlyargnames() > 0 => HOPELESS,
        _ => arity,
    };
    let sig: *const Signature = match signature {
        Some(signature) => Box::into_raw(Box::new(signature)),
        None => std::ptr::null(),
    };
    let code = builtin_code_new_full(name, func, None, arity, sig);
    crate::function_new_builtin(code as *const (), name.to_string(), pyre_object::PY_NULL)
}

// ── fsencode_bytes_w ─────────────────────────────────────────────────
/// `baseobjspace.py:1970 fsencode_w` accepts str, bytes, or an object
/// implementing `__fspath__`, and answers with the filesystem-encoded bytes.
///
/// Bytes, not text: a path byte with no UTF-8 spelling has to reach the syscall
/// as itself. Folding it into a Rust `String` first replaces it with U+FFFD,
/// which both loses the name and makes distinct names alias onto one another.
pub fn fsencode_bytes_w(obj: pyre_object::PyObjectRef) -> Result<Vec<u8>, crate::PyError> {
    Ok(fsencode_path_w(obj)?.as_bytes)
}

/// `objspace.py:438 newfilename(s) = fsdecode(newbytes(s))`: expose the
/// application-level filename spelling derived from its filesystem bytes.
pub fn fsdecode_filename_bytes(data: &[u8]) -> pyre_object::PyObjectRef {
    pyre_object::w_str_from_wtf8_managed(fsdecode_filename_wtf8(data))
}

/// [`fsdecode_filename_bytes`]'s buffer, for a caller that assembles the
/// filename into a larger piece of text rather than handing it out as an
/// object. A Rust `String` cannot hold the lone surrogate a non-UTF-8 path byte
/// decodes to, so that text has to be built as WTF-8 throughout.
pub fn fsdecode_filename_wtf8(data: &[u8]) -> rustpython_wtf8::Wtf8Buf {
    crate::typedef::fsdecode_wtf8_total(data)
}

/// [`fsdecode_filename_bytes`] where the caller can still report a failure:
/// the point a filename first crosses from bytes into text. Under
/// `surrogatepass` a byte with no UTF-8 spelling has nowhere to go, and the
/// call that supplied it is the place that says so.
pub fn fsdecode_filename_checked(data: &[u8]) -> Result<(), crate::PyError> {
    crate::typedef::fsdecode_wtf8(data).map(|_| ())
}

/// The application-level spelling of a string the host handed us, for a caller
/// holding an `OsString` rather than the bytes behind it — a command-line
/// argument, where `targetpypystandalone.py:76-80` builds `sys.argv` out of
/// `space.newfilename` for exactly this reason.
///
/// The two arms are the host's two spellings, not a portability shim. Where the
/// argument is bytes it takes the filesystem decode, so a byte with no UTF-8
/// form comes back as the surrogate escape that re-encodes to itself. Where it
/// is UTF-16 the host already has the code units, and `from_wide` carries them
/// across losslessly; routing those through the byte decode instead would turn
/// an unpaired surrogate into three escapes and stop it round-tripping.
pub fn fsdecode_os_str(name: &std::ffi::OsStr) -> pyre_object::PyObjectRef {
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        let units: Vec<u16> = name.encode_wide().collect();
        pyre_object::w_str_from_wtf8(rustpython_wtf8::Wtf8Buf::from_wide(&units))
    }
    #[cfg(not(windows))]
    {
        fsdecode_filename_bytes(name.as_encoded_bytes())
    }
}

/// [`fsdecode_os_str`]'s buffer, for a caller that needs the spelling as text
/// rather than as an object — a `dict` key it has to hash, say. Pairs with
/// [`fsdecode_os_str`] the way [`fsdecode_filename_wtf8`] pairs with
/// [`fsdecode_filename_bytes`], and for the same reason: a Rust `String`
/// cannot hold the lone surrogate an undecodable byte becomes.
pub fn fsdecode_os_str_wtf8(name: &std::ffi::OsStr) -> rustpython_wtf8::Wtf8Buf {
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        let units: Vec<u16> = name.encode_wide().collect();
        rustpython_wtf8::Wtf8Buf::from_wide(&units)
    }
    #[cfg(not(windows))]
    {
        fsdecode_filename_wtf8(name.as_encoded_bytes())
    }
}

/// The filesystem bytes behind a host `OsStr` — [`fsdecode_os_str_wtf8`]'s
/// inverse, for a caller that has to retain the name as bytes rather than as
/// text: `co_filename`'s stored spelling, which `pycode.py:431
/// filename='fsencode'` names in the same units the syscall took.
///
/// Total, unlike [`fsencode`]: the host handed us this name, so it is already
/// in the filesystem's own units and there is nothing left to reject.
pub fn fsencode_os_str(name: &std::ffi::OsStr) -> Vec<u8> {
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        let units: Vec<u16> = name.encode_wide().collect();
        // `FS_ERRORS` is `surrogatepass` here, so a string's own WTF-8
        // spelling is its filesystem encoding.
        rustpython_wtf8::Wtf8Buf::from_wide(&units)
            .as_bytes()
            .to_vec()
    }
    #[cfg(not(windows))]
    {
        name.as_encoded_bytes().to_vec()
    }
}

/// [`fsencode_os_str`]'s other direction: the host name filesystem bytes
/// spell, for a caller that has to hand them to an API taking an `OsStr`.
pub fn os_string_from_fs_bytes(data: &[u8]) -> std::ffi::OsString {
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStringExt;
        let units: Vec<u16> = fsdecode_filename_wtf8(data).encode_wide().collect();
        std::ffi::OsString::from_wide(&units)
    }
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStringExt;
        std::ffi::OsString::from_vec(data.to_vec())
    }
    #[cfg(not(any(unix, windows)))]
    {
        // No byte spelling on this platform, so the name can only be carried
        // as the best text representation of these bytes.
        std::ffi::OsString::from(String::from_utf8_lossy(data).into_owned())
    }
}

/// `interp_posix.py:140-152 Path`: the syscall spelling and the resolved path
/// object travel together. For `os.PathLike`, `w_path` is the result of the
/// single `__fspath__` call, not the wrapper that supplied it.
pub struct FsEncodedPath {
    /// `Path.as_fd`, `-1` where the argument named a path rather than an open
    /// descriptor. Only the entry points that pass `allow_fd` can set it, so a
    /// caller that took a path-only boundary never has to test it.
    pub as_fd: i32,
    pub as_bytes: Vec<u8>,
    w_path_slot: usize,
    _roots: pyre_object::gc_roots::RootScope,
}

impl FsEncodedPath {
    pub fn w_path(&self) -> pyre_object::PyObjectRef {
        pyre_object::gc_roots::shadow_stack_get(self.w_path_slot)
    }

    /// # Safety
    /// The caller must uphold every validity, runtime-type, aliasing, and lifetime
    /// invariant required by the object and pointer arguments for the entire call.
    pub unsafe fn is_bytes(&self) -> bool {
        unsafe { pyre_object::bytesobject::is_bytes(self.w_path()) }
    }
}

/// The caller-less conversion, which names neither a function nor an argument.
/// It is what `os.fspath`, `os.fsencode` and the builtin `open` report, and what
/// a boundary spelling its path some other way than `path_converter` — Windows'
/// `os.system`, whose argument is text rather than a path — falls back to.
pub fn fsencode_path_w(obj: pyre_object::PyObjectRef) -> Result<FsEncodedPath, crate::PyError> {
    path_or_fd_w(obj, None, false, false)
}

/// [`fsencode_path_w`] for a path-only boundary that names itself. The argument
/// name is the second half: `path_converter` fills `function_name` and
/// `argument_name` from the argument clinic, so `link` rejects its first
/// argument as `src` and its second as `dst` rather than calling both `path`.
pub fn fsencode_path_named_w(
    obj: pyre_object::PyObjectRef,
    funcname: &str,
    argname: &str,
) -> Result<FsEncodedPath, crate::PyError> {
    path_or_fd_w(obj, Some((funcname, argname)), false, false)
}

/// [`fsencode_path_w`] for a boundary that also takes an open file descriptor —
/// `interp_posix.py:611 path=path_or_fd(allow_fd=True)`. `funcname` names the
/// caller in the type error, whose allowed-type list widens with `allow_fd`:
/// `stat` answers "string, bytes, os.PathLike or integer" where `lstat`, which
/// takes no descriptor, answers "string, bytes or os.PathLike".
pub fn fsencode_path_or_fd_w(
    obj: pyre_object::PyObjectRef,
    funcname: &str,
    allow_fd: bool,
) -> Result<FsEncodedPath, crate::PyError> {
    path_or_fd_w(obj, Some((funcname, "path")), allow_fd, false)
}

/// [`fsencode_path_or_fd_w`] for a boundary whose path argument also takes
/// `None` — `interp_scandir.py:20 path_or_fd(allow_fd=…, nullable=True)`, which
/// `listdir` and `scandir` declare. `None` itself is the caller's to resolve
/// (both spell it `"."` and report no filename); what the flag carries here is
/// the allowed-type list, which names `None` alongside the rest.
pub fn fsencode_path_or_fd_nullable_w(
    obj: pyre_object::PyObjectRef,
    funcname: &str,
    allow_fd: bool,
) -> Result<FsEncodedPath, crate::PyError> {
    path_or_fd_w(obj, Some((funcname, "path")), allow_fd, true)
}

/// `_PyType_Name` — the type's own name, with any module that qualifies it
/// dropped. The path boundaries report a rejected argument this way, where the
/// rest of the interpreter reports the qualified name (`array.array` becomes
/// `array` here and stays `array.array` in, say, a concatenation error).
pub(crate) fn short_type_name(obj: pyre_object::PyObjectRef) -> String {
    let name = crate::type_methods::arg_type_name(obj);
    match name.rfind('.') {
        Some(dot) => name[dot + 1..].to_string(),
        None => name,
    }
}

fn path_or_fd_w(
    obj: pyre_object::PyObjectRef,
    caller: Option<(&str, &str)>,
    allow_fd: bool,
    nullable: bool,
) -> Result<FsEncodedPath, crate::PyError> {
    // interp_posix.py:170-180 builds this list from the same two flags. The
    // caller pair is `path_converter`'s `function_name` and `argument_name`;
    // the entry points that convert on someone else's behalf carry neither.
    let allowed_types = match (nullable, allow_fd) {
        (true, true) => "string, bytes, os.PathLike, integer or None",
        (true, false) => "string, bytes, os.PathLike or None",
        (false, true) => "string, bytes, os.PathLike or integer",
        (false, false) => "string, bytes or os.PathLike",
    };
    let reject = |obj: pyre_object::PyObjectRef| -> crate::PyError {
        let tp = short_type_name(obj);
        match caller {
            Some((name, arg)) => crate::PyError::type_error(format!(
                "{name}: {arg} should be {allowed_types}, not {tp}"
            )),
            None => crate::PyError::type_error(format!(
                "expected str, bytes or os.PathLike object, not {tp}"
            )),
        }
    };
    let roots = pyre_object::gc_roots::push_roots();
    let obj_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(obj);

    let (data, w_path_slot, as_fd) = unsafe {
        let obj = pyre_object::gc_roots::shadow_stack_get(obj_slot);
        if nullable && pyre_object::is_none(obj) {
            // `_unwrap_path` answers the omitted argument itself, with the
            // directory it stands for (`interp_posix.py:181` `Path(-1, '.',
            // None, w_None)`), so no boundary has to spell that default again.
            // `None` is not bytes-like, so the names still come back as `str`.
            (b".".to_vec(), obj_slot, -1)
        } else if pyre_object::bytesobject::is_bytes(obj) {
            // Only `bytes` itself. `_unwrap_path`'s buffer arm
            // (`interp_posix.py:188-198`) takes any readable buffer and reports
            // it as deprecated; 3.14 completed that deprecation, so a
            // `bytearray` is now turned away by the same message every other
            // rejected type gets.
            (
                pyre_object::bytesobject::w_bytes_data(obj).to_vec(),
                obj_slot,
                -1,
            )
        } else if pyre_object::is_str(obj) {
            (fsencode(obj)?, obj_slot, -1)
        } else if allow_fd
            && (pyre_object::pyobject::is_int_or_long(obj)
                || crate::baseobjspace::lookup(obj, "__index__").is_some())
        {
            // interp_posix.py:201-210 — the descriptor case is probed with
            // `__index__` and sits BEFORE the PathLike case, so an object
            // carrying both is taken as a descriptor and its `__fspath__` is
            // never called.
            //
            // Where `:202-207` wraps the probe in `except OperationError:
            // pass` and falls through to `__fspath__`, 3.14 lets an
            // `__index__` that raises propagate — measured, an object with a
            // raising `__index__` and a working `__fspath__` reports that
            // exception rather than being statted. The parity suite's oracle
            // is CPython.
            //
            // A bool is an integer, so it reaches this arm and names
            // descriptor 0 or 1; the warning says so before it is used.
            if pyre_object::is_bool(obj) {
                crate::warn::warn_category(
                    "bool is used as a file descriptor",
                    "RuntimeWarning",
                    1,
                )?;
            }
            // The warning runs the installed filters, which are Python, so the
            // argument is read back off the shadow stack the same way the
            // `__fspath__` arm reads it after its call.
            let fd =
                crate::baseobjspace::c_int_w(pyre_object::gc_roots::shadow_stack_get(obj_slot))?;
            // interp_posix.py:269-271 `unwrap_fd` — `-1` is the sentinel for
            // "not a descriptor", so a caller naming it has to be turned away
            // here rather than silently read as a path.
            if fd == -1 {
                return Err(crate::PyError::os_error("invalid file descriptor: -1"));
            }
            (Vec::new(), obj_slot, fd)
        } else {
            // `type(path).__fspath__(path)` — the descriptor read off the type is
            // unbound, so `path` is supplied as the sole argument.
            let Some(fspath_fn) = crate::typedef::r#type(obj)
                .and_then(|pt| crate::baseobjspace::lookup_in_type(pt.as_ptr(), "__fspath__"))
            else {
                return Err(reject(obj));
            };
            let fspath_slot = pyre_object::gc_roots::shadow_stack_len();
            pyre_object::gc_roots::pin_root(fspath_fn);
            let result = crate::call::call_function_impl_result(
                pyre_object::gc_roots::shadow_stack_get(fspath_slot),
                &[pyre_object::gc_roots::shadow_stack_get(obj_slot)],
            )?;
            let result_slot = pyre_object::gc_roots::shadow_stack_len();
            pyre_object::gc_roots::pin_root(result);
            let result = pyre_object::gc_roots::shadow_stack_get(result_slot);
            // interp_posix.py:3049-3051 accepts only `str` or `bytes` back from
            // `__fspath__`; a `bytearray` is a readable buffer but not a path.
            if pyre_object::bytesobject::is_bytes(result) {
                (
                    pyre_object::bytesobject::w_bytes_data(result).to_vec(),
                    result_slot,
                    -1,
                )
            } else if pyre_object::is_str(result) {
                (fsencode(result)?, result_slot, -1)
            } else {
                // interp_posix.py:3053-3058 names both the object that was asked
                // and the type its `__fspath__` answered with.
                let obj = pyre_object::gc_roots::shadow_stack_get(obj_slot);
                return Err(crate::PyError::type_error(format!(
                    "expected {}.__fspath__() to return str or bytes, not {}",
                    short_type_name(obj),
                    short_type_name(result)
                )));
            }
        }
    };
    // baseobjspace.py:2016-2017 `bytesbuf0_w` rejects embedded nulls. A
    // descriptor carries no bytes to check.
    if as_fd == -1 && data.contains(&0) {
        return Err(crate::PyError::value_error("embedded null byte"));
    }
    // Where the filesystem encoding is `surrogatepass`, the byte spelling of a
    // path is UTF-8 and a byte that begins no sequence names nothing at all.
    // Report it here, at the call that supplied it: the host takes UTF-16, so
    // carrying such bytes further would mean inventing a spelling for them and
    // addressing some other file.
    #[cfg(windows)]
    fsdecode_filename_checked(&data)?;
    Ok(FsEncodedPath {
        as_fd,
        as_bytes: data,
        w_path_slot,
        _roots: roots,
    })
}

/// `baseobjspace.py:1962 space.fsencode` — the plain filesystem encode of a
/// `str`, with no argument conversion, `DeprecationWarning`, or embedded-null
/// rejection. The caller must already have established that `obj` is a `str`.
pub fn fsencode(obj: pyre_object::PyObjectRef) -> Result<Vec<u8>, crate::PyError> {
    let wtf8 = unsafe { pyre_object::w_str_get_wtf8(obj) };
    if crate::typedef::FS_ERRORS == "surrogatepass" {
        // The string's own WTF-8 spelling is the encoding: a surrogate keeps
        // its three bytes instead of folding to the one byte an escape names,
        // and nothing is out of range, so no `str` fails to encode.
        return Ok(wtf8.as_bytes().to_vec());
    }
    let mut out = Vec::with_capacity(wtf8.len());
    for (pos, cp) in wtf8.code_points().enumerate() {
        if let Some(ch) = cp.to_char() {
            let mut buf = [0; 4];
            out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
            continue;
        }
        let code = cp.to_u32();
        if (0xDC80..=0xDCFF).contains(&code) {
            out.push((code - 0xDC00) as u8);
        } else {
            return Err(crate::typedef::unicode_encode_error(
                "utf-8",
                obj,
                pos,
                pos + 1,
                "surrogates not allowed",
            ));
        }
    }
    Ok(out)
}

/// The host `PathBuf` a `str` names, taking the filesystem encoding.
///
/// A name that came back from `readdir` carries the surrogate escapes
/// [`fsdecode_filename_bytes`] produced, and those have no `&str` spelling —
/// building the path with `w_str_get_value` aborts the process. Route through
/// [`fsencode`] so the escapes fold back to the original bytes and the path
/// names the same file it was read from.
///
/// A str the filesystem encoding cannot spell raises rather than resolving to
/// some other path: under `surrogateescape`, `sys.path.insert(0, '\ud800')`
/// followed by an import answers `UnicodeEncodeError: surrogates not allowed`,
/// and dropping the entry instead would search on and report the wrong failure.
/// Under `surrogatepass` every `str` has a spelling, so the same entry reaches
/// the host as itself and names a file that is simply not there.
pub fn fspath_buf(obj: pyre_object::PyObjectRef) -> Result<std::path::PathBuf, crate::PyError> {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStringExt;
        let bytes = fsencode(obj)?;
        Ok(std::path::PathBuf::from(std::ffi::OsString::from_vec(
            bytes,
        )))
    }
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStringExt;
        // The host takes UTF-16, and an unpaired surrogate is a code unit it
        // carries: hand the name over as spelled. Going through `str_utf8_w`
        // would reject it, and a lossy re-decode would substitute U+FFFD,
        // which addresses a different name and aliases distinct entries.
        let units: Vec<u16> = unsafe { pyre_object::w_str_get_wtf8(obj) }
            .encode_wide()
            .collect();
        Ok(std::path::PathBuf::from(std::ffi::OsString::from_wide(
            &units,
        )))
    }
    #[cfg(not(any(unix, windows)))]
    {
        // No byte spelling on this platform, so the host API receives the text
        // itself and a surrogate has nowhere to go: `str_utf8_w` reports it as
        // an encoding error. Encoding to bytes and decoding them back lossily
        // would substitute U+FFFD, which addresses a different name and makes
        // distinct entries alias onto one another.
        Ok(std::path::PathBuf::from(crate::baseobjspace::str_utf8_w(
            obj,
        )?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Guard against drift between the constant colocated with
    /// `BuiltinCode` and the id that `pyre-jit/src/eval.rs` asserts at
    /// JitDriver init. Mirror of the W_INT/W_FLOAT trip-wire tests.
    #[test]
    fn builtin_code_gc_type_id_matches_descr() {
        assert_eq!(BUILTIN_CODE_GC_TYPE_ID, 13);
        assert_eq!(
            <BuiltinCode as pyre_object::lltype::GcType>::type_id(),
            BUILTIN_CODE_GC_TYPE_ID
        );
        assert_eq!(
            <BuiltinCode as pyre_object::lltype::GcType>::SIZE,
            std::mem::size_of::<BuiltinCode>()
        );
    }

    /// pypy/interpreter/signature.py:33-46 accessor parity:
    /// `def f(a, b, /, c, d, *args, e, f, **kwargs): ...`
    /// → argnames=[a,b,c,d,e,f], varargname=args, kwargname=kwargs,
    /// posonlyargcount=2, kwonlyargcount=2.
    #[test]
    fn signature_accessor_parity() {
        let sig = Signature::new(
            vec!["a", "b", "c", "d", "e", "f"],
            Some("args"),
            Some("kwargs"),
            2,
            2,
        );
        // num_argnames = len(argnames) - kwonlyargcount = 6 - 2 = 4
        assert_eq!(sig.num_argnames(), 4);
        assert_eq!(sig.num_posonlyargnames(), 2);
        assert_eq!(sig.num_kwonlyargnames(), 2);
        assert!(sig.has_vararg());
        assert!(sig.has_kwarg());
        // scope_length = len(argnames) + has_vararg + has_kwarg = 6 + 1 + 1 = 8
        assert_eq!(sig.scope_length(), 8);
        // find_argname returns -1 for unknown
        assert_eq!(sig.find_argname("a"), 0);
        assert_eq!(sig.find_argname("e"), 4);
        assert_eq!(sig.find_argname("missing"), -1);
        // getallvarnames appends varargname + kwargname
        assert_eq!(
            sig.getallvarnames(),
            vec!["a", "b", "c", "d", "e", "f", "args", "kwargs"],
        );
    }

    /// `def f(a, b): ...` — no *args / **kwargs / kwonly.
    #[test]
    fn signature_minimal_no_extras() {
        let sig = Signature::new(vec!["a", "b"], None, None, 0, 0);
        assert_eq!(sig.num_argnames(), 2);
        assert_eq!(sig.num_kwonlyargnames(), 0);
        assert!(!sig.has_vararg());
        assert!(!sig.has_kwarg());
        assert_eq!(sig.scope_length(), 2);
        assert_eq!(sig.getallvarnames(), vec!["a", "b"]);
    }
}
