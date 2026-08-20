//! The `PyCFunction` carrier -- PyPy `cpyext/methodobject.py`.
//!
//! Upstream's `W_PyCFunctionObject` is an interp-level class holding the
//! `PyMethodDef` pointer, the bound `self` and the defining module.  Pyre has
//! no typed-payload class for it yet, so the carrier is an instance of a native
//! type whose reserved dict keys hold the same three values — the shape
//! `_ctypes`'s `CFuncPtr` already uses for a native function pointer
//! (`module/_ctypes/funcptr.rs`).  Replacing it with a `#[pyre_class]` payload
//! is part of the C-defined-types slice, where a typed mirror has to exist
//! anyway.

use super::pyobject::{self, CPyObject};
use pyre_object::PyObjectRef;
use std::collections::HashMap;
use std::ffi::{CStr, c_char, c_int, c_void};
use std::hash::BuildHasherDefault;
use std::sync::OnceLock;

pub const METH_VARARGS: c_int = 0x0001;
pub const METH_KEYWORDS: c_int = 0x0002;
pub const METH_NOARGS: c_int = 0x0004;
pub const METH_O: c_int = 0x0008;
pub const METH_CLASS: c_int = 0x0010;
pub const METH_STATIC: c_int = 0x0020;
pub const METH_COEXIST: c_int = 0x0040;
pub const METH_FASTCALL: c_int = 0x0080;
pub const METH_METHOD: c_int = 0x0200;

/// One row of a `PyMethodDef` table.
#[repr(C)]
pub struct CPyMethodDef {
    pub ml_name: *const c_char,
    pub ml_meth: *const c_void,
    pub ml_flags: c_int,
    pub ml_doc: *const c_char,
}

/// Reserved carrier keys.  Namespaced so a C module cannot collide with them
/// through `PyModule_AddObject`.
const ML_KEY: &str = "__pyre_ml__";
const SELF_KEY: &str = "__self__";
const NAME_KEY: &str = "__name__";
const QUALNAME_KEY: &str = "__qualname__";
const DOC_KEY: &str = "__doc__";
const MODULE_KEY: &str = "__module__";
/// The defining class a `METH_METHOD` row is handed beside its receiver.
const CLASS_KEY: &str = "__pyre_mm_class__";

/// The carrier keys a store from Python must not reach.
///
/// Everything the carrier knows lives in its instance namespace, which is a
/// mapping like any other; a store into one of these would retarget the
/// receiver a C callable casts to its own struct, or name a different
/// `PyMethodDef` row for it to call.  `__module__` is left writable, which is
/// the one the reference type allows as well.
/// The `tp_methods` / `tp_getset` / `tp_members` descriptor carriers in
/// `typeobject` share this fence and these keys; `__pyre_def__` there holds a
/// definition's address outright, so a store into it is the same hole in a
/// straighter line.
const RESERVED_KEYS: &[&str] = &[
    ML_KEY,
    SELF_KEY,
    NAME_KEY,
    QUALNAME_KEY,
    DOC_KEY,
    CLASS_KEY,
    "__pyre_def__",
    "__objclass__",
];

/// Every `PyMethodDef` row a carrier may name.
///
/// The carrier records an index into this table rather than the definition's
/// address.  Its namespace is reachable from Python — `__dict__` is a mapping
/// like any other — so whatever the carrier holds is under the writer's
/// control; an index naming no row is refused, where an address would be
/// transmuted to a function pointer and called.
///
/// The definitions live in the loaded extension's static storage, which the
/// module cache keeps mapped, so the table only ever grows by one row per
/// distinct method row converted.
#[derive(Default)]
struct MethodDefTable {
    rows: Vec<usize>,
    index_of: HashMap<usize, usize, BuildHasherDefault<std::hash::DefaultHasher>>,
}

static METHOD_DEFS: super::ForkMutex<MethodDefTable> = super::ForkMutex::new(MethodDefTable {
    rows: Vec::new(),
    index_of: HashMap::with_hasher(BuildHasherDefault::new()),
});

pub(super) unsafe fn after_fork_child() {
    unsafe { METHOD_DEFS.reinit_after_fork() };
}

/// The index `method` is filed under, entering it if it is new.
fn intern_method_def(method: *mut CPyMethodDef) -> i64 {
    let address = method as usize;
    let mut table = METHOD_DEFS.lock();
    if let Some(&index) = table.index_of.get(&address) {
        return index as i64;
    }
    let index = table.rows.len();
    table.rows.push(address);
    table.index_of.insert(address, index);
    index as i64
}

/// The definition an index names, or `None` for one that names no row.
fn method_def_at(index: i64) -> Option<*mut CPyMethodDef> {
    let table = METHOD_DEFS.lock();
    let address = *table.rows.get(usize::try_from(index).ok()?)?;
    Some(address as *mut CPyMethodDef)
}

static PYCFUNCTION_TYPE_OBJ: OnceLock<usize> = OnceLock::new();
static PYCMETHOD_TYPE_OBJ: OnceLock<usize> = OnceLock::new();

/// `__setattr__` / `__delattr__` for a carrier type.
///
/// `value` is `None` for the delete spelling.
fn store_attribute(
    carrier: PyObjectRef,
    name: PyObjectRef,
    value: Option<PyObjectRef>,
) -> Result<PyObjectRef, crate::PyError> {
    if !unsafe { pyre_object::unicodeobject::is_str(name) } {
        return Err(crate::PyError::type_error("attribute name must be string"));
    }
    let attribute = unsafe { pyre_object::w_str_get_wtf8(name) }.to_string();
    if attribute != MODULE_KEY {
        let type_name = unsafe { pyre_object::w_type_get_name((*carrier).w_class) };
        let message = if RESERVED_KEYS.contains(&attribute.as_str()) {
            format!("attribute '{attribute}' of '{type_name}' objects is not writable")
        } else {
            format!("'{type_name}' object has no attribute '{attribute}'")
        };
        return Err(crate::PyError::attribute_error(message));
    }
    let dict = crate::baseobjspace::getdict_native(carrier);
    if !dict.is_null() {
        match value {
            Some(value) => unsafe {
                pyre_object::dictmultiobject::w_dict_setitem_str(dict, MODULE_KEY, value)
            },
            None => {
                unsafe { pyre_object::dictmultiobject::w_dict_delitem_str(dict, MODULE_KEY) };
            }
        }
    }
    Ok(pyre_object::w_none())
}

fn descr_setattr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() < 3 {
        return Err(crate::PyError::type_error(
            "__setattr__ requires name and value",
        ));
    }
    store_attribute(args[0], args[1], Some(args[2]))
}

fn descr_delattr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Err(crate::PyError::type_error("__delattr__ requires a name"));
    }
    store_attribute(args[0], args[1], None)
}

/// Install the stores every carrier type refuses, so nothing but
/// `__module__` can be written into its namespace from Python.
pub(super) fn install_attribute_fence(ns: PyObjectRef) {
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__setattr__",
            crate::make_builtin_function_with_arity("__setattr__", descr_setattr, 3),
        );
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__delattr__",
            crate::make_builtin_function_with_arity("__delattr__", descr_delattr, 2),
        );
    }
}

/// The carrier type, named as upstream names its typedef.
///
/// This is what `PyCFunction_Type` names, and so what `PyCFunction_Check`
/// answers for.  An interpreter builtin such as `len` is a *different*
/// `builtin_function_or_method` and answers no: it carries no `PyMethodDef`,
/// so an entry point that agreed it was one would then have nothing to hand
/// back from `PyCFunction_GetFunction` but an error, which a caller spelling
/// the two as one expression does not look for.
pub fn pycfunction_type() -> PyObjectRef {
    *PYCFUNCTION_TYPE_OBJ.get_or_init(|| {
        let tp = crate::typedef::make_builtin_type("builtin_function_or_method", |ns| {
            unsafe {
                pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                    ns,
                    "__call__",
                    crate::make_builtin_function("__call__", descr_call),
                );
                pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                    ns,
                    "__repr__",
                    crate::make_builtin_function_with_arity("__repr__", descr_repr, 1),
                );
            };
            install_attribute_fence(ns);
        });
        unsafe { pyre_object::typeobject::w_type_set_hasdict(tp, true) };
        tp as usize
    }) as PyObjectRef
}

/// `methodobject.py:264 W_PyCMethodObject` — the carrier a `METH_METHOD` row
/// gets, which is everything `pycfunction_type` carries plus the class the
/// definition was declared in.
///
/// This is what `PyCMethod_Type` names, and it derives from
/// `PyCFunction_Type`, so `PyCFunction_Check` answers yes for one.
pub fn pycmethod_type() -> PyObjectRef {
    *PYCMETHOD_TYPE_OBJ.get_or_init(|| {
        let tp = crate::typedef::make_builtin_type_with_base(
            "builtin_method",
            |_ns| {},
            pycfunction_type(),
        );
        unsafe { pyre_object::typeobject::w_type_set_hasdict(tp, true) };
        tp as usize
    }) as PyObjectRef
}

/// Whether `carrier` is one of this module's own, rather than something that
/// merely has the reserved keys in its namespace.
fn is_carrier(carrier: PyObjectRef) -> bool {
    !carrier.is_null()
        && unsafe { crate::baseobjspace::issubtype_w((*carrier).w_class, pycfunction_type()) }
}

/// The `PyMethodDef` behind `object`, or NULL with a `SystemError` when
/// nothing is.
///
/// The type is checked before the namespace is read: `space.interp_w(
/// W_PyCFunctionObject, w_obj)` is what upstream's accessors open with, and
/// without it any object carrying a plausible `__pyre_ml__` would hand a
/// caller a C function pointer.
fn checked_method_def(object: *mut CPyObject) -> Option<*mut CPyMethodDef> {
    let object = unsafe { pyobject::from_ref(object) };
    let definition = is_carrier(object).then(|| method_def(object)).flatten();
    if definition.is_none() {
        super::pyerrors::set_pending_error(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "bad argument to internal function",
        ));
    }
    definition
}

/// `methodobject.py:563 PyCFunction_GetFunction(object)` — the C function the
/// carrier calls.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyCFunction_GetFunction(object: *mut CPyObject) -> *mut c_void {
    match checked_method_def(object) {
        Some(definition) => (unsafe { (*definition).ml_meth }) as *mut c_void,
        None => std::ptr::null_mut(),
    }
}

/// `PyCFunction_GetFlags(object)` — the `METH_*` set the definition carries.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyCFunction_GetFlags(object: *mut CPyObject) -> c_int {
    match checked_method_def(object) {
        Some(definition) => unsafe { (*definition).ml_flags },
        None => -1,
    }
}

/// `PyCFunction_GetSelf(object)` — the receiver the carrier was bound to,
/// borrowed, and NULL for a definition declared `METH_STATIC`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyCFunction_GetSelf(object: *mut CPyObject) -> *mut CPyObject {
    let Some(definition) = checked_method_def(object) else {
        return std::ptr::null_mut();
    };
    if unsafe { (*definition).ml_flags } & METH_STATIC != 0 {
        return std::ptr::null_mut();
    }
    let carrier = unsafe { pyobject::from_ref(object) };
    match carrier_get(carrier, SELF_KEY) {
        Some(receiver) => pyobject::borrow_from(object, receiver),
        None => std::ptr::null_mut(),
    }
}

/// `PyCFunction_New(ml, self)` — the callable a `PyMethodDef` row describes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyCFunction_New(
    method: *mut CPyMethodDef,
    receiver: *mut CPyObject,
) -> *mut CPyObject {
    unsafe { PyCFunction_NewEx(method, receiver, std::ptr::null_mut()) }
}

/// `PyCFunction_NewEx(ml, self, module)` — the same, naming the module the
/// definition came from.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyCFunction_NewEx(
    method: *mut CPyMethodDef,
    receiver: *mut CPyObject,
    module: *mut CPyObject,
) -> *mut CPyObject {
    if method.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    unsafe { PyCMethod_New(method, receiver, module, std::ptr::null_mut()) }
}

/// `methodobject.py:544 PyCMethod_New(ml, self, module, cls)` — the same,
/// naming the class a `METH_METHOD` definition was declared in.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyCMethod_New(
    method: *mut CPyMethodDef,
    receiver: *mut CPyObject,
    module: *mut CPyObject,
    class: *mut super::typeobject::CPyTypeObject,
) -> *mut CPyObject {
    if method.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    let flags = unsafe { (*method).ml_flags };
    let message = match (flags & METH_METHOD != 0, class.is_null()) {
        (true, true) => Some("attempting to create PyCMethod with a METH_METHOD flag but no class"),
        (false, false) => {
            Some("attempting to create PyCFunction with class but no METH_METHOD flag")
        }
        _ => None,
    };
    if let Some(message) = message {
        super::pyerrors::set_pending_error(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            message,
        ));
        return std::ptr::null_mut();
    }
    let class = class as *mut CPyObject;
    super::object::realize_all([receiver, module, class]);
    let receiver = unsafe { pyobject::from_ref(receiver) };
    let module = unsafe { pyobject::from_ref(module) };
    let class = unsafe { pyobject::from_ref(class) };
    if !class.is_null() && !unsafe { pyre_object::is_type(class) } {
        super::pyerrors::set_pending_error(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "bad argument to PyCMethod_New",
        ));
        return std::ptr::null_mut();
    }
    let module = if module.is_null() {
        pyre_object::w_none()
    } else {
        module
    };
    let class = (!class.is_null()).then_some(class);
    super::object::result(new_pycfunction_in_class(method, receiver, module, class))
}

/// `methodobject.py:155 PyCMethod_GetClass(op)` — the class a `METH_METHOD`
/// definition was declared in, borrowed, and NULL for one that is not.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyCMethod_GetClass(
    object: *mut CPyObject,
) -> *mut super::typeobject::CPyTypeObject {
    let Some(definition) = checked_method_def(object) else {
        return std::ptr::null_mut();
    };
    if unsafe { (*definition).ml_flags } & METH_METHOD == 0 {
        return std::ptr::null_mut();
    }
    let carrier = unsafe { pyobject::from_ref(object) };
    match carrier_get(carrier, CLASS_KEY) {
        Some(class) => {
            pyobject::borrow_from(object, class) as *mut super::typeobject::CPyTypeObject
        }
        None => std::ptr::null_mut(),
    }
}

fn carrier_get(carrier: PyObjectRef, key: &str) -> Option<PyObjectRef> {
    let dict = crate::baseobjspace::getdict_native(carrier);
    if dict.is_null() {
        return None;
    }
    unsafe { pyre_object::dictmultiobject::w_dict_getitem_str(dict, key) }
}

fn carrier_set(carrier: PyObjectRef, key: &str, value: PyObjectRef) {
    let dict = crate::baseobjspace::getdict_native(carrier);
    if !dict.is_null() {
        unsafe { pyre_object::dictmultiobject::w_dict_setitem_str(dict, key, value) };
    }
}

fn text_or_none(pointer: *const c_char) -> PyObjectRef {
    if pointer.is_null() {
        return pyre_object::w_none();
    }
    pyre_object::w_str_new(&unsafe { CStr::from_ptr(pointer) }.to_string_lossy())
}

fn text_or_empty(pointer: *const c_char) -> String {
    if pointer.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(pointer) }
        .to_string_lossy()
        .into_owned()
}

/// `methodobject.py:W_PyCFunctionObject.__init__`.
///
/// `w_self` is the bound receiver a module-level function is called with —
/// the module object itself — and `w_module` the `__module__` string.
pub fn new_pycfunction(
    method: *mut CPyMethodDef,
    w_self: PyObjectRef,
    w_module: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    new_pycfunction_in_class(method, w_self, w_module, None)
}

/// [`new_pycfunction`] naming the class a `METH_METHOD` definition was
/// declared in, whose carrier is `pycmethod_type` rather than
/// `pycfunction_type`.
pub fn new_pycfunction_in_class(
    method: *mut CPyMethodDef,
    w_self: PyObjectRef,
    w_module: PyObjectRef,
    w_class: Option<PyObjectRef>,
) -> Result<PyObjectRef, crate::PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let self_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(w_self);
    let module_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(w_module);
    let class_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(w_class.unwrap_or_else(pyre_object::w_none));
    let carrier_type = if w_class.is_some() {
        pycmethod_type()
    } else {
        pycfunction_type()
    };
    let carrier = pyre_object::w_instance_new(carrier_type);
    let carrier_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(carrier);

    let name = text_or_none(unsafe { (*method).ml_name });
    carrier_set(
        pyre_object::gc_roots::shadow_stack_get(carrier_slot),
        NAME_KEY,
        name,
    );
    let name = carrier_get(
        pyre_object::gc_roots::shadow_stack_get(carrier_slot),
        NAME_KEY,
    )
    .unwrap_or_else(pyre_object::w_none);
    carrier_set(
        pyre_object::gc_roots::shadow_stack_get(carrier_slot),
        QUALNAME_KEY,
        name,
    );
    let doc = text_or_none(unsafe { (*method).ml_doc });
    carrier_set(
        pyre_object::gc_roots::shadow_stack_get(carrier_slot),
        DOC_KEY,
        doc,
    );
    carrier_set(
        pyre_object::gc_roots::shadow_stack_get(carrier_slot),
        SELF_KEY,
        pyre_object::gc_roots::shadow_stack_get(self_slot),
    );
    carrier_set(
        pyre_object::gc_roots::shadow_stack_get(carrier_slot),
        MODULE_KEY,
        pyre_object::gc_roots::shadow_stack_get(module_slot),
    );
    let ml = pyre_object::w_int_new(intern_method_def(method));
    carrier_set(
        pyre_object::gc_roots::shadow_stack_get(carrier_slot),
        ML_KEY,
        ml,
    );
    if w_class.is_some() {
        carrier_set(
            pyre_object::gc_roots::shadow_stack_get(carrier_slot),
            CLASS_KEY,
            pyre_object::gc_roots::shadow_stack_get(class_slot),
        );
    }
    Ok(pyre_object::gc_roots::shadow_stack_get(carrier_slot))
}

fn method_def(carrier: PyObjectRef) -> Option<*mut CPyMethodDef> {
    let ml = carrier_get(carrier, ML_KEY)?;
    if !unsafe { pyre_object::is_int(ml) } {
        return None;
    }
    method_def_at(unsafe { pyre_object::w_int_get_value(ml) })
}

fn method_name(carrier: PyObjectRef) -> String {
    carrier_get(carrier, NAME_KEY)
        .filter(|&name| unsafe { pyre_object::unicodeobject::is_str(name) })
        .map(|name| unsafe { pyre_object::w_str_get_wtf8(name) }.to_string())
        .unwrap_or_else(|| "?".to_string())
}

fn descr_repr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let carrier = args[0];
    Ok(pyre_object::w_str_new(&format!(
        "<built-in function {}>",
        method_name(carrier)
    )))
}

/// `methodobject.py:W_PyCFunctionObject.descr_call`.
fn descr_call(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Err(crate::PyError::type_error("__call__ requires self"));
    }
    let carrier = args[0];
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(&args[1..]);
    let Some(method) = method_def(carrier) else {
        return Err(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "cpyext function carrier lost its method definition",
        ));
    };
    let Some(w_self) = carrier_get(carrier, SELF_KEY) else {
        return Err(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "cpyext function carrier lost its receiver",
        ));
    };
    call_method_def_in_class(
        method,
        w_self,
        carrier_get(carrier, CLASS_KEY),
        positional,
        kwargs,
    )
}

/// Validate one call against its `ml_flags` and hand it to the bridge.
///
/// Shared with the `tp_methods` descriptor, whose receiver is the instance the
/// attribute was read through rather than a bound `__self__`.
pub(super) fn call_method_def(
    method: *mut CPyMethodDef,
    w_self: PyObjectRef,
    positional: &[PyObjectRef],
    kwargs: Option<PyObjectRef>,
) -> Result<PyObjectRef, crate::PyError> {
    call_method_def_in_class(method, w_self, None, positional, kwargs)
}

/// [`call_method_def`] naming the class the definition was declared in, which
/// a `METH_METHOD` row is handed beside its receiver.
pub(super) fn call_method_def_in_class(
    method: *mut CPyMethodDef,
    w_self: PyObjectRef,
    defining_class: Option<PyObjectRef>,
    positional: &[PyObjectRef],
    kwargs: Option<PyObjectRef>,
) -> Result<PyObjectRef, crate::PyError> {
    let name = text_or_empty(unsafe { (*method).ml_name });
    let flags = unsafe { (*method).ml_flags } & !(METH_CLASS | METH_STATIC | METH_COEXIST);
    let keywords: Vec<(String, PyObjectRef)> = match kwargs {
        Some(dict) if crate::builtins::has_real_kwargs(kwargs) => unsafe {
            pyre_object::w_dict_str_entries(dict)
                .into_iter()
                .filter(|(key, _)| key != "__pyre_kw__")
                .collect()
        },
        _ => Vec::new(),
    };
    if !keywords.is_empty() && flags & METH_KEYWORDS == 0 {
        return Err(crate::PyError::type_error(format!(
            "{name}() takes no keyword arguments"
        )));
    }
    if flags & METH_NOARGS != 0 && !positional.is_empty() {
        return Err(crate::PyError::type_error(format!(
            "{name}() takes no arguments ({} given)",
            positional.len()
        )));
    }
    if flags & METH_O != 0 && positional.len() != 1 {
        return Err(crate::PyError::type_error(format!(
            "{name}() takes exactly one argument ({} given)",
            positional.len()
        )));
    }
    // Exactly one, not at least one: the flags select which signature
    // `ml_meth` has, so a table declaring two of them names no signature at
    // all and would otherwise be dispatched through whichever this layer
    // happens to test for first.
    if (flags & (METH_NOARGS | METH_O | METH_VARARGS | METH_FASTCALL)).count_ones() != 1 {
        return Err(crate::PyError::runtime_error(format!(
            "{name}() uses an unknown calling convention"
        )));
    }
    // `METH_METHOD` widens the fastcall-with-keywords signature by one
    // argument rather than naming a signature of its own, so it is the only
    // combination it may appear in; in any other it would select a transmute
    // whose arity the callee does not have.
    if flags & METH_METHOD != 0
        && flags & (METH_FASTCALL | METH_KEYWORDS) != (METH_FASTCALL | METH_KEYWORDS)
    {
        return Err(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            format!("{name}() method: bad call flags"),
        ));
    }
    let defining_class = match (flags & METH_METHOD != 0, defining_class) {
        (false, _) => None,
        (true, Some(class)) => Some(class),
        (true, None) => {
            return Err(crate::PyError::new(
                crate::PyErrorKind::SystemError,
                format!("{name}() method: no defining class"),
            ));
        }
    };
    super::call_cfunction_in_class(
        unsafe { (*method).ml_meth },
        flags,
        w_self,
        defining_class,
        positional,
        &keywords,
    )
}

pub(super) fn ensure_linked() {}
