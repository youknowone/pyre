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

/// The carrier type, named as upstream names its typedef.
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
        });
        unsafe { pyre_object::typeobject::w_type_set_hasdict(tp, true) };
        tp as usize
    }) as PyObjectRef
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
    let roots = pyre_object::gc_roots::push_roots();
    let self_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(w_self);
    let module_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(w_module);
    let carrier = pyre_object::w_instance_new(pycfunction_type());
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
    let w_self = carrier_get(carrier, SELF_KEY).unwrap_or_else(pyre_object::w_none);
    call_method_def(method, w_self, positional, kwargs)
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
    super::call_cfunction(
        unsafe { (*method).ml_meth },
        flags,
        w_self,
        positional,
        &keywords,
    )
}

pub(super) fn ensure_linked() {}
