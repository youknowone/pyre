//! The import entry points -- PyPy `cpyext/import_.py`.

use super::object::{argument, result};
use super::pyobject::{self, CPyObject};
use pyre_object::PyObjectRef;
use std::ffi::{CStr, c_char};

/// `sys.modules[name]`, importing it first if it is not there yet.
pub(super) fn import_module(name: &str) -> Result<PyObjectRef, crate::PyError> {
    if let Some(module) = crate::importing::get_sys_module(name) {
        return Ok(module);
    }
    let builtins = crate::importing::get_sys_module("builtins").ok_or_else(|| {
        crate::PyError::new(crate::PyErrorKind::ImportError, "builtins is not loaded")
    })?;
    let import = crate::baseobjspace::getattr_str(builtins, "__import__")?;
    let roots = pyre_object::gc_roots::push_roots();
    let name_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(pyre_object::w_str_new(name));
    let import_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(import);
    // `__import__` answers with the top-level package, so the submodule is
    // read back out of `sys.modules` afterwards.
    crate::call::call_function_impl_result(
        pyre_object::gc_roots::shadow_stack_get(import_slot),
        &[pyre_object::gc_roots::shadow_stack_get(name_slot)],
    )?;
    crate::importing::get_sys_module(name).ok_or_else(|| {
        crate::PyError::new(
            crate::PyErrorKind::ImportError,
            format!("import of {name} left nothing in sys.modules"),
        )
    })
}

fn name_of(pointer: *const c_char) -> Option<String> {
    if pointer.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return None;
    }
    Some(
        unsafe { CStr::from_ptr(pointer) }
            .to_string_lossy()
            .into_owned(),
    )
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyImport_ImportModule(name: *const c_char) -> *mut CPyObject {
    let Some(name) = name_of(name) else {
        return std::ptr::null_mut();
    };
    result(import_module(&name))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyImport_ImportModuleNoBlock(name: *const c_char) -> *mut CPyObject {
    unsafe { PyImport_ImportModule(name) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyImport_Import(name: *mut CPyObject) -> *mut CPyObject {
    let Some(name) = argument(name) else {
        return std::ptr::null_mut();
    };
    if !unsafe { pyre_object::unicodeobject::is_str(name) } {
        super::pyerrors::set_pending_error(crate::PyError::type_error(
            "module name must be a string",
        ));
        return std::ptr::null_mut();
    }
    let text = unsafe { pyre_object::w_str_get_wtf8(name) }.to_string();
    result(import_module(&text))
}

/// `PyImport_AddModuleRef` — the module under `name`, created empty when
/// `sys.modules` does not have it yet.
///
/// The 3.12-and-earlier `PyImport_AddModule` and `PyImport_GetModuleDict`
/// return *borrowed* references with no container for pyre to hang the borrow
/// on, so only the strong-reference forms are provided.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyImport_AddModuleRef(name: *const c_char) -> *mut CPyObject {
    let Some(name) = name_of(name) else {
        return std::ptr::null_mut();
    };
    if let Some(module) = crate::importing::get_sys_module(&name) {
        return pyobject::make_ref(module);
    }
    let roots = pyre_object::gc_roots::push_roots();
    let module = pyre_object::module::w_module_new(&name);
    let module_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(module);
    crate::importing::set_sys_module(&name, pyre_object::gc_roots::shadow_stack_get(module_slot));
    pyobject::make_ref(pyre_object::gc_roots::shadow_stack_get(module_slot))
}

/// `PyImport_GetModule(name)` — the module already in `sys.modules`, or NULL
/// with no exception set.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyImport_GetModule(name: *mut CPyObject) -> *mut CPyObject {
    let Some(name) = argument(name) else {
        return std::ptr::null_mut();
    };
    if !unsafe { pyre_object::unicodeobject::is_str(name) } {
        super::pyerrors::set_pending_error(crate::PyError::type_error(
            "module name must be a string",
        ));
        return std::ptr::null_mut();
    }
    let text = unsafe { pyre_object::w_str_get_wtf8(name) }.to_string();
    match crate::importing::get_sys_module(&text) {
        Some(module) => pyobject::make_ref(module),
        None => std::ptr::null_mut(),
    }
}

pub(super) fn ensure_linked() {
    std::hint::black_box(PyImport_ImportModule as *const ());
    std::hint::black_box(PyImport_ImportModuleNoBlock as *const ());
    std::hint::black_box(PyImport_Import as *const ());
    std::hint::black_box(PyImport_AddModuleRef as *const ());
    std::hint::black_box(PyImport_GetModule as *const ());
}
