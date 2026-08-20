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
    // `__import__` is pinned before the name is built: minting the string
    // allocates, and a collection there would leave this a pre-move address.
    let import_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(import);
    let name_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(pyre_object::w_str_new(name));
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

/// `import.c:4276 PyImport_ImportModuleAttr` — one attribute of a module
/// imported for it, which is the shape a C caller reaching into Python has.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyImport_ImportModuleAttr(
    module_name: *mut CPyObject,
    attribute_name: *mut CPyObject,
) -> *mut CPyObject {
    let module = unsafe { PyImport_Import(module_name) };
    if module.is_null() {
        return std::ptr::null_mut();
    }
    let attribute = unsafe { super::object::PyObject_GetAttr(module, attribute_name) };
    unsafe { pyobject::decref(module) };
    attribute
}

/// The `const char *` spelling of the pair.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyImport_ImportModuleAttrString(
    module_name: *const c_char,
    attribute_name: *const c_char,
) -> *mut CPyObject {
    let module = unsafe { PyImport_ImportModule(module_name) };
    if module.is_null() {
        return std::ptr::null_mut();
    }
    let attribute = unsafe { super::object::PyObject_GetAttrString(module, attribute_name) };
    unsafe { pyobject::decref(module) };
    attribute
}

/// `import_.py:33 PyImport_GetModuleDict()` — `sys.modules`, borrowed.
///
/// The borrow is sound where `PyImport_AddModule`'s is not: the modules dict
/// is made during start-up and lives for as long as the interpreter does, so
/// the mirror it is handed out through outlives every caller.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyImport_GetModuleDict() -> *mut CPyObject {
    let modules = crate::importing::sys_modules_dict();
    if modules.is_null() {
        super::pyerrors::set_pending_error(crate::PyError::new(
            crate::PyErrorKind::RuntimeError,
            "sys.modules is not set",
        ));
        return std::ptr::null_mut();
    }
    pyobject::borrow_mirror(modules)
}

/// `import_.py:58 PyImport_ImportModuleLevelObject(name, globals, locals,
/// fromlist, level)` — `__import__` with each of its arguments.
///
/// A NULL stands for the argument's default, which is what an extension that
/// only wants one of them passes for the rest.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyImport_ImportModuleLevelObject(
    name: *mut CPyObject,
    globals: *mut CPyObject,
    locals: *mut CPyObject,
    fromlist: *mut CPyObject,
    level: std::ffi::c_int,
) -> *mut CPyObject {
    super::object::realize_all([name, globals, locals, fromlist]);
    // Answered before the conversion and before `level`, which is the order
    // `import.c` asks in: a NULL name is the empty-name case rather than a
    // bad internal call, and the message says so.
    if name.is_null() {
        super::pyerrors::set_pending_error(crate::PyError::value_error(
            "Empty module name".to_owned(),
        ));
        return std::ptr::null_mut();
    }
    let Some(name) = argument(name) else {
        return std::ptr::null_mut();
    };
    if level < 0 {
        super::pyerrors::set_pending_error(crate::PyError::value_error(
            "level must be >= 0".to_owned(),
        ));
        return std::ptr::null_mut();
    }
    result(import_module_level(
        name,
        globals,
        locals,
        fromlist,
        level as i64,
    ))
}

/// `PyImport_ImportModuleLevel(name, globals, locals, fromlist, level)` — the
/// `const char *` spelling of the entry point above.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyImport_ImportModuleLevel(
    name: *const c_char,
    globals: *mut CPyObject,
    locals: *mut CPyObject,
    fromlist: *mut CPyObject,
    level: std::ffi::c_int,
) -> *mut CPyObject {
    if name.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    let text = unsafe { super::unicodeobject::PyUnicode_FromString(name) };
    if text.is_null() {
        return std::ptr::null_mut();
    }
    let answer =
        unsafe { PyImport_ImportModuleLevelObject(text, globals, locals, fromlist, level) };
    unsafe { pyobject::decref(text) };
    answer
}

/// The body of [`PyImport_ImportModuleLevelObject`], with the defaults filled
/// in and every argument rooted across the allocations the others make.
fn import_module_level(
    name: PyObjectRef,
    globals: *mut CPyObject,
    locals: *mut CPyObject,
    fromlist: *mut CPyObject,
    level: i64,
) -> Result<PyObjectRef, crate::PyError> {
    let builtins = crate::importing::get_sys_module("builtins").ok_or_else(|| {
        crate::PyError::new(crate::PyErrorKind::ImportError, "builtins is not loaded")
    })?;
    let import = crate::baseobjspace::getattr_str(builtins, "__import__")?;
    let roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(import);
    roots.pin_root(name);
    // Each default allocates, so every argument is on the stack before the
    // next one is built and all five are read back afterwards.
    for (raw, default) in [(globals, None), (locals, None), (fromlist, Some(()))] {
        let given = unsafe { pyobject::from_ref(raw) };
        let value = if !given.is_null() {
            given
        } else if default.is_some() {
            pyre_object::w_tuple_new_array_backed(Vec::new())
        } else {
            pyre_object::w_none()
        };
        roots.pin_root(value);
    }
    roots.pin_root(pyre_object::w_int_new(level));
    let at = |index: usize| pyre_object::gc_roots::shadow_stack_get(base + index);
    crate::call::call_function_impl_result(at(0), &[at(1), at(2), at(3), at(4), at(5)])
}

/// `PyImport_AddModuleRef` — the module under `name`, created empty when
/// `sys.modules` does not have it yet.
///
/// The 3.12-and-earlier `PyImport_AddModule` returns a *borrowed* reference
/// with no container for pyre to hang the borrow on, so only the
/// strong-reference form is provided.
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
    std::hint::black_box(PyImport_ImportModuleAttr as *const ());
    std::hint::black_box(PyImport_ImportModuleAttrString as *const ());
}
