//! Module definitions and method-table conversion -- PyPy
//! `cpyext/modsupport.py`.

use super::methodobject::{CPyMethodDef, METH_CLASS, METH_COEXIST, METH_STATIC, new_pycfunction};
use super::pyerrors::{self, trap};
use super::pyobject::{self, CPyObject, REFCNT_IMMORTAL};
use pyre_object::{PY_NULL, PyObjectRef};
use std::ffi::{CStr, c_char, c_int, c_void};
use std::sync::atomic::{AtomicIsize, Ordering};

/// `Py_mod_create`.
const PY_MOD_CREATE: c_int = 1;
/// `Py_mod_exec`.
const PY_MOD_EXEC: c_int = 2;

/// Reserved module-dict keys.
///
/// CPython keeps `md_def` and `md_state` in the module object's own C struct.
/// Pyre's module has no such fields, so they ride the module dictionary under
/// names a C module cannot produce through `PyModule_AddObject`; the values are
/// addresses, held as ints.
const MD_DEF_KEY: &str = "__pyre_md_def__";
const MD_STATE_KEY: &str = "__pyre_md_state__";

static NEXT_MODULE_INDEX: AtomicIsize = AtomicIsize::new(1);

#[repr(C)]
pub struct CPyModuleDefBase {
    pub ob_base: CPyObject,
    pub m_init: Option<unsafe extern "C" fn() -> *mut CPyObject>,
    pub m_index: isize,
    pub m_copy: *mut CPyObject,
}

#[repr(C)]
pub struct CPyModuleDefSlot {
    pub slot: c_int,
    pub value: *mut c_void,
}

#[repr(C)]
pub struct CPyModuleDef {
    pub m_base: CPyModuleDefBase,
    pub m_name: *const c_char,
    pub m_doc: *const c_char,
    pub m_size: isize,
    pub m_methods: *mut CPyMethodDef,
    pub m_slots: *mut CPyModuleDefSlot,
    pub m_traverse: *const c_void,
    pub m_clear: *const c_void,
    pub m_free: *const c_void,
}

fn text_or_empty(pointer: *const c_char) -> String {
    if pointer.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(pointer) }
            .to_string_lossy()
            .into_owned()
    }
}

fn module_dict(module: PyObjectRef) -> PyObjectRef {
    unsafe { pyre_object::w_module_get_w_dict(module) }
}

fn store(module: PyObjectRef, name: &str, value: PyObjectRef) {
    crate::module_ns_store(module_dict(module), name, value);
}

fn stored_address(module: PyObjectRef, key: &str) -> usize {
    let dict = module_dict(module);
    if dict.is_null() {
        return 0;
    }
    match unsafe { pyre_object::dictmultiobject::w_dict_getitem_str(dict, key) } {
        Some(value) if unsafe { pyre_object::is_int(value) } => unsafe {
            pyre_object::w_int_get_value(value) as usize
        },
        _ => 0,
    }
}

/// `modsupport.py:convert_method_defs`, module branch.
///
/// The table is NUL-name terminated.  `METH_CLASS` / `METH_STATIC` are
/// rejected here exactly as upstream rejects them for a module-level table;
/// their type-level meaning arrives with C-defined types.
fn convert_method_defs(
    module: PyObjectRef,
    methods: *mut CPyMethodDef,
    w_module_name: PyObjectRef,
) -> Result<(), crate::PyError> {
    if methods.is_null() {
        return Ok(());
    }
    let roots = pyre_object::gc_roots::push_roots();
    let module_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(module);
    let name_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(w_module_name);
    let mut index = 0isize;
    loop {
        let method = unsafe { methods.offset(index) };
        if unsafe { (*method).ml_name.is_null() } {
            return Ok(());
        }
        index += 1;
        let name = text_or_empty(unsafe { (*method).ml_name });
        if unsafe { (*method).ml_flags } & (METH_CLASS | METH_STATIC) != 0 {
            return Err(crate::PyError::value_error(
                "module functions cannot set METH_CLASS or METH_STATIC",
            ));
        }
        let function = new_pycfunction(
            method,
            pyre_object::gc_roots::shadow_stack_get(module_slot),
            pyre_object::gc_roots::shadow_stack_get(name_slot),
        )?;
        let function_slot = pyre_object::gc_roots::shadow_stack_len();
        roots.pin_root(function);
        store(
            pyre_object::gc_roots::shadow_stack_get(module_slot),
            &name,
            pyre_object::gc_roots::shadow_stack_get(function_slot),
        );
    }
}

/// Allocate the zero-filled per-module block `m_size` asks for.
///
/// A zero `m_size` still gets a block: [`exec_def`] uses the address as its
/// "already executed" marker, and address 0 cannot say that.
///
/// Never released: pyre has no module deallocation path yet, so the block
/// lives as long as the process — the same lifetime upstream's `md_state` has
/// in practice, since `module_dealloc` only runs when the module dies.
fn allocate_module_state(module: PyObjectRef, size: isize) -> Result<(), crate::PyError> {
    let block = unsafe { libc::calloc(1, size.max(1) as usize) };
    if block.is_null() {
        return Err(crate::PyError::new(
            crate::PyErrorKind::MemoryError,
            "cannot allocate cpyext module state",
        ));
    }
    let roots = pyre_object::gc_roots::push_roots();
    let module_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(module);
    let value = pyre_object::w_int_new(block as usize as i64);
    store(
        pyre_object::gc_roots::shadow_stack_get(module_slot),
        MD_STATE_KEY,
        value,
    );
    Ok(())
}

/// Fill a fresh module from `def`: definition address, file, methods and doc.
///
/// The state block is not allocated here — `PyModule_Create2` allocates it for
/// a single-phase module and [`exec_def`] for a multi-phase one.
fn populate_module(
    module: PyObjectRef,
    def: *mut CPyModuleDef,
    name: &str,
    path: Option<&std::path::Path>,
) -> Result<PyObjectRef, crate::PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let module_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(module);
    let reload = |slot| pyre_object::gc_roots::shadow_stack_get(slot);

    let value = pyre_object::w_int_new(def as usize as i64);
    store(reload(module_slot), MD_DEF_KEY, value);

    if let Some(path) = path {
        let value = crate::gateway::fsdecode_os_str(path.as_os_str());
        let value_slot = pyre_object::gc_roots::shadow_stack_len();
        roots.pin_root(value);
        store(reload(module_slot), "__file__", reload(value_slot));
    }

    let w_name = pyre_object::w_str_new(name);
    let name_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(w_name);
    convert_method_defs(
        reload(module_slot),
        unsafe { (*def).m_methods },
        reload(name_slot),
    )?;

    let doc = text_or_empty(unsafe { (*def).m_doc });
    if !doc.is_empty() {
        let value = pyre_object::w_str_new(&doc);
        let value_slot = pyre_object::gc_roots::shadow_stack_len();
        roots.pin_root(value);
        store(reload(module_slot), "__doc__", reload(value_slot));
    }

    Ok(reload(module_slot))
}

/// `PyModuleDef_Init` -- also the marker returned by PEP 489 init functions.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyModuleDef_Init(def: *mut CPyModuleDef) -> *mut CPyObject {
    if def.is_null() {
        return std::ptr::null_mut();
    }
    unsafe {
        if (*def).m_base.m_index == 0 {
            (*def).m_base.ob_base.ob_refcnt = REFCNT_IMMORTAL;
            (*def).m_base.ob_base.ob_pyre_link = PY_NULL;
            (*def).m_base.ob_base.ob_type = &raw mut pyobject::CPY_MODULE_DEF_TYPE;
            (*def).m_base.m_index = NEXT_MODULE_INDEX.fetch_add(1, Ordering::Relaxed);
        }
        &mut (*def).m_base.ob_base
    }
}

/// `true` when a mirror is a `PyModuleDef` rather than a linked object.
pub(super) fn is_module_def(raw: *mut CPyObject) -> bool {
    !raw.is_null()
        && unsafe { std::ptr::eq((*raw).ob_type, &raw mut pyobject::CPY_MODULE_DEF_TYPE) }
}

/// # Safety
/// `raw` must be a mirror [`is_module_def`] accepts.
pub(super) unsafe fn module_def_of(raw: *mut CPyObject) -> *mut CPyModuleDef {
    raw as *mut CPyModuleDef
}

/// `PyModule_Create2` -- PyPy `modsupport.py:PyModule_Create2`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyModule_Create2(
    def: *mut CPyModuleDef,
    _api_version: c_int,
) -> *mut CPyObject {
    if def.is_null() || unsafe { (*def).m_name.is_null() } {
        unsafe { pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    if unsafe { !(*def).m_slots.is_null() } {
        pyerrors::set_pending_error(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            format!(
                "module {} has multi-phase initialization slots, \
                 but PyModule_Create was called",
                text_or_empty(unsafe { (*def).m_name })
            ),
        ));
        return std::ptr::null_mut();
    }

    let declared_name = text_or_empty(unsafe { (*def).m_name });
    // PyPy `PyModule_Create2` consumes package_context before allocating the
    // module. Releasing the mutex here also avoids holding it across GC.
    let context = super::PACKAGE_CONTEXT.lock().take();
    let (name, path) = context
        .as_ref()
        .map(|(name, path)| (name.as_str(), Some(path.as_path())))
        .unwrap_or((declared_name.as_str(), None));
    let module = pyre_object::w_module_new_managed(name);
    let roots = pyre_object::gc_roots::push_roots();
    let module_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(module);
    let Some(module) = trap(populate_module(
        pyre_object::gc_roots::shadow_stack_get(module_slot),
        def,
        name,
        path,
    )) else {
        return std::ptr::null_mut();
    };
    if unsafe { (*def).m_size } > 0 {
        let module_slot = pyre_object::gc_roots::shadow_stack_len();
        roots.pin_root(module);
        if trap(allocate_module_state(
            pyre_object::gc_roots::shadow_stack_get(module_slot),
            unsafe { (*def).m_size },
        ))
        .is_none()
        {
            return std::ptr::null_mut();
        }
        return pyobject::make_ref(pyre_object::gc_roots::shadow_stack_get(module_slot));
    }
    pyobject::make_ref(module)
}

/// `modsupport.py:create_module_from_def_and_spec` — the PEP 489 create phase.
pub(super) fn create_module_from_def_and_spec(
    def: *mut CPyModuleDef,
    spec: PyObjectRef,
    name: &str,
    path: Option<&std::path::Path>,
) -> Result<PyObjectRef, crate::PyError> {
    if unsafe { (*def).m_size } < 0 {
        return Err(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            format!("module {name}: m_size may not be negative for multi-phase initialization"),
        ));
    }
    let mut create: *mut c_void = std::ptr::null_mut();
    let mut has_execution_slots = false;
    let mut slot = unsafe { (*def).m_slots };
    while !slot.is_null() && unsafe { (*slot).slot } != 0 {
        match unsafe { (*slot).slot } {
            PY_MOD_CREATE => {
                if !create.is_null() {
                    return Err(crate::PyError::new(
                        crate::PyErrorKind::SystemError,
                        format!("module {name} has multiple create slots"),
                    ));
                }
                create = unsafe { (*slot).value };
            }
            PY_MOD_EXEC => has_execution_slots = true,
            other => {
                return Err(crate::PyError::new(
                    crate::PyErrorKind::SystemError,
                    format!("module {name} uses unknown slot ID {other}"),
                ));
            }
        }
        slot = unsafe { slot.add(1) };
    }

    let roots = pyre_object::gc_roots::push_roots();
    let spec_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(spec);
    let module = if create.is_null() {
        pyre_object::w_module_new_managed(name)
    } else {
        let spec_ref = pyobject::make_ref(pyre_object::gc_roots::shadow_stack_get(spec_slot));
        let result = unsafe {
            let call: unsafe extern "C" fn(*mut CPyObject, *mut CPyModuleDef) -> *mut CPyObject =
                std::mem::transmute(create);
            call(spec_ref, def)
        };
        unsafe { pyobject::decref(spec_ref) };
        super::from_c_result(result)?
    };
    let module_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(module);
    if !unsafe { pyre_object::module::is_module(module) } {
        if unsafe { (*def).m_size } > 0
            || unsafe { !(*def).m_traverse.is_null() }
            || unsafe { !(*def).m_clear.is_null() }
            || unsafe { !(*def).m_free.is_null() }
        {
            return Err(crate::PyError::new(
                crate::PyErrorKind::SystemError,
                format!("module {name} is not a module object, but requests module state"),
            ));
        }
        if has_execution_slots {
            return Err(crate::PyError::new(
                crate::PyErrorKind::SystemError,
                format!(
                    "module {name} specifies execution slots, \
                     but did not create a ModuleType instance"
                ),
            ));
        }
        return Ok(pyre_object::gc_roots::shadow_stack_get(module_slot));
    }
    populate_module(
        pyre_object::gc_roots::shadow_stack_get(module_slot),
        def,
        name,
        path,
    )
}

/// `true` once the module owns a state block, which is what
/// `exec_extension_module` reads to tell an executed module from a fresh one.
pub(super) fn has_module_state(module: PyObjectRef) -> bool {
    stored_address(module, MD_STATE_KEY) != 0
}

/// `modsupport.py:exec_def` — the PEP 489 exec phase.
pub(super) fn exec_def(module: PyObjectRef) -> Result<(), crate::PyError> {
    let address = stored_address(module, MD_DEF_KEY);
    if address == 0 {
        return Ok(());
    }
    let def = address as *mut CPyModuleDef;
    let roots = pyre_object::gc_roots::push_roots();
    let module_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(module);
    // The state block is allocated before the first slot runs, both because
    // `PyModule_GetState` is what an exec slot is there to fill and because its
    // address is the marker that keeps a second exec from running.
    if unsafe { (*def).m_size } >= 0
        && !has_module_state(pyre_object::gc_roots::shadow_stack_get(module_slot))
    {
        allocate_module_state(
            pyre_object::gc_roots::shadow_stack_get(module_slot),
            unsafe { (*def).m_size },
        )?;
    }
    let mut slot = unsafe { (*def).m_slots };
    while !slot.is_null() && unsafe { (*slot).slot } != 0 {
        if unsafe { (*slot).slot } == PY_MOD_EXEC {
            let module = pyre_object::gc_roots::shadow_stack_get(module_slot);
            let module_ref = pyobject::make_ref(module);
            let result = unsafe {
                let call: unsafe extern "C" fn(*mut CPyObject) -> c_int =
                    std::mem::transmute((*slot).value);
                call(module_ref)
            };
            unsafe { pyobject::decref(module_ref) };
            if result != 0 {
                return Err(pyerrors::take_pending_error().unwrap_or_else(|| {
                    crate::PyError::new(
                        crate::PyErrorKind::SystemError,
                        format!(
                            "execution of module {} failed without setting an exception",
                            text_or_empty(unsafe { (*def).m_name })
                        ),
                    )
                }));
            }
            if pyerrors::take_pending_error().is_some() {
                return Err(crate::PyError::new(
                    crate::PyErrorKind::SystemError,
                    format!(
                        "execution of module {} raised unreported exception",
                        text_or_empty(unsafe { (*def).m_name })
                    ),
                ));
            }
        }
        slot = unsafe { slot.add(1) };
    }
    Ok(())
}

// ── the module accessors ────────────────────────────────────────────────

fn module_argument(raw: *mut CPyObject, function: &str) -> Option<PyObjectRef> {
    let module = unsafe { pyobject::from_ref(raw) };
    if module.is_null() || !unsafe { pyre_object::module::is_module(module) } {
        pyerrors::set_pending_error(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            format!("{function}(): not a module"),
        ));
        return None;
    }
    Some(module)
}

/// Borrowed, exactly as upstream marks it `result_borrowed=True`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyModule_GetDict(module: *mut CPyObject) -> *mut CPyObject {
    let Some(module) = module_argument(module, "PyModule_GetDict") else {
        return std::ptr::null_mut();
    };
    // Borrowed: the module's mirror owns the reference for as long as it
    // lives, which is what makes the result usable after this returns.
    pyobject::borrow_from(pyobject::as_pyobj(module), module_dict(module))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyModule_GetState(module: *mut CPyObject) -> *mut c_void {
    let Some(module) = module_argument(module, "PyModule_GetState") else {
        return std::ptr::null_mut();
    };
    stored_address(module, MD_STATE_KEY) as *mut c_void
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyModule_GetDef(module: *mut CPyObject) -> *mut CPyModuleDef {
    let Some(module) = module_argument(module, "PyModule_GetDef") else {
        return std::ptr::null_mut();
    };
    stored_address(module, MD_DEF_KEY) as *mut CPyModuleDef
}

/// Steals a reference to `value` on success, which is `PyModule_AddObject`'s
/// documented and much-complained-about contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyModule_AddObject(
    module: *mut CPyObject,
    name: *const c_char,
    value: *mut CPyObject,
) -> c_int {
    if unsafe { PyModule_AddObjectRef(module, name, value) } != 0 {
        return -1;
    }
    unsafe { pyobject::decref(value) };
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyModule_AddObjectRef(
    module: *mut CPyObject,
    name: *const c_char,
    value: *mut CPyObject,
) -> c_int {
    let Some(module) = module_argument(module, "PyModule_AddObjectRef") else {
        return -1;
    };
    if name.is_null() || value.is_null() {
        unsafe { pyerrors::PyErr_BadInternalCall() };
        return -1;
    }
    let key = text_or_empty(name);
    let value = unsafe { pyobject::from_ref(value) };
    store(module, &key, value);
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyModule_AddIntConstant(
    module: *mut CPyObject,
    name: *const c_char,
    value: isize,
) -> c_int {
    let Some(module) = module_argument(module, "PyModule_AddIntConstant") else {
        return -1;
    };
    if name.is_null() {
        unsafe { pyerrors::PyErr_BadInternalCall() };
        return -1;
    }
    let key = text_or_empty(name);
    let roots = pyre_object::gc_roots::push_roots();
    let module_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(module);
    let value = pyre_object::w_int_new(value as i64);
    store(
        pyre_object::gc_roots::shadow_stack_get(module_slot),
        &key,
        value,
    );
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyModule_AddStringConstant(
    module: *mut CPyObject,
    name: *const c_char,
    value: *const c_char,
) -> c_int {
    let Some(module) = module_argument(module, "PyModule_AddStringConstant") else {
        return -1;
    };
    if name.is_null() || value.is_null() {
        unsafe { pyerrors::PyErr_BadInternalCall() };
        return -1;
    }
    let key = text_or_empty(name);
    let text = text_or_empty(value);
    let roots = pyre_object::gc_roots::push_roots();
    let module_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(module);
    let value = pyre_object::w_str_new(&text);
    store(
        pyre_object::gc_roots::shadow_stack_get(module_slot),
        &key,
        value,
    );
    0
}

pub(super) fn ensure_linked() {
    std::hint::black_box(PyModuleDef_Init as *const ());
    std::hint::black_box(PyModule_Create2 as *const ());
    std::hint::black_box(PyModule_GetDict as *const ());
    std::hint::black_box(PyModule_GetState as *const ());
    std::hint::black_box(PyModule_GetDef as *const ());
    std::hint::black_box(PyModule_AddObject as *const ());
    std::hint::black_box(PyModule_AddObjectRef as *const ());
    std::hint::black_box(PyModule_AddIntConstant as *const ());
    std::hint::black_box(PyModule_AddStringConstant as *const ());
}
