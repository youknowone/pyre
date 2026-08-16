//! Module definitions and method-table conversion -- PyPy
//! `cpyext/modsupport.py`.

use super::methodobject::{CPyMethodDef, METH_CLASS, METH_COEXIST, METH_STATIC, new_pycfunction};
use super::pyerrors::{self, trap};
use super::pyobject::{self, CPyObject, REFCNT_IMMORTAL};
use pyre_object::{PY_NULL, PyObjectRef};
use std::ffi::{CStr, c_char, c_int, c_long, c_void};
use std::hash::BuildHasherDefault;
use std::sync::atomic::{AtomicIsize, Ordering};

/// `Py_mod_create`.
const PY_MOD_CREATE: c_int = 1;
/// `Py_mod_exec`.
const PY_MOD_EXEC: c_int = 2;
/// `Py_mod_multiple_interpreters`.
const PY_MOD_MULTIPLE_INTERPRETERS: c_int = 3;
/// `Py_mod_gil`.
const PY_MOD_GIL: c_int = 4;

/// The two versions a module may declare: the full API version an extension
/// built against `Python.h` carries, and the stable-ABI one a limited-API
/// build carries instead.  Both must match `Python.h`.
const PYTHON_API_VERSION: c_int = 1013;
const PYTHON_ABI_VERSION: c_int = 3;

/// The `md_def` and `md_state` CPython keeps in the module object's own C
/// struct.
///
/// Pyre's module has no such fields.  The module dictionary is not a place to
/// put them: it is the module's Python-visible namespace, so `vars(mod)` and
/// `mod.__dict__[...] = 0` both reach it, and `PyModule_GetState` would then
/// hand a C extension an address of the writer's choosing to dereference.  A
/// mirror is the C-side half of the module, never moves, and dies with the
/// module, so the pair is filed under the mirror's address instead.
#[derive(Clone, Copy, Default)]
struct ModuleFields {
    md_def: usize,
    md_state: usize,
}

type ModuleTable =
    std::collections::HashMap<usize, ModuleFields, BuildHasherDefault<std::hash::DefaultHasher>>;
static MODULE_FIELDS: super::ForkMutex<ModuleTable> = super::ForkMutex::new(
    std::collections::HashMap::with_hasher(BuildHasherDefault::new()),
);

pub(super) unsafe fn after_fork_child() {
    unsafe { MODULE_FIELDS.reinit_after_fork() };
}

/// The pair a module has recorded, all zero for one that has recorded none.
///
/// The read never builds a mirror: a module with none has nothing filed.
fn fields(module: PyObjectRef) -> ModuleFields {
    let mirror = pyobject::as_pyobj(module) as usize;
    if mirror == 0 {
        return ModuleFields::default();
    }
    MODULE_FIELDS
        .lock()
        .get(&mirror)
        .copied()
        .unwrap_or_default()
}

/// Record one of the pair, building the module's mirror if it has none.
///
/// # Safety
/// The caller must be holding `module` rooted, as [`pyobject::borrow_mirror`]
/// requires.
fn set_field(module: PyObjectRef, update: impl FnOnce(&mut ModuleFields)) {
    // Outside the lock: building a mirror takes the census lock of its own.
    let mirror = pyobject::borrow_mirror(module) as usize;
    if mirror == 0 {
        return;
    }
    update(MODULE_FIELDS.lock().entry(mirror).or_default());
}

/// Drop what a dying module mirror recorded.
///
/// The state block itself is not freed: pyre has no module deallocation path,
/// and an extension may still hold the address.
pub(super) fn forget_module_fields(mirror: usize) {
    MODULE_FIELDS.lock().remove(&mirror);
}

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
    set_field(module, |fields| fields.md_state = block as usize);
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

    set_field(reload(module_slot), |fields| fields.md_def = def as usize);

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
            (*def).m_base.ob_base.ob_type = &raw mut super::typeobject::CPY_MODULE_DEF_TYPE;
            (*def).m_base.m_index = NEXT_MODULE_INDEX.fetch_add(1, Ordering::Relaxed);
        }
        &mut (*def).m_base.ob_base
    }
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
    // `moduleobject.c:322-343` records each of the two declarations once and
    // rejects a repeat.  What either one selects is guarded by
    // `!_Py_IsMainInterpreter` or by the per-interpreter GIL, neither of which
    // pyre has, so recording is the whole of it here.
    let mut declared_interpreters = false;
    let mut declared_gil = false;
    let mut slot = unsafe { (*def).m_slots };
    while !slot.is_null() && unsafe { (*slot).slot } != 0 {
        let repeated = |what: &str| {
            Err(crate::PyError::new(
                crate::PyErrorKind::SystemError,
                format!("module {name} has multiple {what} slots"),
            ))
        };
        match unsafe { (*slot).slot } {
            PY_MOD_CREATE => {
                if !create.is_null() {
                    return repeated("create");
                }
                create = unsafe { (*slot).value };
            }
            PY_MOD_EXEC => has_execution_slots = true,
            PY_MOD_MULTIPLE_INTERPRETERS => {
                if declared_interpreters {
                    return repeated("Py_mod_multiple_interpreters");
                }
                declared_interpreters = true;
            }
            PY_MOD_GIL => {
                if declared_gil {
                    return repeated("Py_mod_gil");
                }
                declared_gil = true;
            }
            other => {
                return Err(crate::PyError::new(
                    crate::PyErrorKind::SystemError,
                    format!("module {name} uses unknown slot ID {other}"),
                ));
            }
        }
        slot = unsafe { slot.add(1) };
    }

    // The create slot is handed the spec as its first argument and reads
    // `spec.name` out of it, so a NULL is dereferenced inside the extension.
    // An import that reaches here before the importlib bootstrap can build a
    // spec has none to hand over.
    if !create.is_null() && spec.is_null() {
        return Err(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            format!("module {name}: the create slot needs a module spec"),
        ));
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
    fields(module).md_state != 0
}

/// `modsupport.py:exec_def` — the PEP 489 exec phase, on the definition the
/// module recorded for itself.
pub(super) fn exec_def(module: PyObjectRef) -> Result<(), crate::PyError> {
    let address = fields(module).md_def;
    if address == 0 {
        return Ok(());
    }
    exec_def_of(module, address as *mut CPyModuleDef)
}

/// The exec phase against an explicitly named definition, which is the form
/// `PyModule_ExecDef` hands in.
fn exec_def_of(module: PyObjectRef, def: *mut CPyModuleDef) -> Result<(), crate::PyError> {
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
    fields(module).md_state as *mut c_void
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyModule_GetDef(module: *mut CPyObject) -> *mut CPyModuleDef {
    let Some(module) = module_argument(module, "PyModule_GetDef") else {
        return std::ptr::null_mut();
    };
    fields(module).md_def as *mut CPyModuleDef
}

/// Steals a reference to `value` on success, which is `PyModule_AddObject`'s
/// documented and much-complained-about contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyModule_AddObject(
    module: *mut CPyObject,
    name: *const c_char,
    value: *mut CPyObject,
) -> c_int {
    super::object::realize_all([module, value]);
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
    super::object::realize_all([module, value]);
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

/// The value is a C `long`, which is the width the reference declaration says
/// and narrower than a `Py_ssize_t` wherever the two differ.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyModule_AddIntConstant(
    module: *mut CPyObject,
    name: *const c_char,
    value: c_long,
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

/// `module.py:24` with the name object kept as it arrived: a module name can
/// carry a lone surrogate, so projecting it through a `&str` would lose the
/// buffer the caller handed over.
///
/// The four entries beyond `__name__` are the ones a module built this way is
/// documented to arrive with; `__file__` is deliberately not among them, and
/// is the caller's to add.
fn new_module(w_name: PyObjectRef) -> PyObjectRef {
    let roots = pyre_object::gc_roots::push_roots();
    let name_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(w_name);
    // The empty name is the anonymous-module sentinel, so the name is attached
    // afterwards rather than through the allocation.
    let module = pyre_object::w_module_new_managed("");
    let module_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(module);
    let reload = |slot| pyre_object::gc_roots::shadow_stack_get(slot);
    unsafe { pyre_object::w_module_set_name(reload(module_slot), reload(name_slot)) };
    store(reload(module_slot), "__name__", reload(name_slot));
    for key in ["__doc__", "__package__", "__loader__", "__spec__"] {
        store(reload(module_slot), key, pyre_object::w_none());
    }
    reload(module_slot)
}

/// `modsupport.py:PyModule_New` — only `__name__` is filled in; `__file__` is
/// the caller's to provide.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyModule_New(name: *const c_char) -> *mut CPyObject {
    if name.is_null() {
        unsafe { pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    let name = text_or_empty(name);
    pyobject::make_ref(new_module(pyre_object::w_str_new(&name)))
}

/// `modsupport.py:PyModule_NewObject`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyModule_NewObject(name: *mut CPyObject) -> *mut CPyObject {
    let w_name = unsafe { pyobject::from_ref(name) };
    if w_name.is_null() {
        unsafe { pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    pyobject::make_ref(new_module(w_name))
}

/// The `__name__` a module reports.
///
/// Upstream reads `w_mod.w_name`, the name frozen at construction; the module
/// dictionary is read here instead so that a module renamed after import
/// reports the new name, and every module pyre builds seeds both from the same
/// object.
fn module_name_object(module: PyObjectRef, function: &str) -> Option<PyObjectRef> {
    let dict = module_dict(module);
    let name = if dict.is_null() {
        None
    } else {
        unsafe { pyre_object::dictmultiobject::w_dict_getitem_str(dict, "__name__") }
    };
    match name {
        Some(name) if unsafe { pyre_object::is_str(name) } => Some(name),
        _ => {
            pyerrors::set_pending_error(crate::PyError::new(
                crate::PyErrorKind::SystemError,
                format!("{function}(): nameless module"),
            ));
            None
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyModule_GetNameObject(module: *mut CPyObject) -> *mut CPyObject {
    let Some(module) = module_argument(module, "PyModule_GetNameObject") else {
        return std::ptr::null_mut();
    };
    let Some(name) = module_name_object(module, "PyModule_GetNameObject") else {
        return std::ptr::null_mut();
    };
    pyobject::make_ref(name)
}

/// The buffer belongs to the name's mirror, which the module dictionary keeps
/// alive for as long as the module — the lifetime upstream relies on for the
/// same `char *`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyModule_GetName(module: *mut CPyObject) -> *const c_char {
    let Some(module) = module_argument(module, "PyModule_GetName") else {
        return std::ptr::null();
    };
    let Some(name) = module_name_object(module, "PyModule_GetName") else {
        return std::ptr::null();
    };
    unsafe { super::unicodeobject::PyUnicode_AsUTF8(pyobject::borrow_mirror(name)) }
}

/// The `__file__` a module was loaded from.
///
/// A missing or non-`str` entry is the `SystemError` the entry point is
/// documented to raise, rather than the `KeyError` a plain `getitem` would
/// surface.
fn module_filename_object(module: PyObjectRef, function: &str) -> Option<PyObjectRef> {
    let dict = module_dict(module);
    let file = if dict.is_null() {
        None
    } else {
        unsafe { pyre_object::dictmultiobject::w_dict_getitem_str(dict, "__file__") }
    };
    match file {
        Some(file) if unsafe { pyre_object::is_str(file) } => Some(file),
        _ => {
            pyerrors::set_pending_error(crate::PyError::new(
                crate::PyErrorKind::SystemError,
                format!("{function}(): module filename missing"),
            ));
            None
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyModule_GetFilenameObject(module: *mut CPyObject) -> *mut CPyObject {
    let Some(module) = module_argument(module, "PyModule_GetFilenameObject") else {
        return std::ptr::null_mut();
    };
    let Some(file) = module_filename_object(module, "PyModule_GetFilenameObject") else {
        return std::ptr::null_mut();
    };
    pyobject::make_ref(file)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyModule_GetFilename(module: *mut CPyObject) -> *const c_char {
    let Some(module) = module_argument(module, "PyModule_GetFilename") else {
        return std::ptr::null();
    };
    let Some(file) = module_filename_object(module, "PyModule_GetFilename") else {
        return std::ptr::null();
    };
    unsafe { super::unicodeobject::PyUnicode_AsUTF8(pyobject::borrow_mirror(file)) }
}

/// Releases `value` whether the store succeeds or not, which is what makes
/// this the spelling that composes with a call producing a new reference.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyModule_Add(
    module: *mut CPyObject,
    name: *const c_char,
    value: *mut CPyObject,
) -> c_int {
    super::object::realize_all([module, value]);
    let result = unsafe { PyModule_AddObjectRef(module, name, value) };
    if !value.is_null() {
        unsafe { pyobject::decref(value) };
    }
    result
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyModule_SetDocString(
    module: *mut CPyObject,
    doc: *const c_char,
) -> c_int {
    let Some(module) = module_argument(module, "PyModule_SetDocString") else {
        return -1;
    };
    if doc.is_null() {
        unsafe { pyerrors::PyErr_BadInternalCall() };
        return -1;
    }
    let text = text_or_empty(doc);
    let roots = pyre_object::gc_roots::push_roots();
    let module_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(module);
    let value = pyre_object::w_str_new(&text);
    store(
        pyre_object::gc_roots::shadow_stack_get(module_slot),
        "__doc__",
        value,
    );
    0
}

/// `modsupport.py:PyModule_AddFunctions`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyModule_AddFunctions(
    module: *mut CPyObject,
    methods: *mut CPyMethodDef,
) -> c_int {
    let Some(module) = module_argument(module, "PyModule_AddFunctions") else {
        return -1;
    };
    let roots = pyre_object::gc_roots::push_roots();
    let module_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(module);
    let Some(name) = module_name_object(
        pyre_object::gc_roots::shadow_stack_get(module_slot),
        "PyModule_AddFunctions",
    ) else {
        return -1;
    };
    let name_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(name);
    if trap(convert_method_defs(
        pyre_object::gc_roots::shadow_stack_get(module_slot),
        methods,
        pyre_object::gc_roots::shadow_stack_get(name_slot),
    ))
    .is_none()
    {
        return -1;
    }
    0
}

/// `modsupport.py:PyModule_ExecDef` — the exec phase driven by the caller
/// rather than by import, against the definition it names.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyModule_ExecDef(module: *mut CPyObject, def: *mut CPyModuleDef) -> c_int {
    let Some(module) = module_argument(module, "PyModule_ExecDef") else {
        return -1;
    };
    if def.is_null() {
        unsafe { pyerrors::PyErr_BadInternalCall() };
        return -1;
    }
    if trap(exec_def_of(module, def)).is_none() {
        return -1;
    }
    0
}

/// `modsupport.py:PyModule_FromDefAndSpec2` — the PEP 489 create phase reached
/// from C instead of from import, so the module records no `__file__`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyModule_FromDefAndSpec2(
    def: *mut CPyModuleDef,
    spec: *mut CPyObject,
    module_api_version: c_int,
) -> *mut CPyObject {
    if def.is_null() {
        unsafe { pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    let w_spec = if spec.is_null() {
        pyre_object::w_none()
    } else {
        unsafe { pyobject::from_ref(spec) }
    };
    let roots = pyre_object::gc_roots::push_roots();
    let spec_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(w_spec);
    let Some(name) = trap(
        crate::baseobjspace::getattr_str(
            pyre_object::gc_roots::shadow_stack_get(spec_slot),
            "name",
        )
        .and_then(|w_name| crate::baseobjspace::text_w(w_name).map(str::to_owned)),
    ) else {
        return std::ptr::null_mut();
    };
    if module_api_version != PYTHON_API_VERSION && module_api_version != PYTHON_ABI_VERSION {
        let truncated: String = name.chars().take(100).collect();
        let message = format!(
            "Python C API version mismatch for module {truncated}: \
             This Python has API version {PYTHON_API_VERSION}, \
             module {truncated} has version {module_api_version}."
        );
        if trap(crate::warn::warn_category(&message, "RuntimeWarning", 1)).is_none() {
            return std::ptr::null_mut();
        }
    }
    let Some(module) = trap(create_module_from_def_and_spec(
        def,
        pyre_object::gc_roots::shadow_stack_get(spec_slot),
        &name,
        None,
    )) else {
        return std::ptr::null_mut();
    };
    pyobject::make_ref(module)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyModule_Check(object: *mut CPyObject) -> c_int {
    let object = unsafe { pyobject::from_ref(object) };
    (!object.is_null() && unsafe { pyre_object::module::is_module(object) }) as c_int
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyModule_CheckExact(object: *mut CPyObject) -> c_int {
    let object = unsafe { pyobject::from_ref(object) };
    (!object.is_null() && super::object::is_exactly(object, &pyre_object::MODULE_TYPE)) as c_int
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
    std::hint::black_box(PyModule_New as *const ());
    std::hint::black_box(PyModule_NewObject as *const ());
    std::hint::black_box(PyModule_GetName as *const ());
    std::hint::black_box(PyModule_GetNameObject as *const ());
    std::hint::black_box(PyModule_GetFilename as *const ());
    std::hint::black_box(PyModule_GetFilenameObject as *const ());
    std::hint::black_box(PyModule_Add as *const ());
    std::hint::black_box(PyModule_SetDocString as *const ());
    std::hint::black_box(PyModule_AddFunctions as *const ());
    std::hint::black_box(PyModule_ExecDef as *const ());
    std::hint::black_box(PyModule_FromDefAndSpec2 as *const ());
    std::hint::black_box(PyModule_Check as *const ());
    std::hint::black_box(PyModule_CheckExact as *const ());
}
