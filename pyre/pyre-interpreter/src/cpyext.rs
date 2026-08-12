//! CPython C-API compatibility layer -- PyPy `pypy/module/cpyext/`.
//!
//! This first vertical slice implements the object bridge and the single-phase
//! `PyModule_Create` import path.  The C-visible object is deliberately *not*
//! pyre's internal [`pyre_object::PyObject`]: PyPy likewise uses a separate raw
//! refcounted mirror (`cpyext/pyobject.py`) and links it to the moving GC object.

#![cfg(all(
    feature = "cpyext",
    not(feature = "sandbox"),
    any(target_os = "macos", target_os = "linux")
))]

use parking_lot::ReentrantMutex;
use pyre_object::{PY_NULL, PyObjectRef};
use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::ffi::{CStr, c_char, c_int, c_void};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicIsize, Ordering};
use std::sync::{LazyLock, Mutex};

/// The large ownership contribution held by the linked pyre object.
///
/// PyPy calls this `rawrefcount.REFCNT_FROM_PYPY`.  Ordinary C references are
/// added above it by the public `Py_INCREF`/`Py_DECREF` inline functions.
pub const REFCNT_FROM_PYRE: isize = 1 << (isize::BITS - 3);

/// C-visible `PyObject`, matching PyPy's `parse/cpyext_object.h` shape.
#[repr(C)]
pub struct CPyObject {
    pub ob_refcnt: isize,
    pub ob_pyre_link: PyObjectRef,
    pub ob_type: *mut CPyTypeObject,
}

/// Opaque in the initial public header.  A real mirror, rather than the
/// internal RPython-vtable-like `pyre_object::PyType`, is nevertheless used so
/// the two ABIs can never be confused accidentally.
#[repr(C)]
pub struct CPyTypeObject {
    _opaque: usize,
}

// Process-global immortal type mirrors, matching PyPy's static API objects.
static mut CPY_MODULE_TYPE: CPyTypeObject = CPyTypeObject { _opaque: 0 };
static mut CPY_MODULE_DEF_TYPE: CPyTypeObject = CPyTypeObject { _opaque: 0 };
static NEXT_MODULE_INDEX: AtomicIsize = AtomicIsize::new(1);

#[repr(C)]
pub struct CPyModuleDefBase {
    pub ob_base: CPyObject,
    pub m_init: Option<unsafe extern "C" fn() -> *mut CPyObject>,
    pub m_index: isize,
    pub m_copy: *mut CPyObject,
}

#[repr(C)]
pub struct CPyMethodDef {
    pub ml_name: *const c_char,
    pub ml_meth: *const c_void,
    pub ml_flags: c_int,
    pub ml_doc: *const c_char,
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

struct ExtensionCacheEntry {
    /// Copy of the module dict, exactly PyPy `State.extensions[path]`.
    dict: usize,
    /// Keeps the library and its static module definition alive.
    _handle: usize,
}

/// PyPy `State.extensions`: the upstream owner is a process/interpreter state
/// dictionary, so a `HashMap` is intentional here rather than a side-table
/// workaround.  The values are copied module dictionaries, not per-object
/// optimization metadata.
static EXTENSIONS: LazyLock<Mutex<HashMap<PathBuf, ExtensionCacheEntry>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// PyPy/rawrefcount's P-list analogue.  The relationship itself lives on the
/// raw object (`ob_pyre_link`); this Vec is only the collector's census of
/// mirrors whose inline link must be visited.
static RAW_OBJECTS: LazyLock<Mutex<Vec<usize>>> = LazyLock::new(|| Mutex::new(Vec::new()));
static CPYEXT_GC_ACTIVE: AtomicBool = AtomicBool::new(false);

/// `State.package_context`, consumed by `PyModule_Create2` while `PyInit_*`
/// runs. Imports are serialized by the import lock, as in PyPy.
static PACKAGE_CONTEXT: LazyLock<Mutex<Option<(String, PathBuf)>>> =
    LazyLock::new(|| Mutex::new(None));

/// PyPy performs cpyext initialization under the GIL. Pyre currently has no
/// process GIL, so this reentrant boundary preserves the same serialization
/// while still allowing an init function to import another extension.
struct ForkExtensionLoadLock(UnsafeCell<ReentrantMutex<()>>);
unsafe impl Sync for ForkExtensionLoadLock {}

impl ForkExtensionLoadLock {
    const fn new() -> Self {
        Self(UnsafeCell::new(ReentrantMutex::new(())))
    }

    fn get(&self) -> &ReentrantMutex<()> {
        unsafe { &*self.0.get() }
    }

    unsafe fn reinit_after_fork(&self) {
        unsafe { self.0.get().write(ReentrantMutex::new(())) };
    }
}

static EXTENSION_LOAD_LOCK: ForkExtensionLoadLock = ForkExtensionLoadLock::new();
static DLOPEN_FLAGS: AtomicIsize = AtomicIsize::new((libc::RTLD_NOW | libc::RTLD_LOCAL) as isize);

pub fn register_sys_dlopenflags(ns: PyObjectRef) {
    crate::module_ns_store(
        ns,
        "getdlopenflags",
        crate::make_builtin_function_with_arity(
            "getdlopenflags",
            |_| {
                Ok(pyre_object::w_int_new(
                    DLOPEN_FLAGS.load(Ordering::Relaxed) as i64
                ))
            },
            0,
        ),
    );
    crate::module_ns_store(
        ns,
        "setdlopenflags",
        crate::make_builtin_function_with_arity(
            "setdlopenflags",
            |args| {
                DLOPEN_FLAGS.store(
                    crate::baseobjspace::gateway_int_w(args[0])? as isize,
                    Ordering::Relaxed,
                );
                Ok(pyre_object::w_none())
            },
            1,
        ),
    );
}

type ExtensionLoadGuard = parking_lot::lock_api::ReentrantMutexGuard<
    'static,
    parking_lot::RawMutex,
    parking_lot::RawThreadId,
    (),
>;

#[majit_macros::dont_look_inside]
fn extension_load_lock() -> ExtensionLoadGuard {
    if let Some(guard) = EXTENSION_LOAD_LOCK.get().try_lock() {
        return guard;
    }
    let blocked = crate::module::thread::before_external_block();
    let guard = EXTENSION_LOAD_LOCK.get().lock();
    drop(blocked);
    guard
}

pub fn after_fork_child() {
    unsafe { EXTENSION_LOAD_LOCK.reinit_after_fork() };
    *PACKAGE_CONTEXT.lock().unwrap() = None;
}

pub const fn soabi() -> &'static str {
    if cfg!(target_os = "macos") {
        "pyre314-darwin"
    } else if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        "pyre314-x86_64-linux-gnu"
    } else if cfg!(all(target_os = "linux", target_arch = "aarch64")) {
        "pyre314-aarch64-linux-gnu"
    } else {
        "pyre314-linux-gnu"
    }
}

pub const fn extension_suffix() -> &'static str {
    if cfg!(target_os = "macos") {
        ".pyre314-darwin.so"
    } else if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        ".pyre314-x86_64-linux-gnu.so"
    } else if cfg!(all(target_os = "linux", target_arch = "aarch64")) {
        ".pyre314-aarch64-linux-gnu.so"
    } else {
        ".pyre314-linux-gnu.so"
    }
}

fn attach_module(module: PyObjectRef) -> *mut CPyObject {
    let raw = Box::into_raw(Box::new(CPyObject {
        // New reference returned by PyModule_Create, plus PyPy's link share.
        ob_refcnt: REFCNT_FROM_PYRE + 1,
        ob_pyre_link: module,
        ob_type: (&raw mut CPY_MODULE_TYPE),
    }));
    RAW_OBJECTS.lock().unwrap().push(raw as usize);
    CPYEXT_GC_ACTIVE.store(true, Ordering::Release);
    raw
}

unsafe fn detach_unowned_module_mirror(raw: *mut CPyObject) {
    if raw.is_null()
        || unsafe { (*raw).ob_refcnt != REFCNT_FROM_PYRE }
        || unsafe { !std::ptr::eq((*raw).ob_type, &raw mut CPY_MODULE_TYPE) }
    {
        return;
    }
    let address = raw as usize;
    RAW_OBJECTS
        .lock()
        .unwrap()
        .retain(|&candidate| candidate != address);
    unsafe {
        (*raw).ob_pyre_link = PY_NULL;
        (*raw).ob_refcnt -= REFCNT_FROM_PYRE;
        drop(Box::from_raw(raw));
    }
}

/// Forward/mark C-mirror links and cached module dictionaries.
///
/// External C ownership (`refcnt > REFCNT_FROM_PYRE`) keeps the linked object
/// alive, which is the essential P-link rule from `rawrefcount.rst`. Cached
/// dicts are interpreter-state roots just like PyPy's `State.extensions`.
pub fn walk_gc_roots(visitor: &mut dyn FnMut(&mut PyObjectRef)) {
    if !CPYEXT_GC_ACTIVE.load(Ordering::Acquire) {
        return;
    }
    for &address in RAW_OBJECTS.lock().unwrap().iter() {
        let raw = address as *mut CPyObject;
        unsafe {
            if (*raw).ob_refcnt > REFCNT_FROM_PYRE && !(*raw).ob_pyre_link.is_null() {
                visitor(&mut (*raw).ob_pyre_link);
            }
        }
    }
    for entry in EXTENSIONS.lock().unwrap().values_mut() {
        let mut dict = entry.dict as PyObjectRef;
        if !dict.is_null() {
            visitor(&mut dict);
            entry.dict = dict as usize;
        }
    }
}

fn module_from_cached_dict(name: &str, dict: PyObjectRef) -> PyObjectRef {
    let roots = pyre_object::gc_roots::push_roots();
    let dict_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(dict);
    let module = pyre_object::w_module_new_managed(name);
    let module_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(module);
    let module = pyre_object::gc_roots::shadow_stack_get(module_slot);
    let module_dict = unsafe { pyre_object::w_module_get_w_dict(module) };
    crate::type_methods::dict_update1(
        module_dict,
        pyre_object::gc_roots::shadow_stack_get(dict_slot),
    )
    .expect("cpyext cached module dictionaries are ordinary dicts");
    pyre_object::gc_roots::shadow_stack_get(module_slot)
}

fn cached_extension(name: &str, path: &Path) -> Option<PyObjectRef> {
    let roots = pyre_object::gc_roots::push_roots();
    let dict_slot = {
        let cache = EXTENSIONS.lock().unwrap();
        let dict = cache.get(path)?.dict as PyObjectRef;
        let slot = pyre_object::gc_roots::shadow_stack_len();
        roots.pin_root(dict);
        slot
    };
    Some(module_from_cached_dict(
        name,
        pyre_object::gc_roots::shadow_stack_get(dict_slot),
    ))
}

fn cached_extension_handle(path: &Path) -> Option<usize> {
    EXTENSIONS
        .lock()
        .unwrap()
        .get(path)
        .map(|entry| entry._handle)
}

fn fixup_extension(module: PyObjectRef, name: &str, path: &Path, handle: usize) {
    let roots = pyre_object::gc_roots::push_roots();
    let module_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(module);
    crate::importing::set_sys_module(name, module);
    let module = pyre_object::gc_roots::shadow_stack_get(module_slot);
    let dict = unsafe { pyre_object::w_module_get_w_dict(module) };
    let dict_copy = unsafe { pyre_object::dictmultiobject::w_dict_copy(dict) };
    EXTENSIONS.lock().unwrap().insert(
        path.to_path_buf(),
        ExtensionCacheEntry {
            dict: dict_copy as usize,
            _handle: handle,
        },
    );
    CPYEXT_GC_ACTIVE.store(true, Ordering::Release);
}

fn init_symbol(name: &str) -> Result<String, crate::PyError> {
    let basename = name.rsplit('.').next().unwrap_or(name);
    if !basename.is_ascii() {
        return Err(crate::PyError::new(
            crate::PyErrorKind::ImportError,
            format!("non-ASCII cpyext init names are not implemented yet: {basename}"),
        ));
    }
    Ok(format!("PyInit_{basename}"))
}

fn extension_import_error(message: String, name: &str, path: &Path) -> crate::PyError {
    let roots = pyre_object::gc_roots::push_roots();
    let name_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(pyre_object::w_str_new(name));
    let path_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(crate::gateway::fsdecode_os_str(path.as_os_str()));
    crate::PyError::import_error_name_path(
        message,
        pyre_object::gc_roots::shadow_stack_get(name_slot),
        pyre_object::gc_roots::shadow_stack_get(path_slot),
    )
}

/// PyPy `cpyext.api.create_extension_module`, single-phase branch.
pub fn load_extension_module(name: &str, path: &Path) -> Result<PyObjectRef, crate::PyError> {
    let _load_guard = extension_load_lock();
    ensure_linked();
    // `host_env::ctypes` deduplicates libraries by the native handle without
    // maintaining an open count. A second open followed by drop would unload
    // the one library owned by `EXTENSIONS`, so resolve PyPy's cache before
    // opening on this host abstraction.
    if let Some(handle) = cached_extension_handle(path) {
        let symbol = init_symbol(name)?;
        rustpython_host_env::ctypes::lookup_function_symbol_addr(handle, symbol.as_bytes())
            .map_err(|_error| {
                extension_import_error(
                    format!(
                        "function {symbol} not found in library '{}'",
                        path.display()
                    ),
                    name,
                    path,
                )
            })?;
        if let Some(module) = cached_extension(name, path) {
            let roots = pyre_object::gc_roots::push_roots();
            let module_slot = pyre_object::gc_roots::shadow_stack_len();
            roots.pin_root(module);
            crate::importing::set_sys_module(
                name,
                pyre_object::gc_roots::shadow_stack_get(module_slot),
            );
            return Ok(pyre_object::gc_roots::shadow_stack_get(module_slot));
        }
    }
    let mut mode = DLOPEN_FLAGS.load(Ordering::Relaxed) as c_int;
    if mode & (libc::RTLD_LAZY | libc::RTLD_NOW) == 0 {
        mode |= libc::RTLD_NOW;
    }
    let handle =
        rustpython_host_env::ctypes::open_library_with_mode(path, mode).map_err(|error| {
            extension_import_error(
                format!("cannot load extension '{}': {error}", path.display()),
                name,
                path,
            )
        })?;

    let symbol = match init_symbol(name) {
        Ok(symbol) => symbol,
        Err(error) => {
            rustpython_host_env::ctypes::drop_library(handle);
            return Err(error);
        }
    };
    let address =
        rustpython_host_env::ctypes::lookup_function_symbol_addr(handle, symbol.as_bytes())
            .map_err(|_error| {
                rustpython_host_env::ctypes::drop_library(handle);
                extension_import_error(
                    format!(
                        "function {symbol} not found in library '{}'",
                        path.display()
                    ),
                    name,
                    path,
                )
            })?;

    let old_context = PACKAGE_CONTEXT
        .lock()
        .unwrap()
        .replace((name.to_string(), path.to_path_buf()));
    let init: unsafe extern "C" fn() -> *mut CPyObject = unsafe { std::mem::transmute(address) };
    let result = unsafe { init() };
    *PACKAGE_CONTEXT.lock().unwrap() = old_context;

    if result.is_null() {
        rustpython_host_env::ctypes::drop_library(handle);
        return Err(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            format!("initialization of {name} failed without raising an exception"),
        ));
    }
    let module = unsafe { (*result).ob_pyre_link };
    if unsafe { (*result).ob_type.is_null() } {
        rustpython_host_env::ctypes::drop_library(handle);
        return Err(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            format!("init function of {name} returned an uninitialized object"),
        ));
    }
    if module.is_null() {
        rustpython_host_env::ctypes::drop_library(handle);
        return Err(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            format!("init function of {name} returned an unsupported object"),
        ));
    }
    unsafe {
        // Transfer the init function's owned result to the interpreter object,
        // exactly `get_w_obj_and_decref` in PyPy.
        (*result).ob_refcnt -= 1;
        detach_unowned_module_mirror(result);
    }
    fixup_extension(module, name, path, handle);
    Ok(module)
}

pub fn create_dynamic(spec: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let spec_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(spec);
    let w_name = crate::baseobjspace::getattr_str(
        pyre_object::gc_roots::shadow_stack_get(spec_slot),
        "name",
    )?;
    let name_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(w_name);
    let w_origin = crate::baseobjspace::getattr_str(
        pyre_object::gc_roots::shadow_stack_get(spec_slot),
        "origin",
    )?;
    let origin_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(w_origin);
    let name =
        crate::baseobjspace::text0_wtf8_w(pyre_object::gc_roots::shadow_stack_get(name_slot))?
            .to_string();
    crate::baseobjspace::text0_wtf8_w(pyre_object::gc_roots::shadow_stack_get(origin_slot))?;
    let path = PathBuf::from(crate::gateway::os_string_from_fs_bytes(
        &crate::gateway::fsencode(pyre_object::gc_roots::shadow_stack_get(origin_slot))?,
    ));
    let path = if path.components().count() == 1 {
        PathBuf::from(".").join(path)
    } else {
        path
    };
    load_extension_module(&name, &path)
}

/// `PyModuleDef_Init` -- also the marker returned by PEP 489 init functions.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyModuleDef_Init(def: *mut CPyModuleDef) -> *mut CPyObject {
    if def.is_null() {
        return std::ptr::null_mut();
    }
    unsafe {
        if (*def).m_base.m_index == 0 {
            (*def).m_base.ob_base.ob_refcnt = 1;
            (*def).m_base.ob_base.ob_pyre_link = PY_NULL;
            (*def).m_base.ob_base.ob_type = &raw mut CPY_MODULE_DEF_TYPE;
            (*def).m_base.m_index = NEXT_MODULE_INDEX.fetch_add(1, Ordering::Relaxed);
        }
        &mut (*def).m_base.ob_base
    }
}

/// `PyModule_Create2` -- PyPy `modsupport.py:PyModule_Create2`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyModule_Create2(
    def: *mut CPyModuleDef,
    _api_version: c_int,
) -> *mut CPyObject {
    if def.is_null() || unsafe { (*def).m_name.is_null() } {
        return std::ptr::null_mut();
    }
    // Method conversion is the next cpyext slice. Reject rather than silently
    // constructing a module that drops a non-empty PyMethodDef table.
    if unsafe { !(*def).m_methods.is_null() && !(*(*def).m_methods).ml_name.is_null() } {
        return std::ptr::null_mut();
    }
    if unsafe {
        !(*def).m_slots.is_null()
            || (*def).m_size != -1
            || !(*def).m_traverse.is_null()
            || !(*def).m_clear.is_null()
            || !(*def).m_free.is_null()
    } {
        return std::ptr::null_mut();
    }

    let declared_name = unsafe { CStr::from_ptr((*def).m_name) }.to_string_lossy();
    // PyPy `PyModule_Create2` consumes package_context before allocating the
    // module. Releasing the mutex here also avoids holding it across GC.
    let context = PACKAGE_CONTEXT.lock().unwrap().take();
    let (name, path) = context
        .as_ref()
        .map(|(name, path)| (name.as_str(), Some(path.as_path())))
        .unwrap_or((&declared_name, None));
    let module = pyre_object::w_module_new_managed(name);
    let roots = pyre_object::gc_roots::push_roots();
    let module_slot = pyre_object::gc_roots::shadow_stack_len();
    roots.pin_root(module);

    if unsafe { !(*def).m_doc.is_null() } {
        let doc = unsafe { CStr::from_ptr((*def).m_doc) }.to_string_lossy();
        if !doc.is_empty() {
            let value = pyre_object::w_str_new(&doc);
            let value_slot = pyre_object::gc_roots::shadow_stack_len();
            roots.pin_root(value);
            let module = pyre_object::gc_roots::shadow_stack_get(module_slot);
            crate::module_ns_store(
                unsafe { pyre_object::w_module_get_w_dict(module) },
                "__doc__",
                pyre_object::gc_roots::shadow_stack_get(value_slot),
            );
        }
    }
    if let Some(path) = path {
        let value = crate::gateway::fsdecode_os_str(path.as_os_str());
        let value_slot = pyre_object::gc_roots::shadow_stack_len();
        roots.pin_root(value);
        let module = pyre_object::gc_roots::shadow_stack_get(module_slot);
        crate::module_ns_store(
            unsafe { pyre_object::w_module_get_w_dict(module) },
            "__file__",
            pyre_object::gc_roots::shadow_stack_get(value_slot),
        );
    }
    attach_module(pyre_object::gc_roots::shadow_stack_get(module_slot))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn Py_IncRef(object: *mut CPyObject) {
    if !object.is_null() {
        unsafe { (*object).ob_refcnt += 1 };
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn Py_DecRef(object: *mut CPyObject) {
    if object.is_null() {
        return;
    }
    unsafe { (*object).ob_refcnt -= 1 };
    if unsafe { (*object).ob_refcnt } == REFCNT_FROM_PYRE {
        unsafe { detach_unowned_module_mirror(object) };
        return;
    }
    if unsafe { (*object).ob_refcnt } != 0 {
        return;
    }
    // Type-specific `_Py_Dealloc` is introduced with C-defined types. The
    // first slice never exposes a heap mirror whose zero path is not handled
    // by `detach_unowned_module_mirror` above.
}

/// Force the linker to retain the public C entry points in every native pyre
/// executable. `pyrex/build.rs` additionally exports these names from the main
/// program so a bundle/shared object can resolve them at `dlopen` time.
pub fn ensure_linked() {
    std::hint::black_box(PyModuleDef_Init as *const ());
    std::hint::black_box(PyModule_Create2 as *const ());
    std::hint::black_box(Py_IncRef as *const ());
    std::hint::black_box(Py_DecRef as *const ());
}
