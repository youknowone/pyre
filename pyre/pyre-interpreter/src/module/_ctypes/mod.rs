//! _ctypes module — PyPy: `pypy/module/_rawffi/` plus `lib_pypy/_ctypes/`.
//!
//! `interp_ctypes` holds the module-level surface: library handles (dlopen /
//! dlsym / dlclose), size/align queries, and the raw-memory helpers.  The type
//! machinery is split across the submodules — `stginfo` carries a ctypes type's
//! layout, `metaclass` builds the types and their fields, `cdata` holds the
//! scalar instance buffer, and `funcptr` marshals and performs the foreign
//! call.  The submodules need `host_env`; without it the module is limited to
//! the placeholder surface at the foot of `interp_ctypes`.

crate::pyre_module_init!(interp_ctypes);

/// Store into a builtin type's namespace — the dict `make_builtin_type` hands
/// its init closure.  The type-namespace sibling of `module_ns_store`.
#[cfg(all(any(unix, windows), feature = "host_env"))]
fn type_ns_store(ns: pyre_object::PyObjectRef, name: &str, value: pyre_object::PyObjectRef) {
    unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(ns, name, value) }
}

#[cfg(all(any(unix, windows), feature = "host_env"))]
pub mod callbacks;
#[cfg(all(any(unix, windows), feature = "host_env"))]
pub mod cdata;
#[cfg(all(windows, feature = "host_env"))]
mod com;
#[cfg(all(any(unix, windows), feature = "host_env"))]
pub mod funcptr;
#[cfg(all(any(unix, windows), feature = "host_env"))]
pub mod metaclass;
#[cfg(all(any(unix, windows), feature = "host_env"))]
mod seh;
#[cfg(all(any(unix, windows), feature = "host_env"))]
pub mod stginfo;
