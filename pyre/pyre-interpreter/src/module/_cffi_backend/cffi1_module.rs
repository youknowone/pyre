//! Generated CFFI extension loading — PyPy:
//! `pypy/module/_cffi_backend/cffi1_module.py`.

use std::path::Path;

use pyre_object::PyObjectRef;

use super::{cdlopen, ffi_obj, lib_obj, parse_c_type};
use crate::PyError;

const VERSION_EXPORT: usize = 0x0a03;

/// PyPy `cffi1_module.load_cffi1_module`.
///
/// A generated PyPy-mode CFFI library exports a tiny initializer which
/// replaces `p[0]` with its ABI version and `p[1]` with its immutable type
/// context.  The Python-visible objects remain interpreter-owned; native code
/// only supplies declarations and function addresses.
#[majit_macros::dont_look_inside]
pub fn load_cffi1_module(
    name: &str,
    path: &Path,
    init_address: usize,
) -> Result<PyObjectRef, PyError> {
    type InitFn = unsafe extern "C" fn(*mut *const core::ffi::c_void);

    let init: InitFn = unsafe { core::mem::transmute(init_address) };
    let mut export = [core::ptr::null(); 16];
    export[0] = VERSION_EXPORT as *const core::ffi::c_void;
    // PyPy puts `get_ll_cffi_call_python()` in this slot.  Pyre's generated
    // stdlib modules use ordinary `ffi.callback()` values and have no
    // `extern "Python"` declarations, so there is no generated trampoline
    // to bind here.  A context carrying that flag is rejected below rather
    // than installing a semantic null callback.
    export[1] = core::ptr::null();
    unsafe { init(export.as_mut_ptr()) };

    let version = export[0] as usize as i64;
    if !(cdlopen::VERSION_MIN..=cdlopen::VERSION_MAX).contains(&version) {
        return Err(PyError::new(
            crate::PyErrorKind::ImportError,
            format!(
                "cffi extension module '{name}' uses an unknown version tag {version:#x}. \
                 This module might need a more recent version of PyPy. The current PyPy \
                 provides CFFI {}.",
                super::interp_cffi_backend::VERSION
            ),
        ));
    }
    let src_ctx = export[1].cast::<parse_c_type::TypeContextS>();
    if src_ctx.is_null() {
        return Err(PyError::new(
            crate::PyErrorKind::ImportError,
            format!("cffi extension module '{name}' did not provide a type context"),
        ));
    }
    if unsafe { (*src_ctx).flags & 1 } != 0 {
        return Err(PyError::new(
            crate::PyErrorKind::ImportError,
            format!(
                "cffi extension module '{name}' uses extern \"Python\", which is not supported"
            ),
        ));
    }

    let roots = pyre_object::gc_roots::push_roots();
    let ffi_slot = roots.base();
    let _ = roots.pin_root(ffi_obj::initialize_ffi(
        ffi_obj::ffi_type_object(),
        src_ctx,
    )?);
    let lib_slot = ffi_slot + 1;
    let _ = roots.pin_root(lib_obj::new_lib(
        roots.get(ffi_slot),
        name,
        lib_obj::FLAVOR_STATIC,
        0,
        false,
    )?);
    if !unsafe { (*src_ctx).includes }.is_null() {
        lib_obj::make_includes_from(roots.get(lib_slot), unsafe { (*src_ctx).includes })?;
    }

    let module_slot = lib_slot + 1;
    let _ = roots.pin_root(pyre_object::w_module_new_managed(name));
    let file_slot = module_slot + 1;
    let _ = roots.pin_root(crate::gateway::fsdecode_os_str(path.as_os_str()));
    // `module_ns_store` allocates, so neither the decoded pathname nor the
    // dictionary read off the module survives one as a native local: publish
    // the pathname, and take the dictionary off the rooted module each time.
    let dict = || unsafe { pyre_object::w_module_get_w_dict(roots.get(module_slot)) };
    crate::module_ns_store(dict(), "__file__", roots.get(file_slot));
    crate::module_ns_store(dict(), "ffi", roots.get(ffi_slot));
    crate::module_ns_store(dict(), "lib", roots.get(lib_slot));
    crate::importing::set_sys_module(name, roots.get(module_slot));
    crate::importing::set_sys_module(&format!("{name}.lib"), roots.get(lib_slot));
    Ok(roots.get(module_slot))
}
