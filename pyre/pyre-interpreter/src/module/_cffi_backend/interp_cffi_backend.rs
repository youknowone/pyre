//! `_cffi_backend`'s module surface — PyPy:
//! `pypy/module/_cffi_backend/moduledef.py`.

use super::parse_c_type;

/// `pypy/module/_cffi_backend/__init__.py:VERSION`.  cffi's Python half
/// compares this against its own version and refuses to run on a mismatch, so
/// it tracks the vendored `lib_pypy/cffi` rather than anything of pyre's.
pub const VERSION: &str = "1.18.0.dev0";

pub fn register_module(ns: pyre_object::PyObjectRef) {
    crate::module_ns_store(ns, "__version__", pyre_object::w_str_new(VERSION));

    // `clibffi.FFI_DEFAULT_ABI`.  `FFI_CDECL` is the win32 spelling of the
    // same value and is defined on every platform.
    let abi = default_abi() as i64;
    crate::module_ns_store(ns, "FFI_DEFAULT_ABI", pyre_object::w_int_new(abi));
    crate::module_ns_store(ns, "FFI_CDECL", pyre_object::w_int_new(abi));

    register_rtld_constants(ns);

    crate::module_ns_store(
        ns,
        "_get_common_types",
        crate::make_builtin_function_with_arity("_get_common_types", get_common_types, 1),
    );
}

/// `clibffi.FFI_DEFAULT_ABI`.
#[cfg(all(
    any(
        target_os = "linux",
        target_os = "macos",
        target_os = "windows",
        target_os = "android"
    ),
    not(any(target_env = "musl", target_env = "sgx"))
))]
fn default_abi() -> u32 {
    libffi::raw::ffi_abi_FFI_DEFAULT_ABI
}

#[cfg(not(all(
    any(
        target_os = "linux",
        target_os = "macos",
        target_os = "windows",
        target_os = "android"
    ),
    not(any(target_env = "musl", target_env = "sgx"))
)))]
fn default_abi() -> u32 {
    // No libffi on this target, so no foreign call can be made at all; the
    // constant still has to exist because cffi's Python half reads it at
    // import time.
    0
}

/// `moduledef.get_dict_rtld_constants` — the names `rdynload` found, with
/// the four cffi always needs defaulted to 0 where the platform lacks them.
fn register_rtld_constants(ns: pyre_object::PyObjectRef) {
    #[cfg(unix)]
    let found: &[(&str, i64)] = &[
        ("RTLD_LAZY", libc::RTLD_LAZY as i64),
        ("RTLD_NOW", libc::RTLD_NOW as i64),
        ("RTLD_GLOBAL", libc::RTLD_GLOBAL as i64),
        ("RTLD_LOCAL", libc::RTLD_LOCAL as i64),
        #[cfg(any(target_os = "linux", target_os = "macos", target_os = "android"))]
        ("RTLD_NODELETE", libc::RTLD_NODELETE as i64),
        #[cfg(any(target_os = "linux", target_os = "macos", target_os = "android"))]
        ("RTLD_NOLOAD", libc::RTLD_NOLOAD as i64),
        #[cfg(any(target_os = "linux", target_os = "android"))]
        ("RTLD_DEEPBIND", libc::RTLD_DEEPBIND as i64),
    ];
    #[cfg(not(unix))]
    let found: &[(&str, i64)] = &[
        ("RTLD_LAZY", 0),
        ("RTLD_NOW", 0),
        ("RTLD_GLOBAL", 0),
        ("RTLD_LOCAL", 0),
    ];
    for (name, value) in found {
        crate::module_ns_store(ns, name, pyre_object::w_int_new(*value));
    }
}

/// `func._get_common_types(dict)` — fill the mapping cffi's parser consults
/// for the type names every platform spells the same way.
fn get_common_types(
    args: &[pyre_object::PyObjectRef],
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    let w_dict = args
        .first()
        .copied()
        .ok_or_else(|| crate::PyError::type_error("_get_common_types() missing dict argument"))?;
    let roots = pyre_object::gc_roots::push_roots();
    let _ = roots.pin_root(w_dict);
    let slot = roots.base();
    let mut index = 0;
    while let Some((key, value)) = parse_c_type::enum_common_types(index) {
        let w_value = pyre_object::w_str_new(value);
        // SAFETY: cffi's Python half only ever passes a real dict here
        // (`cffi/commontypes.py:_get_common_types(_CACHE)`).
        unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str(roots.get(slot), key, w_value);
        }
        index += 1;
    }
    Ok(pyre_object::w_none())
}
