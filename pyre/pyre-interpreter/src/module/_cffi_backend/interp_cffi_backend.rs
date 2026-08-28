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

    // `moduledef.py interpleveldefs` — the fixed-arity entry points first.
    for (name, f, arity) in [
        (
            "new_primitive_type",
            super::func::new_primitive_type as crate::gateway::BuiltinCodeFn,
            1u16,
        ),
        ("new_pointer_type", super::func::new_pointer_type, 1),
        ("new_array_type", super::func::new_array_type, 2),
        ("new_void_type", super::func::new_void_type, 0),
        ("cast", super::func::cast, 2),
        ("typeof", super::func::typeof_, 1),
        ("sizeof", super::func::sizeof, 1),
        ("alignof", super::func::alignof, 1),
        ("getcname", super::func::getcname, 2),
        ("unpack", super::func::unpack, 2),
        ("memmove", super::func::memmove, 3),
        ("release", super::func::release, 1),
        ("_get_types", super::func::get_types, 0),
        ("_get_common_types", get_common_types, 1),
        ("new_struct_type", super::func::new_struct_type, 1),
        ("new_union_type", super::func::new_union_type, 1),
        ("new_enum_type", super::func::new_enum_type, 4),
    ] {
        crate::module_ns_store(
            ns,
            name,
            crate::gateway::with_module(
                MODULE,
                crate::make_module_builtin_function_with_arity(name, f, arity),
            ),
        );
    }
    // `newp(ctype, init=None)`, `string(cdata, maxlen=-1)`,
    // `typeoffsetof(ctype, field_or_index, following=0)` and
    // `rawaddressof(ctype, cdata, offset=0)` all carry a default.
    for (name, f) in [
        ("newp", super::func::newp as crate::gateway::BuiltinCodeFn),
        ("string", super::func::string),
        ("typeoffsetof", super::func::typeoffsetof),
        ("rawaddressof", super::func::rawaddressof),
        ("complete_struct_or_union", super::func::complete_struct_or_union),
        ("new_function_type", super::func::new_function_type),
        ("load_library", super::func::load_library),
    ] {
        crate::module_ns_store(
            ns,
            name,
            crate::gateway::with_module(MODULE, crate::make_module_builtin_function(name, f)),
        );
    }
    // The types `moduledef.py` publishes.
    crate::module_ns_store(ns, "CType", super::ctypeobj::ctype_type());
    crate::module_ns_store(ns, "_CDataBase", super::cdataobj::cdata_type());
    crate::module_ns_store(
        ns,
        "__CData_iterator",
        super::ctypearray::cdata_iter_type(),
    );
    crate::module_ns_store(ns, "CField", super::ctypestruct::cfield_type());
    crate::module_ns_store(ns, "CLibrary", super::libraryobj::clibrary_type());
}

const MODULE: &str = "_cffi_backend";

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
pub fn default_abi() -> u32 {
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
pub fn default_abi() -> u32 {
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
