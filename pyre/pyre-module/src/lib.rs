//! Optional builtin modules — `pypy/module/` (non-essential subset).
//!
//! PyPy classifies builtin modules into three tiers:
//!
//! | Tier       | PyPy config          | pyre location          |
//! |------------|----------------------|------------------------|
//! | Essential  | always loaded        | `pyre-interpreter`     |
//! | Default    | on by default        | `pyre-module` (here)   |
//! | Working    | opt-in               | `pyre-module` (here)   |
//!
//! Essential modules (`__builtin__`, `sys`) live in `pyre-interpreter`
//! because they are inseparable from the interpreter bootstrap.
//!
//! Everything else belongs here.  Modules will be migrated from
//! `pyre-interpreter/src/module/` as they grow.

/// Adapt a `W_Root` sweep hook (`fn(PyObjectRef)`) to the collector's
/// address-taking `DestructorFn`.
macro_rules! gc_destructor {
    ($hook:path) => {{
        unsafe fn destructor(obj_addr: usize) {
            unsafe { $hook(obj_addr as pyre_object::PyObjectRef) }
        }
        destructor as majit_gc::trace::DestructorFn
    }};
}

/// Keep the `module::` path prefix so harvested hint paths and
/// `should_lower_module` stay `module::<name>` after the crate split.
pub mod module;

/// Register default/working modules into the interpreter builtin table.
///
/// The interpreter does not depend on this crate. The final binary calls
/// [`register`] before `install_builtin_modules`.
pub fn install_optional_modules() {
    pyre_interpreter::importing::register_builtin_module("_abc", module::_abc::init);
    pyre_interpreter::importing::register_builtin_module("_bisect", module::_bisect::init);
    pyre_interpreter::importing::register_builtin_module("_blake2", module::_blake2::init);
    #[cfg(all(
        feature = "full",
        feature = "host_env",
        not(feature = "sandbox"),
        not(target_arch = "wasm32")
    ))]
    pyre_interpreter::importing::register_builtin_module(
        "_cffi_backend",
        module::_cffi_backend::init,
    );
    #[cfg(all(feature = "full", not(feature = "sandbox")))]
    pyre_interpreter::importing::register_builtin_module("_ctypes", module::_ctypes::init);
    pyre_interpreter::importing::register_builtin_module("_bz2", module::_bz2::init);
    pyre_interpreter::importing::register_builtin_module("_csv", module::_csv::init);
    pyre_interpreter::importing::register_builtin_module(
        "_contextvars",
        module::_contextvars::init,
    );
    pyre_interpreter::importing::register_builtin_module("_functools", module::_functools::init);
    pyre_interpreter::importing::register_builtin_module("_codecs_cn", module::_codecs_cn::init);
    pyre_interpreter::importing::register_builtin_module("_codecs_hk", module::_codecs_hk::init);
    pyre_interpreter::importing::register_builtin_module(
        "_codecs_iso2022",
        module::_codecs_iso2022::init,
    );
    pyre_interpreter::importing::register_builtin_module("_codecs_jp", module::_codecs_jp::init);
    pyre_interpreter::importing::register_builtin_module("_codecs_kr", module::_codecs_kr::init);
    pyre_interpreter::importing::register_builtin_module("_codecs_tw", module::_codecs_tw::init);
    pyre_interpreter::importing::register_builtin_module("_hashlib", module::_hashlib::init);
    pyre_interpreter::importing::register_builtin_module("_heapq", module::_heapq::init);
    pyre_interpreter::importing::register_builtin_module("_json", module::_json::init);
    pyre_interpreter::importing::register_builtin_module("_lsprof", module::_lsprof::init);
    pyre_interpreter::importing::register_builtin_module("_lzma", module::_lzma::init);
    pyre_interpreter::importing::register_builtin_module(
        "_immutables_map",
        module::_immutables_map::init,
    );
    pyre_interpreter::importing::register_builtin_module(
        "_multibytecodec",
        module::_multibytecodec::init,
    );
    #[cfg(not(feature = "sandbox"))]
    pyre_interpreter::importing::register_builtin_module(
        "_multiprocessing",
        module::_multiprocessing::init,
    );
    pyre_interpreter::importing::register_builtin_module("_opcode", module::_opcode::init);
    #[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
    pyre_interpreter::importing::register_builtin_module("_overlapped", module::_overlapped::init);
    #[cfg(all(unix, not(feature = "sandbox")))]
    pyre_interpreter::importing::register_builtin_module("_posixshmem", module::_posixshmem::init);
    #[cfg(all(unix, not(feature = "sandbox")))]
    pyre_interpreter::importing::register_builtin_module(
        "_posixsubprocess",
        module::_posixsubprocess::init,
    );
    pyre_interpreter::importing::register_builtin_module(
        "_pypy_generic_alias",
        module::_pypy_generic_alias::init,
    );
    pyre_interpreter::importing::register_builtin_module("_queue", module::_queue::init);
    #[cfg(all(feature = "full", not(feature = "sandbox")))]
    pyre_interpreter::importing::register_builtin_module("_socket", module::_socket::init);
    #[cfg(all(
        feature = "full",
        not(target_arch = "wasm32"),
        not(feature = "sandbox")
    ))]
    pyre_interpreter::importing::register_builtin_module("_ssl", module::_ssl::init);
    // Frozen importlib imports `_stat` while bootstrapping a sandbox that
    // mounts no stdlib files, so this stays an unconditional builtin.
    pyre_interpreter::importing::register_builtin_module("_stat", module::_stat::init);
    pyre_interpreter::importing::register_builtin_module("_statistics", module::_statistics::init);
    pyre_interpreter::importing::register_builtin_module(
        "_suggestions",
        module::_suggestions::init,
    );
    pyre_interpreter::importing::register_builtin_module("_symtable", module::_symtable::init);
    pyre_interpreter::importing::register_builtin_module("_template", module::_template::init);
    pyre_interpreter::importing::register_builtin_module("_tokenize", module::_tokenize::init);
    pyre_interpreter::importing::register_builtin_module("_typing", module::_typing::init);
    #[cfg(windows)]
    pyre_interpreter::importing::register_builtin_module("_winapi", module::_winapi::init);
    #[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
    pyre_interpreter::importing::register_builtin_module("_wmi", module::_wmi::init);
    #[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
    pyre_interpreter::importing::register_builtin_module("_uuid", module::_uuid::init);
    #[cfg(target_os = "macos")]
    pyre_interpreter::importing::register_builtin_module("_scproxy", module::_scproxy::init);
    pyre_interpreter::importing::register_builtin_module("atexit", module::atexit::init);
    pyre_interpreter::importing::register_builtin_module("binascii", module::binascii::init);
    pyre_interpreter::importing::register_builtin_module("cmath", module::cmath::init);
    pyre_interpreter::importing::register_builtin_module("errno", module::errno::init);
    #[cfg(all(not(target_arch = "wasm32"), not(feature = "sandbox")))]
    pyre_interpreter::importing::register_builtin_module(
        "faulthandler",
        module::faulthandler::init,
    );
    pyre_interpreter::importing::register_builtin_module("math", module::math::init);
    #[cfg(all(windows, feature = "host_env"))]
    pyre_interpreter::importing::register_builtin_module("msvcrt", module::msvcrt::init);
    pyre_interpreter::importing::register_builtin_module("pypyjit", module::pypyjit::init);
    #[cfg(all(
        feature = "full",
        not(target_arch = "wasm32"),
        feature = "host_env",
        not(feature = "sandbox")
    ))]
    pyre_interpreter::importing::register_builtin_module("mmap", module::mmap::init);
    #[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
    pyre_interpreter::importing::register_builtin_module("fcntl", module::fcntl::init);
    #[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
    pyre_interpreter::importing::register_builtin_module("grp", module::grp::init);
    #[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
    pyre_interpreter::importing::register_builtin_module("pwd", module::pwd::init);
    pyre_interpreter::importing::register_builtin_module("pyexpat", module::pyexpat::init);
    #[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
    pyre_interpreter::importing::register_builtin_module("resource", module::resource::init);
    #[cfg(all(feature = "full", not(feature = "sandbox")))]
    pyre_interpreter::importing::register_builtin_module("select", module::select::init);
    #[cfg(all(unix, not(feature = "sandbox")))]
    pyre_interpreter::importing::register_builtin_module("syslog", module::syslog::init);
    #[cfg(all(unix, not(feature = "sandbox")))]
    pyre_interpreter::importing::register_builtin_module("termios", module::termios::init);
    #[cfg(windows)]
    pyre_interpreter::importing::register_builtin_module("winreg", module::winreg::init);
    #[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
    pyre_interpreter::importing::register_builtin_module("winsound", module::winsound::init);
    pyre_interpreter::importing::register_builtin_module("unicodedata", module::unicodedata::init);
    pyre_interpreter::importing::register_builtin_module("zlib", module::zlib::init);
}

/// Immortal `#[pyre_class]` types allocated through `allocate`.  The
/// collector never walks them, so `build_gc` only registers their
/// `w_class` offset.  `select` is compiled out of a sandbox build
/// (`module/mod.rs`'s `pub mod select`), so its descriptors carry that
/// gate too.
pub fn all_immortal_w_class_only_descriptors()
-> Vec<&'static pyre_object::lltype::PyreClassDescriptor> {
    #[allow(unused_imports)]
    use pyre_object::lltype::PyreClassPyTypeOf;
    vec![
        #[cfg(all(feature = "full", unix, feature = "host_env", not(feature = "sandbox")))]
        <module::select::interp_select::Poll as PyreClassPyTypeOf>::DESCRIPTOR,
        #[cfg(all(
            feature = "full",
            target_os = "macos",
            feature = "host_env",
            not(feature = "sandbox")
        ))]
        <module::select::interp_kqueue::W_Kqueue as PyreClassPyTypeOf>::DESCRIPTOR,
        #[cfg(all(
            feature = "full",
            target_os = "macos",
            feature = "host_env",
            not(feature = "sandbox")
        ))]
        <module::select::interp_kevent::W_Kevent as PyreClassPyTypeOf>::DESCRIPTOR,
    ]
}

/// Install [`install_optional_modules`] as the interpreter's optional-module hook.
///
/// Also run from a constructor so a binary that links this crate publishes
/// the rclass aliases before any `init_subclass_ranges` OnceLock in a
/// parallel test can freeze an incomplete census. PyPy has the same
/// modules in the translated program from process start.
#[::ctor::ctor(unsafe)]
fn register_on_load() {
    register();
}

fn hook_mini_buffer_params(obj: pyre_object::PyObjectRef) -> Option<(*mut u8, usize)> {
    #[cfg(all(
        feature = "full",
        feature = "host_env",
        not(feature = "sandbox"),
        not(target_arch = "wasm32")
    ))]
    {
        return module::_cffi_backend::cbuffer::mini_buffer_params(obj);
    }
    #[cfg(not(all(
        feature = "full",
        feature = "host_env",
        not(feature = "sandbox"),
        not(target_arch = "wasm32")
    )))]
    {
        let _ = obj;
        None
    }
}

fn hook_cffi_finalizer_kind(obj: pyre_object::PyObjectRef) -> Option<bool> {
    #[cfg(all(
        feature = "full",
        feature = "host_env",
        not(feature = "sandbox"),
        not(target_arch = "wasm32")
    ))]
    {
        return module::_cffi_backend::cdataobj::ec_finalizer_kind(obj);
    }
    #[cfg(not(all(
        feature = "full",
        feature = "host_env",
        not(feature = "sandbox"),
        not(target_arch = "wasm32")
    )))]
    {
        let _ = obj;
        None
    }
}

fn hook_run_cffi_finalize(obj: pyre_object::PyObjectRef) {
    #[cfg(all(
        feature = "full",
        feature = "host_env",
        not(feature = "sandbox"),
        not(target_arch = "wasm32")
    ))]
    {
        module::_cffi_backend::cdataobj::run_ec_finalize(obj);
    }
    #[cfg(not(all(
        feature = "full",
        feature = "host_env",
        not(feature = "sandbox"),
        not(target_arch = "wasm32")
    )))]
    {
        let _ = obj;
    }
}

fn hook_close_cffi_fileobj(obj: pyre_object::PyObjectRef) {
    #[cfg(all(
        feature = "full",
        feature = "host_env",
        not(feature = "sandbox"),
        not(target_arch = "wasm32")
    ))]
    {
        module::_cffi_backend::ctypeptr::close_cffi_fileobj(obj);
    }
    #[cfg(not(all(
        feature = "full",
        feature = "host_env",
        not(feature = "sandbox"),
        not(target_arch = "wasm32")
    )))]
    {
        let _ = obj;
    }
}

fn hook_ctypes_buffer_view(
    obj: pyre_object::PyObjectRef,
) -> Option<(
    pyre_object::PyObjectRef,
    usize,
    usize,
    String,
    usize,
    Vec<usize>,
)> {
    #[cfg(all(
        feature = "full",
        any(unix, windows),
        feature = "host_env",
        not(feature = "sandbox")
    ))]
    {
        return module::_ctypes::cdata::cdata_buffer_view(obj);
    }
    #[cfg(not(all(
        feature = "full",
        any(unix, windows),
        feature = "host_env",
        not(feature = "sandbox")
    )))]
    {
        let _ = obj;
        None
    }
}

fn hook_ctypes_bytes_object(obj: pyre_object::PyObjectRef) -> Option<pyre_object::PyObjectRef> {
    #[cfg(all(
        feature = "full",
        any(unix, windows),
        feature = "host_env",
        not(feature = "sandbox")
    ))]
    {
        return module::_ctypes::cdata::cdata_bytes_object(obj);
    }
    #[cfg(not(all(
        feature = "full",
        any(unix, windows),
        feature = "host_env",
        not(feature = "sandbox")
    )))]
    {
        let _ = obj;
        None
    }
}

fn hook_ctypes_array_instance(obj: pyre_object::PyObjectRef) -> bool {
    #[cfg(all(
        feature = "full",
        any(unix, windows),
        feature = "host_env",
        not(feature = "sandbox")
    ))]
    {
        return module::_ctypes::metaclass::is_array_instance(obj);
    }
    #[cfg(not(all(
        feature = "full",
        any(unix, windows),
        feature = "host_env",
        not(feature = "sandbox")
    )))]
    {
        let _ = obj;
        false
    }
}

fn hook_ctypes_pointer_instance(obj: pyre_object::PyObjectRef) -> bool {
    #[cfg(all(
        feature = "full",
        any(unix, windows),
        feature = "host_env",
        not(feature = "sandbox")
    ))]
    {
        return module::_ctypes::metaclass::is_pointer_instance(obj);
    }
    #[cfg(not(all(
        feature = "full",
        any(unix, windows),
        feature = "host_env",
        not(feature = "sandbox")
    )))]
    {
        let _ = obj;
        false
    }
}

fn hook_load_cffi1_module(
    name: &str,
    path: &std::path::Path,
    init_address: usize,
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    #[cfg(all(
        feature = "full",
        feature = "host_env",
        not(feature = "sandbox"),
        not(target_arch = "wasm32")
    ))]
    {
        return module::_cffi_backend::cffi1_module::load_cffi1_module(name, path, init_address);
    }
    #[cfg(not(all(
        feature = "full",
        feature = "host_env",
        not(feature = "sandbox"),
        not(target_arch = "wasm32")
    )))]
    {
        let _ = (name, path, init_address);
        Err(pyre_interpreter::PyError::system_error(
            "_cffi_backend is not available",
        ))
    }
}

pub fn register() {
    pyre_interpreter::importing::set_optional_module_hooks(
        pyre_interpreter::importing::OptionalModuleHooks {
            install_modules: install_optional_modules,
            walk_global_roots: walk_optional_global_roots,
            walk_prebuilt_slots: |fwd| {
                module::_csv::walk_csv_state_gc(fwd);
            },
            subclass_range_aliases: optional_subclass_range_aliases,
            publish_fnaddrs: publish_optional_fnaddrs,
            mini_buffer_params: hook_mini_buffer_params,
            cffi_finalizer_kind: hook_cffi_finalizer_kind,
            run_cffi_finalize: hook_run_cffi_finalize,
            close_cffi_fileobj: hook_close_cffi_fileobj,
            load_cffi1_module: hook_load_cffi1_module,
            ctypes_buffer_view: hook_ctypes_buffer_view,
            ctypes_bytes_object: hook_ctypes_bytes_object,
            ctypes_array_instance: hook_ctypes_array_instance,
            ctypes_pointer_instance: hook_ctypes_pointer_instance,
            gc_types: module_gc_types,
            immortal_w_class_only_descriptors: all_immortal_w_class_only_descriptors,
            libffi_cif_shape: hook_libffi_cif_shape,
            math_builtin_name: module::math::interp_math::math_builtin_name,
            math1_gamma_result_finite: module::math::interp_math::math1_gamma_result_finite,
        },
    );
}

/// `jit_libffi.py`'s reading of a `CIF_DESCRIPTION` block for the tracer.
unsafe fn hook_libffi_cif_shape(
    cif_description: usize,
) -> Option<pyre_interpreter::importing::LibffiCifShape> {
    #[cfg(all(
        feature = "full",
        feature = "host_env",
        not(feature = "sandbox"),
        not(target_arch = "wasm32")
    ))]
    {
        use module::_cffi_backend::jit_libffi::{self, types};
        use pyre_interpreter::importing::{LibffiCifShape, LibffiType};
        // `getkind(0)` is `OTHER`; there is no record to size.
        let read = |ffi_type: usize| LibffiType {
            kind: unsafe { types::getkind(ffi_type) } as u8,
            size: if ffi_type == 0 {
                0
            } else {
                unsafe { types::getsize(ffi_type) }
            },
        };
        let nargs = unsafe { jit_libffi::nargs(cif_description) };
        return Some(LibffiCifShape {
            rtype: read(unsafe { jit_libffi::rtype(cif_description) }),
            args: (0..nargs)
                .map(|i| {
                    (
                        read(unsafe { jit_libffi::atype(cif_description, i) }),
                        unsafe { jit_libffi::exchange_arg(cif_description, i) },
                    )
                })
                .collect(),
        });
    }
    #[cfg(not(all(
        feature = "full",
        feature = "host_env",
        not(feature = "sandbox"),
        not(target_arch = "wasm32")
    )))]
    {
        let _ = cif_description;
        None
    }
}

/// The GC types this crate's modules own, in `build_gc` registration order
/// within each `ModuleGcAnchor`.
fn module_gc_types() -> Vec<pyre_interpreter::importing::ModuleGcType> {
    let mut types = Vec::new();
    module::_tokenize::gc_types(&mut types);
    module::_functools::gc_types(&mut types);
    module::unicodedata::gc_types(&mut types);
    module::_json::gc_types(&mut types);
    module::_hashlib::gc_types(&mut types);
    module::zlib::gc_types(&mut types);
    module::_bz2::gc_types(&mut types);
    module::_lzma::gc_types(&mut types);
    module::_lsprof::gc_types(&mut types);
    module::_queue::gc_types(&mut types);
    #[cfg(all(
        feature = "full",
        not(target_arch = "wasm32"),
        not(feature = "sandbox")
    ))]
    module::_ssl::gc_types(&mut types);
    #[cfg(all(
        feature = "full",
        not(target_arch = "wasm32"),
        feature = "host_env",
        not(feature = "sandbox")
    ))]
    module::mmap::gc_types(&mut types);
    #[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
    module::_overlapped::gc_types(&mut types);
    #[cfg(windows)]
    module::_winapi::gc_types(&mut types);
    #[cfg(all(
        feature = "full",
        feature = "host_env",
        not(feature = "sandbox"),
        not(target_arch = "wasm32")
    ))]
    module::_cffi_backend::gc_types(&mut types);
    types
}

/// `ll_math.py` C llexternals. The front retargets the Opaque `f64`
/// inherent methods `ll_math::f64_method_llexternal` names onto these
/// paths; the raising `ll_math_*` wrappers stay around them.
fn publish_optional_fnaddrs(entries: &mut Vec<(&'static str, i64)>) {
    use module::math::interp_math as math;

    fn pair(
        entries: &mut Vec<(&'static str, i64)>,
        module_path: &'static str,
        root_path: &'static str,
        fnptr: *const (),
    ) {
        let addr = fnptr as usize as i64;
        if addr != 0 {
            entries.push((module_path, addr));
            entries.push((root_path, addr));
        }
    }

    for (module_path, root_path, fnptr) in [
        (
            "ll_math::math_hypot",
            "math_hypot",
            math::jit_math_hypot as *const (),
        ),
        (
            "ll_math::math_atan2",
            "math_atan2",
            math::jit_math_atan2 as *const (),
        ),
        (
            "ll_math::math_copysign",
            "math_copysign",
            math::jit_math_copysign as *const (),
        ),
        (
            "ll_math::math_floor",
            "math_floor",
            math::jit_math_floor_raw as *const (),
        ),
        (
            "ll_math::math_ceil",
            "math_ceil",
            math::jit_math_ceil_raw as *const (),
        ),
        (
            "ll_math::math_log",
            "math_log",
            math::jit_math_log_raw as *const (),
        ),
        (
            "ll_math::math_log10",
            "math_log10",
            math::jit_math_log10_raw as *const (),
        ),
        (
            "ll_math::math_log1p",
            "math_log1p",
            math::jit_math_log1p_raw as *const (),
        ),
        (
            "ll_math::math_exp",
            "math_exp",
            math::jit_math_exp_raw as *const (),
        ),
        (
            "ll_math::math_exp2",
            "math_exp2",
            math::jit_math_exp2_raw as *const (),
        ),
        (
            "ll_math::math_expm1",
            "math_expm1",
            math::jit_math_expm1_raw as *const (),
        ),
        (
            "ll_math::math_pow",
            "math_pow",
            math::jit_math_pow_raw as *const (),
        ),
        (
            "ll_math::math_sqrt",
            "math_sqrt",
            math::jit_math_sqrt_raw as *const (),
        ),
        (
            "ll_math::math_cbrt",
            "math_cbrt",
            math::jit_math_cbrt_raw as *const (),
        ),
        (
            "ll_math::math_sin",
            "math_sin",
            math::jit_math_sin_raw as *const (),
        ),
        (
            "ll_math::math_cos",
            "math_cos",
            math::jit_math_cos_raw as *const (),
        ),
        (
            "ll_math::math_tan",
            "math_tan",
            math::jit_math_tan_raw as *const (),
        ),
        (
            "ll_math::math_asin",
            "math_asin",
            math::jit_math_asin_raw as *const (),
        ),
        (
            "ll_math::math_acos",
            "math_acos",
            math::jit_math_acos_raw as *const (),
        ),
        (
            "ll_math::math_atan",
            "math_atan",
            math::jit_math_atan_raw as *const (),
        ),
        (
            "ll_math::math_sinh",
            "math_sinh",
            math::jit_math_sinh_raw as *const (),
        ),
        (
            "ll_math::math_cosh",
            "math_cosh",
            math::jit_math_cosh_raw as *const (),
        ),
        (
            "ll_math::math_tanh",
            "math_tanh",
            math::jit_math_tanh_raw as *const (),
        ),
        (
            "ll_math::math_asinh",
            "math_asinh",
            math::jit_math_asinh_raw as *const (),
        ),
        (
            "ll_math::math_acosh",
            "math_acosh",
            math::jit_math_acosh_raw as *const (),
        ),
        (
            "ll_math::math_atanh",
            "math_atanh",
            math::jit_math_atanh_raw as *const (),
        ),
        (
            "ll_math::math_fmod",
            "math_fmod",
            math::jit_math_fmod_raw as *const (),
        ),
    ] {
        pair(entries, module_path, root_path, fnptr);
    }

    fn single(entries: &mut Vec<(&'static str, i64)>, path: &'static str, fnptr: *const ()) {
        let addr = fnptr as usize as i64;
        if addr != 0 {
            entries.push((path, addr));
        }
    }

    // `%` over two floats: `lloperation.py` has no `float_mod`, so the
    // codewriter lowers it to a residual call of this name carrying the C
    // `fmod` signature rather than the raising wrapper's.
    single(
        entries,
        "ll_math_fmod",
        math::jit_math_fmod_raw as *const (),
    );

    // `pymath` is outside the extraction set, so every call of it reaches the
    // artifact as an un-lowerable target.  `ulp` takes and returns one float,
    // which the residual-call ABI carries, so binding its real address makes
    // the call executable; the rest of the family returns `Result<f64, _>`,
    // which is wider than a result slot, and stays unpublished.
    single(
        entries,
        "pymath::math::misc::ulp",
        pymath::math::ulp as *const (),
    );
    #[cfg(all(
        feature = "full",
        not(target_arch = "wasm32"),
        feature = "host_env",
        not(feature = "sandbox")
    ))]
    {
        let mmap_type: fn() -> pyre_object::PyObjectRef = module::mmap::interp_mmap::mmap_type;
        let addr = mmap_type as *const () as usize as i64;
        if addr != 0 {
            entries.push((
                "pyre_interpreter::module::mmap::interp_mmap::mmap_type",
                addr,
            ));
            entries.push(("pyre_interpreter::mmap_type", addr));
            entries.push(("pyre_module::module::mmap::interp_mmap::mmap_type", addr));
        }
    }
    #[cfg(all(
        feature = "full",
        feature = "host_env",
        not(feature = "sandbox"),
        not(target_arch = "wasm32")
    ))]
    {
        use module::_cffi_backend::{cdataobj, cerrno, ctypefunc, jit_libffi, misc};
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::cdataobj::raw_malloc_varsize_char",
            cdataobj::raw_malloc_varsize_char as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::cdataobj::raw_malloc_varsize_char",
            cdataobj::raw_malloc_varsize_char as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::cdataobj::raw_malloc_varsize_zero",
            cdataobj::raw_malloc_varsize_zero as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::cdataobj::raw_malloc_varsize_zero",
            cdataobj::raw_malloc_varsize_zero as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::cdataobj::raw_free",
            cdataobj::raw_free as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::cdataobj::raw_free",
            cdataobj::raw_free as *const (),
        );
        // When the frontend inlines `raw_malloc_varsize_char` / `raw_free`
        // the residual names the C leaf (`libc::unix::malloc`). Bind those
        // paths to the same helpers so the descent scan does not see a
        // symbolic hash (`support.py _ll_1_raw_malloc_varsize`).
        single(
            entries,
            "libc::unix::malloc",
            cdataobj::raw_malloc_varsize_char as *const (),
        );
        single(
            entries,
            "libc::malloc",
            cdataobj::raw_malloc_varsize_char as *const (),
        );
        single(
            entries,
            "libc::unix::free",
            cdataobj::raw_free as *const (),
        );
        single(entries, "libc::free", cdataobj::raw_free as *const ());
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::cerrno::errno_before",
            cerrno::errno_before as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::cerrno::errno_before",
            cerrno::errno_before as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::cerrno::errno_after",
            cerrno::errno_after as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::cerrno::errno_after",
            cerrno::errno_after as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::ctypefunc::get_mustfree_flag",
            ctypefunc::get_mustfree_flag as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::ctypefunc::get_mustfree_flag",
            ctypefunc::get_mustfree_flag as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::jit_libffi::imp::exchange_size",
            jit_libffi::exchange_size as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::jit_libffi::imp::exchange_size",
            jit_libffi::exchange_size as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::jit_libffi::exchange_size",
            jit_libffi::exchange_size as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::jit_libffi::exchange_size",
            jit_libffi::exchange_size as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::jit_libffi::imp::exchange_result",
            jit_libffi::exchange_result as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::jit_libffi::imp::exchange_result",
            jit_libffi::exchange_result as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::jit_libffi::exchange_result",
            jit_libffi::exchange_result as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::jit_libffi::exchange_result",
            jit_libffi::exchange_result as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::jit_libffi::imp::exchange_arg",
            jit_libffi::exchange_arg as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::jit_libffi::imp::exchange_arg",
            jit_libffi::exchange_arg as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::jit_libffi::exchange_arg",
            jit_libffi::exchange_arg as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::jit_libffi::exchange_arg",
            jit_libffi::exchange_arg as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::jit_libffi::imp::rtype",
            jit_libffi::rtype as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::jit_libffi::imp::rtype",
            jit_libffi::rtype as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::jit_libffi::rtype",
            jit_libffi::rtype as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::jit_libffi::rtype",
            jit_libffi::rtype as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::jit_libffi::imp::nargs",
            jit_libffi::nargs as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::jit_libffi::imp::nargs",
            jit_libffi::nargs as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::jit_libffi::nargs",
            jit_libffi::nargs as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::jit_libffi::nargs",
            jit_libffi::nargs as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::jit_libffi::imp::types::getkind",
            jit_libffi::types::getkind as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::jit_libffi::imp::types::getkind",
            jit_libffi::types::getkind as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::jit_libffi::types::getkind",
            jit_libffi::types::getkind as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::jit_libffi::types::getkind",
            jit_libffi::types::getkind as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::jit_libffi::imp::types::getsize",
            jit_libffi::types::getsize as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::jit_libffi::imp::types::getsize",
            jit_libffi::types::getsize as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::jit_libffi::types::getsize",
            jit_libffi::types::getsize as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::jit_libffi::types::getsize",
            jit_libffi::types::getsize as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::jit_libffi::imp::jit_ffi_call_impl_int",
            jit_libffi::jit_ffi_call_impl_int as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::jit_libffi::imp::jit_ffi_call_impl_int",
            jit_libffi::jit_ffi_call_impl_int as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::jit_libffi::jit_ffi_call_impl_int",
            jit_libffi::jit_ffi_call_impl_int as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::jit_libffi::jit_ffi_call_impl_int",
            jit_libffi::jit_ffi_call_impl_int as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::jit_libffi::imp::jit_ffi_call_impl_float",
            jit_libffi::jit_ffi_call_impl_float as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::jit_libffi::imp::jit_ffi_call_impl_float",
            jit_libffi::jit_ffi_call_impl_float as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::jit_libffi::jit_ffi_call_impl_float",
            jit_libffi::jit_ffi_call_impl_float as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::jit_libffi::jit_ffi_call_impl_float",
            jit_libffi::jit_ffi_call_impl_float as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::jit_libffi::imp::jit_ffi_call_impl_singlefloat",
            jit_libffi::jit_ffi_call_impl_singlefloat as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::jit_libffi::imp::jit_ffi_call_impl_singlefloat",
            jit_libffi::jit_ffi_call_impl_singlefloat as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::jit_libffi::jit_ffi_call_impl_singlefloat",
            jit_libffi::jit_ffi_call_impl_singlefloat as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::jit_libffi::jit_ffi_call_impl_singlefloat",
            jit_libffi::jit_ffi_call_impl_singlefloat as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::jit_libffi::imp::jit_ffi_call_impl_void",
            jit_libffi::jit_ffi_call_impl_void as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::jit_libffi::imp::jit_ffi_call_impl_void",
            jit_libffi::jit_ffi_call_impl_void as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::jit_libffi::jit_ffi_call_impl_void",
            jit_libffi::jit_ffi_call_impl_void as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::jit_libffi::jit_ffi_call_impl_void",
            jit_libffi::jit_ffi_call_impl_void as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::jit_libffi::imp::jit_ffi_call_impl_any",
            jit_libffi::jit_ffi_call_impl_any as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::jit_libffi::imp::jit_ffi_call_impl_any",
            jit_libffi::jit_ffi_call_impl_any as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::jit_libffi::jit_ffi_call_impl_any",
            jit_libffi::jit_ffi_call_impl_any as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::jit_libffi::jit_ffi_call_impl_any",
            jit_libffi::jit_ffi_call_impl_any as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::cdataobj::raw_ptradd",
            cdataobj::raw_ptradd as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::cdataobj::raw_ptradd",
            cdataobj::raw_ptradd as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::cdataobj::raw_read_ptr",
            cdataobj::raw_read_ptr as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::cdataobj::raw_read_ptr",
            cdataobj::raw_read_ptr as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_read_i8",
            misc::raw_read_i8 as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::misc::raw_read_i8",
            misc::raw_read_i8 as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_read_u8",
            misc::raw_read_u8 as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::misc::raw_read_u8",
            misc::raw_read_u8 as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_write_i8",
            misc::raw_write_i8 as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::misc::raw_write_i8",
            misc::raw_write_i8 as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_write_u8",
            misc::raw_write_u8 as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::misc::raw_write_u8",
            misc::raw_write_u8 as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_read_i16",
            misc::raw_read_i16 as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::misc::raw_read_i16",
            misc::raw_read_i16 as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_read_u16",
            misc::raw_read_u16 as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::misc::raw_read_u16",
            misc::raw_read_u16 as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_write_i16",
            misc::raw_write_i16 as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::misc::raw_write_i16",
            misc::raw_write_i16 as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_write_u16",
            misc::raw_write_u16 as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::misc::raw_write_u16",
            misc::raw_write_u16 as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_read_i32",
            misc::raw_read_i32 as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::misc::raw_read_i32",
            misc::raw_read_i32 as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_read_u32",
            misc::raw_read_u32 as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::misc::raw_read_u32",
            misc::raw_read_u32 as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_write_i32",
            misc::raw_write_i32 as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::misc::raw_write_i32",
            misc::raw_write_i32 as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_write_u32",
            misc::raw_write_u32 as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::misc::raw_write_u32",
            misc::raw_write_u32 as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_read_i64",
            misc::raw_read_i64 as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::misc::raw_read_i64",
            misc::raw_read_i64 as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_read_u64",
            misc::raw_read_u64 as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::misc::raw_read_u64",
            misc::raw_read_u64 as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_write_i64",
            misc::raw_write_i64 as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::misc::raw_write_i64",
            misc::raw_write_i64 as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_write_u64",
            misc::raw_write_u64 as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::misc::raw_write_u64",
            misc::raw_write_u64 as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_read_f32",
            misc::raw_read_f32 as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::misc::raw_read_f32",
            misc::raw_read_f32 as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_write_f32",
            misc::raw_write_f32 as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::misc::raw_write_f32",
            misc::raw_write_f32 as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_read_f64",
            misc::raw_read_f64 as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::misc::raw_read_f64",
            misc::raw_read_f64 as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_write_f64",
            misc::raw_write_f64 as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::misc::raw_write_f64",
            misc::raw_write_f64 as *const (),
        );
    }
    #[cfg(all(
        feature = "full",
        any(unix, windows),
        feature = "host_env",
        not(feature = "sandbox")
    ))]
    {
        let f = module::_ctypes::cdata::cdata_bytes_object as *const ();
        single(
            entries,
            "pyre_interpreter::module::_ctypes::cdata::cdata_bytes_object",
            f,
        );
        single(entries, "pyre_interpreter::cdata_bytes_object", f);
        single(
            entries,
            "pyre_module::module::_ctypes::cdata::cdata_bytes_object",
            f,
        );
    }
}

fn walk_optional_global_roots(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    let _ = visitor;
    #[cfg(all(not(target_arch = "wasm32"), not(feature = "sandbox")))]
    module::faulthandler::handler::walk_faulthandler_roots(visitor);
    #[cfg(all(
        feature = "full",
        any(unix, windows),
        feature = "host_env",
        not(feature = "sandbox")
    ))]
    module::_ctypes::cdata::walk_pyobj_container_roots(visitor);
}

fn optional_subclass_range_aliases() -> Vec<pyre_object::pyobject::SubclassRangeAlias> {
    use pyre_object::lltype::PyreClassPyTypeOf;
    use pyre_object::pyobject::subclass_range_alias;

    fn typed<T: PyreClassPyTypeOf>() -> &'static pyre_object::PyType {
        unsafe { &*T::PYTYPE }
    }

    let mut aliases = vec![
        subclass_range_alias(129, typed::<module::_tokenize::W_TokenizerIter>()),
        // `unicodedata.UCD` sits at AUTO-ID 157, before `__pypy__.Bufferable`.
        subclass_range_alias(157, typed::<module::unicodedata::W_UCD>()),
        subclass_range_alias(162, typed::<module::_json::W_Scanner>()),
        subclass_range_alias(163, typed::<module::_json::W_Encoder>()),
        // `_hashlib`'s per-object digest/HMAC contexts follow their Python
        // owners and have sweep-time native-state destructors in build_gc.
        subclass_range_alias(164, typed::<module::_hashlib::W_HashState>()),
        subclass_range_alias(165, typed::<module::_hashlib::W_Hmac>()),
        subclass_range_alias(172, typed::<module::_bz2::W_BZ2Compressor>()),
        subclass_range_alias(173, typed::<module::_bz2::W_BZ2Decompressor>()),
        // `_lzma`'s two stream objects own their liblzma coder, unconditional
        // so their ids agree on wasm/native.
        subclass_range_alias(174, typed::<module::_lzma::W_LZMACompressor>()),
        subclass_range_alias(175, typed::<module::_lzma::W_LZMADecompressor>()),
        // PyPy zlib stream wrappers own their native stream and lock directly.
        // Keep these unconditional entries ahead of target-gated native types.
        subclass_range_alias(169, typed::<module::zlib::W_Compress>()),
        subclass_range_alias(170, typed::<module::zlib::W_Decompress>()),
        subclass_range_alias(171, typed::<module::zlib::W_ZlibDecompressor>()),
        // `_lsprof`'s profiler and stats result owners are unconditional.
        subclass_range_alias(176, typed::<module::_lsprof::W_Profiler>()),
        subclass_range_alias(177, typed::<module::_lsprof::W_StatsEntry>()),
        subclass_range_alias(178, typed::<module::_lsprof::W_StatsSubEntry>()),
        // `_queue.SimpleQueue` is unconditional and carries a native FIFO, so
        // it closes the ungated aliases ahead of the target-gated ones.
        subclass_range_alias(179, typed::<module::_queue::W_SimpleQueue>()),
        // `functools.KeyWrapper` keeps the id it had beside the thread types.
        subclass_range_alias(156, typed::<module::_functools::W_KeyWrapper>()),
    ];
    // The rustls-backed `_ssl` aliases preserve `build_gc`'s registration
    // order for `W_SSLContext`, `W_MemoryBIO`, and `W_SSLSession`.
    #[cfg(all(
        feature = "full",
        not(target_arch = "wasm32"),
        not(feature = "sandbox")
    ))]
    {
        aliases.push(subclass_range_alias(
            190,
            typed::<module::_ssl::W_SSLContext>(),
        ));
        aliases.push(subclass_range_alias(
            191,
            typed::<module::_ssl::W_MemoryBIO>(),
        ));
        aliases.push(subclass_range_alias(
            192,
            typed::<module::_ssl::W_SSLSession>(),
        ));
        aliases.push(subclass_range_alias(
            193,
            typed::<module::_ssl::W_SSLSocket>(),
        ));
        aliases.push(subclass_range_alias(
            194,
            typed::<module::_ssl::W_Certificate>(),
        ));
    }
    // `mmap.mmap` follows the optional SSL tail on ordinary Unix/Windows
    // builds. A sandbox or host_env-off build has no `mmap` module at all.
    #[cfg(all(
        feature = "full",
        not(target_arch = "wasm32"),
        feature = "host_env",
        not(feature = "sandbox")
    ))]
    aliases.push(subclass_range_alias(195, typed::<module::mmap::W_MMap>()));
    #[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
    aliases.push(subclass_range_alias(
        196,
        typed::<module::_overlapped::W_Overlapped>(),
    ));
    #[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
    aliases.push(subclass_range_alias(
        197,
        typed::<module::_winapi::overlapped::W_Overlapped>(),
    ));
    // `_cffi_backend` sits at the rclass tail (196 on Unix, 199 on Windows
    // where overlapped/console occupy 196-198).
    #[cfg(all(
        feature = "full",
        feature = "host_env",
        not(feature = "sandbox"),
        not(target_arch = "wasm32")
    ))]
    {
        #[cfg(windows)]
        const CFFI_FIRST_TYPE_ID: u32 = 199;
        #[cfg(not(windows))]
        const CFFI_FIRST_TYPE_ID: u32 = 196;
        aliases.push(subclass_range_alias(
            CFFI_FIRST_TYPE_ID,
            typed::<module::_cffi_backend::ctypeobj::W_CType>(),
        ));
        aliases.push(subclass_range_alias(
            CFFI_FIRST_TYPE_ID + 1,
            typed::<module::_cffi_backend::ctypearray::W_CDataIter>(),
        ));
        aliases.push(subclass_range_alias(
            CFFI_FIRST_TYPE_ID + 2,
            typed::<module::_cffi_backend::cdataobj::W_CData>(),
        ));
        aliases.push(subclass_range_alias(
            CFFI_FIRST_TYPE_ID + 3,
            typed::<module::_cffi_backend::ctypestruct::W_CField>(),
        ));
        aliases.push(subclass_range_alias(
            CFFI_FIRST_TYPE_ID + 4,
            typed::<module::_cffi_backend::libraryobj::W_Library>(),
        ));
        aliases.push(subclass_range_alias(
            CFFI_FIRST_TYPE_ID + 5,
            typed::<module::_cffi_backend::allocator::W_Allocator>(),
        ));
        aliases.push(subclass_range_alias(
            CFFI_FIRST_TYPE_ID + 6,
            typed::<module::_cffi_backend::cbuffer::MiniBuffer>(),
        ));
        aliases.push(subclass_range_alias(
            CFFI_FIRST_TYPE_ID + 7,
            typed::<module::_cffi_backend::func::OffsetInBytes>(),
        ));
        aliases.push(subclass_range_alias(
            CFFI_FIRST_TYPE_ID + 8,
            typed::<module::_cffi_backend::ffi_obj::W_FFIObject>(),
        ));
        aliases.push(subclass_range_alias(
            CFFI_FIRST_TYPE_ID + 9,
            typed::<module::_cffi_backend::realize_c_type::W_RawFuncType>(),
        ));
        aliases.push(subclass_range_alias(
            CFFI_FIRST_TYPE_ID + 10,
            typed::<module::_cffi_backend::lib_obj::W_LibObject>(),
        ));
        aliases.push(subclass_range_alias(
            CFFI_FIRST_TYPE_ID + 11,
            typed::<module::_cffi_backend::cglob::W_GlobSupport>(),
        ));
        aliases.push(subclass_range_alias(
            CFFI_FIRST_TYPE_ID + 12,
            typed::<module::_cffi_backend::wrapper::W_FunctionWrapper>(),
        ));
    }
    aliases
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    /// The `_csv::dialect_class::type_object` accessor is hand-written (not
    /// `#[pyre_methods]` / `py_class!`), yet the front recognizer stamps every
    /// `type_object` accessor `dont_look_inside`.  It must still publish a
    /// residual address, or a traced `_csv.Dialect` type lookup residualizes to
    /// a symbolic fnaddr and inline JIT descent aborts.
    #[test]
    fn jit_trace_fnaddrs_covers_hand_written_csv_dialect_type_object() {
        let bindings: HashMap<&'static str, i64> =
            pyre_interpreter::jit_trace_fnaddrs().into_iter().collect();
        assert!(
            bindings.contains_key("pyre_module::module::_csv::dialect_class::type_object"),
            "hand-written _csv::dialect_class::type_object must publish a residual fnaddr",
        );
        assert!(
            bindings.contains_key("module::_csv::dialect_class::type_object"),
            "the crate-stripped alias must resolve too",
        );
    }

    /// `#[pyre_methods]` wrappers in this crate must appear in the
    /// process-global fnaddr table. The prepass reads the same table
    /// from a host copy of this crate; a missing row here is the same
    /// defect as a build script that forgot to link `pyre-module`.
    #[test]
    fn jit_trace_fnaddrs_covers_moved_hashlib_wrapper() {
        let bindings: HashMap<&'static str, i64> =
            pyre_interpreter::jit_trace_fnaddrs().into_iter().collect();
        assert!(
            bindings.contains_key(
                "pyre_module::module::_hashlib::hash_state_class::__majit_wrap___new__"
            ),
            "moved _hashlib #[pyre_methods] wrappers must publish residual fnaddrs",
        );
    }

    #[cfg(all(
        feature = "full",
        not(target_arch = "wasm32"),
        feature = "host_env",
        not(feature = "sandbox")
    ))]
    #[test]
    fn jit_trace_fnaddrs_covers_moved_mmap_type() {
        let bindings: HashMap<&'static str, i64> =
            pyre_interpreter::jit_trace_fnaddrs().into_iter().collect();
        assert!(
            bindings.contains_key("pyre_module::module::mmap::interp_mmap::mmap_type"),
            "moved mmap_type must publish a residual fnaddr",
        );
    }

    #[cfg(all(feature = "full", unix, not(feature = "sandbox")))]
    #[test]
    fn jit_trace_fnaddrs_covers_moved_select_wrapper() {
        let bindings: HashMap<&'static str, i64> =
            pyre_interpreter::jit_trace_fnaddrs().into_iter().collect();
        assert!(
            bindings
                .contains_key("pyre_module::module::select::interp_select::__majit_wrap_register"),
            "moved select #[pyre_methods] wrappers must publish residual fnaddrs",
        );
    }

    #[test]
    fn jit_trace_fnaddrs_covers_moved_lzma_wrapper() {
        let bindings: HashMap<&'static str, i64> =
            pyre_interpreter::jit_trace_fnaddrs().into_iter().collect();
        assert!(
            bindings.contains_key(
                "pyre_module::module::_lzma::compressor_methods::__majit_wrap___new__"
            ),
            "moved _lzma #[pyre_methods] wrappers must publish residual fnaddrs",
        );
    }

    #[test]
    fn jit_trace_fnaddrs_covers_moved_unicodedata_wrapper() {
        let bindings: HashMap<&'static str, i64> =
            pyre_interpreter::jit_trace_fnaddrs().into_iter().collect();
        assert!(
            bindings.contains_key("pyre_module::module::unicodedata::__majit_wrap_category"),
            "moved unicodedata #[pyre_methods] wrappers must publish residual fnaddrs",
        );
    }

    #[test]
    fn jit_trace_fnaddrs_covers_moved_bz2_wrapper() {
        let bindings: HashMap<&'static str, i64> =
            pyre_interpreter::jit_trace_fnaddrs().into_iter().collect();
        assert!(
            bindings.contains_key(
                "pyre_module::module::_bz2::compressor_methods::__majit_wrap___new__"
            ),
            "moved _bz2 #[pyre_methods] wrappers must publish residual fnaddrs",
        );
    }

    #[cfg(all(
        feature = "full",
        not(target_arch = "wasm32"),
        not(feature = "sandbox")
    ))]
    #[test]
    fn jit_trace_fnaddrs_covers_moved_ssl_wrapper() {
        let bindings: HashMap<&'static str, i64> =
            pyre_interpreter::jit_trace_fnaddrs().into_iter().collect();
        assert!(
            bindings.contains_key(
                "pyre_module::module::_ssl::ssl_session_methods::__majit_wrap___new__"
            ),
            "moved _ssl #[pyre_methods] wrappers must publish residual fnaddrs",
        );
    }

    #[test]
    fn jit_trace_fnaddrs_covers_moved_lsprof_wrapper() {
        let bindings: HashMap<&'static str, i64> =
            pyre_interpreter::jit_trace_fnaddrs().into_iter().collect();
        assert!(
            bindings.contains_key(
                "pyre_module::module::_lsprof::profiler_methods::__majit_wrap___new__"
            ),
            "moved _lsprof #[pyre_methods] wrappers must publish residual fnaddrs",
        );
    }

    #[test]
    fn jit_trace_fnaddrs_covers_moved_ll_math_hypot() {
        let bindings: HashMap<&'static str, i64> =
            pyre_interpreter::jit_trace_fnaddrs().into_iter().collect();
        assert!(
            bindings.contains_key("ll_math::math_hypot"),
            "moved ll_math hypot residual must publish after optional-module register",
        );
        assert!(
            bindings.contains_key("math_hypot"),
            "the crate-root hypot alias must resolve too",
        );
        for leaf in ["math_asin", "math_acosh", "math_expm1", "math_log1p"] {
            assert!(
                bindings.contains_key(leaf),
                "ll_math {leaf} residual must publish after optional-module register",
            );
        }
    }
}
