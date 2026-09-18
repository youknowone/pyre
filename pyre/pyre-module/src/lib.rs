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

/// Keep the `module::` path prefix so harvested hint paths and
/// `should_lower_module` stay `module::<name>` after the crate split.
pub mod module;

/// Register default/working modules into the interpreter builtin table.
///
/// The interpreter does not depend on this crate. The final binary calls
/// [`register`] before `install_builtin_modules`.
pub fn install_optional_modules() {
    pyre_interpreter::importing::register_builtin_module("_bisect", module::_bisect::init);
    pyre_interpreter::importing::register_builtin_module("_blake2", module::_blake2::init);
    pyre_interpreter::importing::register_builtin_module("_bz2", module::_bz2::init);
    pyre_interpreter::importing::register_builtin_module("_csv", module::_csv::init);
    pyre_interpreter::importing::register_builtin_module("_codecs_cn", module::_codecs_cn::init);
    pyre_interpreter::importing::register_builtin_module("_codecs_hk", module::_codecs_hk::init);
    pyre_interpreter::importing::register_builtin_module(
        "_codecs_iso2022",
        module::_codecs_iso2022::init,
    );
    pyre_interpreter::importing::register_builtin_module("_codecs_jp", module::_codecs_jp::init);
    pyre_interpreter::importing::register_builtin_module("_codecs_kr", module::_codecs_kr::init);
    pyre_interpreter::importing::register_builtin_module("_codecs_tw", module::_codecs_tw::init);
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
    pyre_interpreter::importing::register_builtin_module("_queue", module::_queue::init);
    #[cfg(not(feature = "sandbox"))]
    pyre_interpreter::importing::register_builtin_module("_socket", module::_socket::init);
    #[cfg(all(not(target_arch = "wasm32"), not(feature = "sandbox")))]
    pyre_interpreter::importing::register_builtin_module("_ssl", module::_ssl::init);
    pyre_interpreter::importing::register_builtin_module("_statistics", module::_statistics::init);
    pyre_interpreter::importing::register_builtin_module(
        "_suggestions",
        module::_suggestions::init,
    );
    pyre_interpreter::importing::register_builtin_module("_template", module::_template::init);
    pyre_interpreter::importing::register_builtin_module("_tokenize", module::_tokenize::init);
    #[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
    pyre_interpreter::importing::register_builtin_module("_uuid", module::_uuid::init);
    #[cfg(target_os = "macos")]
    pyre_interpreter::importing::register_builtin_module("_scproxy", module::_scproxy::init);
    pyre_interpreter::importing::register_builtin_module("binascii", module::binascii::init);
    pyre_interpreter::importing::register_builtin_module("cmath", module::cmath::init);
    #[cfg(all(not(target_arch = "wasm32"), not(feature = "sandbox")))]
    pyre_interpreter::importing::register_builtin_module(
        "faulthandler",
        module::faulthandler::init,
    );
    pyre_interpreter::importing::register_builtin_module("math", module::math::init);
    #[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
    pyre_interpreter::importing::register_builtin_module("fcntl", module::fcntl::init);
    #[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
    pyre_interpreter::importing::register_builtin_module("grp", module::grp::init);
    #[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
    pyre_interpreter::importing::register_builtin_module("pwd", module::pwd::init);
    pyre_interpreter::importing::register_builtin_module("pyexpat", module::pyexpat::init);
    #[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
    pyre_interpreter::importing::register_builtin_module("resource", module::resource::init);
    #[cfg(all(unix, not(feature = "sandbox")))]
    pyre_interpreter::importing::register_builtin_module("syslog", module::syslog::init);
    #[cfg(all(unix, not(feature = "sandbox")))]
    pyre_interpreter::importing::register_builtin_module("termios", module::termios::init);
    pyre_interpreter::importing::register_builtin_module("unicodedata", module::unicodedata::init);
    pyre_interpreter::importing::register_builtin_module("zlib", module::zlib::init);
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
        },
    );
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
}

fn walk_optional_global_roots(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    let _ = visitor;
    #[cfg(all(not(target_arch = "wasm32"), not(feature = "sandbox")))]
    module::faulthandler::handler::walk_faulthandler_roots(visitor);
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
    ];
    // The rustls-backed `_ssl` aliases preserve `build_gc`'s registration
    // order for `W_SSLContext`, `W_MemoryBIO`, and `W_SSLSession`.
    #[cfg(all(not(target_arch = "wasm32"), not(feature = "sandbox")))]
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
    #[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
    aliases.push(subclass_range_alias(
        196,
        typed::<module::_overlapped::W_Overlapped>(),
    ));
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
