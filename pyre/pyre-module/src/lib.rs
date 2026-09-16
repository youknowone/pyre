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
    pyre_interpreter::importing::register_builtin_module("pyexpat", module::pyexpat::init);
    #[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
    pyre_interpreter::importing::register_builtin_module("resource", module::resource::init);
    #[cfg(all(unix, not(feature = "sandbox")))]
    pyre_interpreter::importing::register_builtin_module("syslog", module::syslog::init);
    #[cfg(all(unix, not(feature = "sandbox")))]
    pyre_interpreter::importing::register_builtin_module("termios", module::termios::init);
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

/// `ll_math.py` C llexternals. The front retargets Opaque
/// `f64::{hypot,atan2,copysign,floor,ceil,ln,exp,sin,cos,powf,sqrt,log10}`
/// to these paths; the raising `ll_math_*` wrappers stay around them.
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

    pair(
        entries,
        "ll_math::math_hypot",
        "math_hypot",
        math::jit_math_hypot as *const (),
    );
    pair(
        entries,
        "ll_math::math_atan2",
        "math_atan2",
        math::jit_math_atan2 as *const (),
    );
    pair(
        entries,
        "ll_math::math_copysign",
        "math_copysign",
        math::jit_math_copysign as *const (),
    );
    pair(
        entries,
        "ll_math::math_floor",
        "math_floor",
        math::jit_math_floor_raw as *const (),
    );
    pair(
        entries,
        "ll_math::math_ceil",
        "math_ceil",
        math::jit_math_ceil_raw as *const (),
    );
    pair(
        entries,
        "ll_math::math_log",
        "math_log",
        math::jit_math_log_raw as *const (),
    );
    pair(
        entries,
        "ll_math::math_exp",
        "math_exp",
        math::jit_math_exp_raw as *const (),
    );
    pair(
        entries,
        "ll_math::math_sin",
        "math_sin",
        math::jit_math_sin_raw as *const (),
    );
    pair(
        entries,
        "ll_math::math_cos",
        "math_cos",
        math::jit_math_cos_raw as *const (),
    );
    pair(
        entries,
        "ll_math::math_pow",
        "math_pow",
        math::jit_math_pow_raw as *const (),
    );
    pair(
        entries,
        "ll_math::math_sqrt",
        "math_sqrt",
        math::jit_math_sqrt_raw as *const (),
    );
    pair(
        entries,
        "ll_math::math_log10",
        "math_log10",
        math::jit_math_log10_raw as *const (),
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
        subclass_range_alias(162, typed::<module::_json::W_Scanner>()),
        subclass_range_alias(163, typed::<module::_json::W_Encoder>()),
        subclass_range_alias(172, typed::<module::_bz2::W_BZ2Compressor>()),
        subclass_range_alias(173, typed::<module::_bz2::W_BZ2Decompressor>()),
    ];
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
    }
}
