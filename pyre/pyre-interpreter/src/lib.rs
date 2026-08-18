#![allow(
    ambiguous_glob_reexports,
    // PyObjectRef and frame pointers are translated runtime handles. Public
    // object-space wrappers keep their safe Python-facing contract while raw
    // access remains encapsulated inside the interpreter. Function signatures,
    // enum shapes, and indexed opcode loops mirror PyPy's interpreter sources.
    clippy::approx_constant,
    clippy::doc_lazy_continuation,
    clippy::duplicate_underscore_argument,
    clippy::empty_line_after_doc_comments,
    clippy::enum_variant_names,
    clippy::explicit_counter_loop,
    clippy::iter_skip_next,
    clippy::macro_metavars_in_unsafe,
    clippy::manual_memcpy,
    clippy::manual_unwrap_or_default,
    clippy::missing_safety_doc,
    clippy::module_inception,
    clippy::multiple_bound_locations,
    clippy::mut_from_ref,
    clippy::needless_borrow,
    clippy::needless_range_loop,
    clippy::needless_update,
    clippy::neg_cmp_op_on_partial_ord,
    clippy::new_ret_no_self,
    clippy::nonminimal_bool,
    clippy::not_unsafe_ptr_arg_deref,
    clippy::result_unit_err,
    clippy::same_item_push,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unnecessary_cast,
    clippy::vec_box,
    clippy::while_let_loop,
    dead_code,
    non_snake_case,
    unconditional_recursion,
    unsafe_op_in_unsafe_fn,
    unused_assignments,
    unused_doc_comments,
    unused_imports,
    unused_mut,
    unused_unsafe,
    unused_variables
)]

//! pyre-interpreter — the Python interpreter.
//!
//! PyPy equivalent: pypy/interpreter/
//!
//! Contains the object space (baseobjspace.rs), bytecode evaluation (eval.rs),
//! frame management (pyframe.rs), function call dispatch (call.rs),
//! import machinery (importing.rs), builtin functions (builtins.rs),
//! type definitions (typedef.rs), and builtin modules (module/).

// ── Bytecode / compiler re-exports (was pyre-bytecode) ──
pub mod compile;
pub use compile::*;

// ── Core interpreter modules ──
pub mod _pypy_generic_alias;
pub mod _structseq;
pub mod argument;
pub mod astcompiler;
pub mod baseobjspace;
pub mod builtins;
#[cfg(all(
    feature = "cpyext",
    not(feature = "sandbox"),
    any(target_os = "macos", target_os = "linux")
))]
pub mod cpyext;
pub mod display;
pub mod error;
pub mod executioncontext;
pub mod frame_array;
pub mod function;
pub mod gateway;
// The OS-call seam (real syscalls vs. sandbox marshalling trampolines). Unix
// only: the real bodies use libc and the unix `OsStr`/`OsString` byte views.
#[cfg(unix)]
pub mod host_seam;

// On non-unix targets (wasm and Windows) the Unix syscall seam is configured
// out, but the process-environment and diagnostic-stdio operations need
// neither libc nor the sandbox trampoline (sandbox is Unix-only). Provide the
// subset shared code reaches so those call sites keep the same seam shape on
// every target.
#[cfg(not(unix))]
pub mod host_seam {
    /// Read a process environment value, in the same filesystem-bytes spelling
    /// the unix seam hands back. A Windows value holding an unpaired surrogate
    /// keeps it as its WTF-8 encoding, so the caller that turns the bytes back
    /// into a platform string recovers the code units the host stored.
    pub fn getenv(name: &[u8]) -> Result<Option<Vec<u8>>, ()> {
        let name = std::str::from_utf8(name).map_err(|_| ())?;
        Ok(std::env::var_os(name).map(|value| crate::gateway::fsencode_os_str(&value)))
    }

    /// Emit bytes to the interpreter's stdout (fd 1).
    ///
    /// Carries `dont_look_inside` for the same reason as the unix body: the
    /// write is the host boundary, so the front end residualizes the call.
    /// Both spellings need it — a corpus extracted on a non-unix host sees
    /// only this one.
    #[majit_macros::dont_look_inside]
    pub fn emit_stdout(bytes: &[u8]) {
        let bytes = crate::stdio_line_endings_bytes(bytes, "stdout");
        let bytes = bytes.as_ref();
        if super::print_hook_emit_bytes(bytes) {
            return;
        }
        use std::io::Write;
        let _ = std::io::stdout().write_all(bytes);
        flush_stdout_when_unbuffered();
    }

    /// Settle fd 1 when `-u` / PYTHONUNBUFFERED asked for no buffering.
    ///
    /// `std::io::Stdout` is a `LineWriter`, so without this a value carrying no
    /// newline waits in its buffer until process exit — the opposite of what
    /// the flag requests.
    pub fn flush_stdout_when_unbuffered() {
        if crate::importing::unbuffered_flag() {
            use std::io::Write;
            let _ = std::io::stdout().flush();
        }
    }

    /// Emit bytes to the interpreter's stderr (fd 2).
    #[majit_macros::dont_look_inside]
    pub fn emit_stderr(bytes: &[u8]) {
        let bytes = crate::stdio_line_endings_bytes(bytes, "stderr");
        let bytes = bytes.as_ref();
        if super::stderr_hook_emit(bytes) {
            return;
        }
        use std::io::Write;
        let _ = std::io::stderr().write_all(bytes);
    }

    /// Flush the interpreter's stdout (fd 1).
    pub fn flush_stdout() {
        use std::io::Write;
        let _ = std::io::stdout().flush();
    }
}
pub mod async_operation;
pub mod jit_fnaddr;
pub mod launch_env;
pub mod listobject;
pub mod listsort;
pub mod opcode_ops;
pub mod pycode;
pub mod pyopcode;
pub mod pytraceback;
pub mod reduce_protocol;
pub mod runtime_ops;
pub mod shared_opcode;
pub mod sliceobject;
pub mod stack_check;
pub mod syntax_warnings;
pub mod type_methods;
pub mod typedef;
pub mod warn;

// ── Execution and import modules ──
pub mod call;
pub mod eval;
pub mod importing;
pub mod module;
pub mod objspace;
pub mod pyframe;

/// Test-only hash-hook installer shared across the crate's `#[cfg(test)]`
/// modules.  Production installs `space.hash_w` at boot
/// (`pyre-jit::eval::init_jit_hooks`, before the first user statement);
/// unit tests that build object- or str-keyed dicts must install the same
/// single hash path on their own thread, because
/// `pyre_object::dict_eq_hook` stores the hook thread-locally and libtest
/// runs each `#[test]` on a fresh thread.  The `test-hooks` feature exposes
/// the module to downstream test builds (`pyre-jit`'s dev-dependency enables
/// it) whose `#[test]`s drive interpreter frames through `init_typeobjects`
/// without compiling this crate under `cfg(test)`.
#[cfg(any(test, feature = "test-hooks"))]
pub mod test_hooks {
    use pyre_object::PyObjectRef;

    /// `baseobjspace.py:840-845 hash_w` — the single hash entry point,
    /// mirrored for tests via `builtins::try_hash_value`.  On error it
    /// records the pending exception the same way the production
    /// trampoline does (`pyre-jit::eval`'s `pyre_object_hash_w_trampoline`).
    unsafe fn test_hash_w(obj: PyObjectRef) -> i64 {
        match crate::builtins::try_hash_value(obj) {
            Ok(h) => h,
            Err(e) => {
                crate::baseobjspace::set_pending_dict_hash_error(e);
                pyre_object::dict_eq_hook::signal_hash_error(obj);
                0
            }
        }
    }

    unsafe fn test_hash_str(ptr: *const u8, len: usize) -> i64 {
        crate::builtins::hash_str_bytes(unsafe { std::slice::from_raw_parts(ptr, len) })
    }

    /// Install the real `hash_w` and `hash_str` on the current test thread.
    /// Call at the top of any `#[test]` that constructs an object/str-keyed
    /// dict.  Public (under the same gate) so downstream test builds that
    /// enable `test-hooks` can install the hook at their own test chokepoints
    /// for `#[test]`s that never reach `init_typeobjects`.
    pub fn install_hash_hook() {
        pyre_object::dict_eq_hook::register_hash_w_hook(test_hash_w);
        pyre_object::dict_eq_hook::register_hash_str_hook(test_hash_str);
    }
}

// ── Declarative builtin-module registration ──
//
// `pypy/module/<name>/moduledef.py` declares its surface as a dict
// literal — `interpleveldefs = { 'name': 'interp_x.func', ... }` — and
// PyPy's MixedModule machinery walks the dict at import time.  Pyre
// mirrors that with the `py_module!` macro below: each call expands to
// a `pub fn init(ns: PyObjectRef)` that stores every entry via
// `module_ns_store`.  The previous one-line `moduledef.rs` shim
// (which did nothing but `super::interp_x::register_module(ns)`) has
// been retired across every builtin module.
//
// The macro is intentionally minimal at the value layer — each entry's
// RHS is just a `PyObjectRef` expression — so call-site code stays the
// same as the hand-written `module_ns_store` calls it replaces.  An
// `extra_init: |ns| { ... }` escape hatch covers PyPy's
// `buildloaders` / `startup` post-processing (constants loops, cfg
// gating, helper-typed registration).  For modules whose
// `register_module` body is too large to inline (`_socket`, `_sre`,
// `sys`), `mod.rs` falls back to `pub use interp_x::register_module
// as init;` — semantically identical to the macro form, deferred to a
// later pass.

/// PyPy MixedModule-style declarative module registration.
///
/// Mirrors `pypy/module/<name>/moduledef.py`:
///
/// ```text
/// class Module(MixedModule):
///     interpleveldefs = {
///         'pickle': 'interp_copyreg.pickle',
///         'dispatch_table': 'space.newdict()',
///         'sin':            'interp_math.sin',     # arity inferred upstream
///     }
/// ```
///
/// Becomes:
///
/// ```ignore
/// crate::py_module! {
///     "math",
///     interpleveldefs: {
///         "pi"  => pyre_object::floatobject::w_float_new(pymath::math::PI),
///     },
///     functions: {
///         "sin"   / 1 = interp_math::sin,           // fixed-arity
///         "atan2" / 2 = interp_math::atan2,
///         "log"   / *  = interp_math::log,          // varargs
///     },
///     module_functions: {
///         "getweakrefcount" / 1 = interp__weakref::getweakrefcount,
///     },
/// }
/// ```
///
/// `interpleveldefs` carries arbitrary `PyObjectRef` expressions;
/// `functions` / `module_functions` are PyPy's `interp_X.fn` string-ref
/// shorthand expanded inline — the name appears once on the LHS, the
/// function path once on the RHS, and the macro injects the
/// `make_builtin_function*` call.  `extra_init: |ns| { ... }` is the
/// escape hatch for `buildloaders` / `startup` post-processing
/// (constants loops, cfg gating).
///
/// The `name:` slot is currently informational — `importing.rs` still
/// owns the module-name -> init-fn map.  A follow-up may use it to drive
/// an inventory-style auto-registration.
#[macro_export]
macro_rules! py_module {
    (
        $name:literal
        $(, interpleveldefs: { $($key:literal => $value:expr),* $(,)? })?
        $(, int_constants: { $($int_key:literal => $int_value:expr),* $(,)? })?
        $(, exceptions: { $($exc_key:literal => $exc_base:expr),* $(,)? })?
        $(, appleveldefs: { $($appfile:literal => [ $($appname:literal),* $(,)? ]),* $(,)? })?
        $(, inline_app: { $($inline_src:literal => [ $($inline_name:literal),* $(,)? ]),* $(,)? })?
        $(, inline_functions: {
            $(
                fn $ifn_name:ident ( $($ifn_args:tt)* ) $(-> $ifn_ret:ty)? $ifn_body:block
            )*
        })?
        $(, functions: { $($fn_key:literal / $fn_arity:tt = $fn_path:expr),* $(,)? })?
        $(, module_functions: { $($mfn_key:literal / $mfn_arity:tt = $mfn_path:expr),* $(,)? })?
        $(, extra_init: |$ns:ident| $body:block)?
        $(,)?
    ) => {
        pub fn init(ns: ::pyre_object::PyObjectRef) {
            let _name = $name;
            $($(
                $crate::module_ns_store(ns, $key, $value);
            )*)?
            // int_constants: integer module constants — PyPy MixedModule
            // `interpleveldefs = {'NAME': 'space.wrap(value)'}` for the
            // common int case (errno/fcntl/select flags).  Each `$int_value`
            // is an `i64`-valued expression wrapped via `w_int_new`, saving
            // the per-entry `module_ns_store(ns, k, w_int_new(v))`.
            $($(
                $crate::module_ns_store(
                    ns, $int_key,
                    ::pyre_object::w_int_new($int_value as i64),
                );
            )*)?
            // exceptions: module-local exception classes — PyPy
            // `new_exception_class("<mod>.Name", base)` (error.py:857).
            // The class name is auto-qualified as `"<$name>.<key>"` and
            // built via `make_exc_type` (which also records it in the
            // exc-class registry); the short `key` is the attribute name
            // stored in the module dict.  The RHS is the base class
            // expression, e.g. `lookup_exc_class("OSError").unwrap()`.
            $($(
                $crate::module_ns_store(
                    ns, $exc_key,
                    $crate::builtins::make_exc_type(
                        ::std::concat!($name, ".", $exc_key),
                        $crate::builtins::exc_exception_new,
                        $exc_base,
                    ),
                );
            )*)?
            // appleveldefs: bundle Python source via `include_str!` at
            // compile time, then resolve each name through
            // `appleveldef_install`.  Mirrors PyPy MixedModule's
            // `appleveldefs = {"name": "app_X:name"}` lookup, but the
            // .py file is statically linked into the binary rather than
            // read off the filesystem at module-init time.
            $($(
                $crate::importing::appleveldef_install(
                    ns,
                    include_str!($appfile),
                    $appfile,
                    &[ $( $appname ),* ],
                );
            )*)?
            // inline_app: PyPy `applevel(r'''…''')` (gateway.py:823) —
            // embed a Python snippet inline; the runtime executes it the
            // same way as appleveldefs but the source comes from a
            // string literal instead of `include_str!` on a sibling .py
            // file.  Names listed in the `=> [...]` brackets get copied
            // out of the app namespace into the module dict.
            $($(
                $crate::importing::appleveldef_install(
                    ns,
                    $inline_src,
                    "<inline>",
                    &[ $( $inline_name ),* ],
                );
            )*)?
            // inline_functions: `#[pyre_function]` typed defs whose name +
            // arity are derived from the signature.  Replaces the
            // separate `#[pyre_function] fn X` + `"X" / N = X` pair.
            $($(
                {
                    #[$crate::pyre_function]
                    fn $ifn_name ( $($ifn_args)* ) $(-> $ifn_ret)? $ifn_body
                    $crate::module_ns_store(
                        ns,
                        stringify!($ifn_name),
                        $crate::gateway::with_module(
                            $name,
                            $crate::make_module_builtin_function_with_arity_and_maybe_sig(
                                stringify!($ifn_name),
                                $ifn_name,
                                ::paste::paste! { [<$ifn_name _pyre_arity>]() },
                                ::paste::paste! { [<$ifn_name _pyre_sig>]() },
                            ),
                        ),
                    );
                }
            )*)?
            $($(
                $crate::module_ns_store(
                    ns, $fn_key,
                    $crate::gateway::with_module(
                        $name,
                        $crate::py_module_fn!($fn_key, $fn_arity, $fn_path),
                    ),
                );
            )*)?
            $($(
                $crate::module_ns_store(
                    ns, $mfn_key,
                    $crate::gateway::with_module(
                        $name,
                        $crate::py_module_module_fn!($mfn_key, $mfn_arity, $mfn_path),
                    ),
                );
            )*)?
            $(
                {
                    let $ns: ::pyre_object::PyObjectRef = ns;
                    $body
                }
            )?
        }
    };
}

/// PyPy `class W_X(W_Root) + TypeDef(...)` equivalent — emits a thread-
/// local `type_object()` accessor that lazily builds a `W_TypeObject`
/// inheriting from `object`, populated with typed methods.  Each method
/// receives `self_obj: PyObjectRef` as its first parameter (PyPy's
/// `self` post-`@interp2app`) and any remaining typed parameters are
/// auto-unwrapped via `#[pyre_function]`.  Method arity (including
/// `self_obj`) is derived from the signature.  Instances carry
/// `__dict__` by default (matching PyPy `hasdict=True` for most
/// W_Root subclasses); state is stored as Python attributes on `self`
/// via `getattr`/`setattr` until a typed-payload backend is added.
///
/// ```ignore
/// crate::py_class! {
///     "_random.Random",
///     methods: {
///         fn __init__(self_obj: PyObjectRef, seed: i64) -> Result<(), crate::PyError> {
///             crate::baseobjspace::setattr_str(self_obj, "_state", ::pyre_object::w_int_new(seed))?;
///             Ok(())
///         }
///         fn random(self_obj: PyObjectRef) -> f64 {
///             // ... read self._state, mutate, write back
///         }
///     }
/// }
/// ```
///
/// expands to (roughly):
///
/// ```ignore
/// pub fn type_object() -> ::pyre_object::PyObjectRef {
///     static CELL: ::std::sync::OnceLock<usize> = ::std::sync::OnceLock::new();
///     *CELL.get_or_init(|| {
///         let tp = crate::typedef::make_builtin_type("_random.Random", |ns| {
///             #[crate::pyre_function]
///             fn __init__(self_obj: PyObjectRef, seed: i64) -> Result<(), crate::PyError> { ... }
///             crate::dict_storage_store(ns, "__init__",
///                 crate::make_builtin_function_with_arity("__init__", __init__, 2));
///             // ... more methods
///         });
///         unsafe { ::pyre_object::typeobject::w_type_set_hasdict(tp, true) };
///         tp as usize
///     }) as ::pyre_object::PyObjectRef
/// }
/// ```
#[macro_export]
macro_rules! py_class {
    (
        $name:literal
        $(, methods: {
            $(
                fn $mname:ident ( $($margs:tt)* ) $(-> $mret:ty)? $mbody:block
            )*
        })?
        $(, properties: {
            $(
                fn $pname:ident ( $($pargs:tt)* ) $(-> $pret:ty)? $pbody:block
            )*
        })?
        $(,)?
    ) => {
        pub fn type_object() -> ::pyre_object::PyObjectRef {
            static CELL: ::std::sync::OnceLock<usize> = ::std::sync::OnceLock::new();
            *CELL.get_or_init(|| {
                let tp = $crate::typedef::make_builtin_type($name, |ns| {
                    // `make_builtin_function` (varargs, no arity check) is
                    // used here rather than `_with_arity` because methods
                    // with `Option<T>` parameters need to accept calls with
                    // fewer args (PyPy `def f(self, s=None)`).  The
                    // `#[pyre_function]` wrapper uses bounds-checked
                    // `args.len()` for Option arms so missing-arg → None,
                    // while required args still index `args[N]` directly.
                    $($(
                        {
                            #[$crate::pyre_function]
                            fn $mname ( $($margs)* ) $(-> $mret)? $mbody
                            unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                                ns,
                                stringify!($mname),
                                $crate::make_builtin_function(stringify!($mname), $mname),
                            ) };
                        }
                    )*)?
                    // `properties:` — each fn registered as a
                    // `GetSetProperty` descriptor so `obj.name`
                    // returns the value directly (PyPy
                    // `GetSetProperty(W_X.fget_name)`).
                    $($(
                        {
                            #[$crate::pyre_function]
                            fn $pname ( $($pargs)* ) $(-> $pret)? $pbody
                            unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                                ns,
                                stringify!($pname),
                                $crate::typedef::make_getset_descriptor_named(
                                    $crate::make_builtin_function(stringify!($pname), $pname),
                                    stringify!($pname),
                                ),
                            ) };
                        }
                    )*)?
                });
                unsafe { ::pyre_object::typeobject::w_type_set_hasdict(tp, true) };
                tp as usize
            }) as ::pyre_object::PyObjectRef
        }

        // Publish this accessor's residual-call address so the JIT's
        // `dont_look_inside` residual for `type_object` resolves through
        // `jit_trace_fnaddrs` (which iterates this slice), mirroring the
        // `#[pyre_methods]`-generated accessor.  Native only, matching the slice.
        #[cfg(not(target_arch = "wasm32"))]
        #[::linkme::distributed_slice(::pyre_object::lltype::PYRE_TYPE_OBJECT_FNADDRS)]
        #[allow(non_upper_case_globals)]
        static __PYRE_TYPE_OBJECT_FNADDR: ::pyre_object::lltype::TypeObjectFnDescriptor =
            ::pyre_object::lltype::TypeObjectFnDescriptor {
                path: ::core::concat!(::core::module_path!(), "::type_object"),
                func: type_object,
            };
    };
}

/// Typed-payload variant of [`py_class!`] — binds the Python-level
/// `W_TypeObject` to a Rust `#[pyre_class]` struct so instances
/// allocate the typed payload (`<W_X>::allocate(payload)`) and carry
/// the class's own static `PyType` in `ob_header.ob_type` instead of
/// piggy-backing on `INSTANCE_TYPE`.
///
/// The first argument names a `#[pyre_class]`-attributed struct that
/// owns the layout; the second is the Python-visible type name; the
/// `methods:` / `properties:` arms mirror [`py_class!`].
///
/// ```ignore
/// #[crate::pyre_class("_random.Random", type_id = 53)]
/// pub struct W_Random {
///     pub state: u64,
/// }
///
/// crate::py_class_typed! {
///     W_Random as "_random.Random",
///     methods: {
///         fn random(self_obj: PyObjectRef) -> f64 {
///             let w = W_Random::from_obj(self_obj).unwrap();
///             w.state = w.state.wrapping_mul(6364136223846793005).wrapping_add(1);
///             (w.state as f64) / (u64::MAX as f64)
///         }
///     }
/// }
/// ```
#[macro_export]
macro_rules! py_class_typed {
    (
        $struct:ident as $name:literal
        $(, methods: {
            $(
                fn $mname:ident ( $($margs:tt)* ) $(-> $mret:ty)? $mbody:block
            )*
        })?
        $(, properties: {
            $(
                fn $pname:ident ( $($pargs:tt)* ) $(-> $pret:ty)? $pbody:block
            )*
        })?
        $(,)?
    ) => {
        pub fn type_object() -> ::pyre_object::PyObjectRef {
            static CELL: ::std::sync::OnceLock<usize> = ::std::sync::OnceLock::new();
            *CELL.get_or_init(|| {
                let tp = $crate::typedef::make_builtin_type_with_layout(
                    $name,
                    |ns| {
                        $($(
                            {
                                #[$crate::pyre_function]
                                fn $mname ( $($margs)* ) $(-> $mret)? $mbody
                                unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                                    ns,
                                    stringify!($mname),
                                    $crate::make_builtin_function(stringify!($mname), $mname),
                                ) };
                            }
                        )*)?
                        $($(
                            {
                                #[$crate::pyre_function]
                                fn $pname ( $($pargs)* ) $(-> $pret)? $pbody
                                unsafe { pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                                    ns,
                                    stringify!($pname),
                                    $crate::typedef::make_getset_descriptor_named(
                                        $crate::make_builtin_function(stringify!($pname), $pname),
                                        stringify!($pname),
                                    ),
                                ) };
                            }
                        )*)?
                    },
                    $crate::typedef::w_object(),
                    <$struct as $crate::PyreClassPyTypeOf>::PYTYPE,
                );
                // Eagerly bind the W_TypeObject to the static
                // `PyType` so `<$struct>::allocate(...)` can stamp
                // `ob_header.w_class` at construction without racing
                // the post-init typedef pass (matches
                // `getset_descriptor_type()`'s eager `set_instantiate`).
                ::pyre_object::pyobject::set_instantiate(
                    unsafe { &*<$struct as $crate::PyreClassPyTypeOf>::PYTYPE },
                    tp,
                );
                tp as usize
            }) as ::pyre_object::PyObjectRef
        }

        // Publish this accessor's residual-call address so the JIT's
        // `dont_look_inside` residual for `type_object` resolves through
        // `jit_trace_fnaddrs` (which iterates this slice), mirroring the
        // `#[pyre_methods]`-generated accessor.  Native only, matching the slice.
        #[cfg(not(target_arch = "wasm32"))]
        #[::linkme::distributed_slice(::pyre_object::lltype::PYRE_TYPE_OBJECT_FNADDRS)]
        #[allow(non_upper_case_globals)]
        static __PYRE_TYPE_OBJECT_FNADDR: ::pyre_object::lltype::TypeObjectFnDescriptor =
            ::pyre_object::lltype::TypeObjectFnDescriptor {
                path: ::core::concat!(::core::module_path!(), "::type_object"),
                func: type_object,
            };
    };
}

/// Helper for `py_module!`'s `functions:` arm.  `*` → varargs
/// (`make_module_builtin_function`); numeric arity →
/// `make_module_builtin_function_with_arity`.  Functions directly in a
/// mixed-module are non-descriptors (`mixedmodule.py:_load_lazily` converts
/// every mixed-module function to `BuiltinFunction`, whose typedef omits
/// `__get__`), so storing one on a user class must not synthesize a bound
/// method — identical to the `module_functions:` arm.
///
/// A declared arity is also enforced: the body behind it indexes exactly that
/// many slots, so `check_declared_arity` runs first and a call that does not
/// fit raises `TypeError` instead of reading past the end of `args`.  A body
/// that accepts a range of counts declares `*` and checks itself.
#[macro_export]
macro_rules! py_module_fn {
    ($key:literal, *, $path:expr) => {
        $crate::make_module_builtin_function($key, $path)
    };
    ($key:literal, $arity:literal, $path:expr) => {
        $crate::make_module_builtin_function_with_arity(
            $key,
            $crate::py_checked_arity_fn!($key, $arity, $path),
            $arity,
        )
    };
}

/// Wrap a declared-arity builtin body in its positional-count check.  The
/// wrapper captures nothing, so it still coerces to a `BuiltinCodeFn` pointer.
#[macro_export]
macro_rules! py_checked_arity_fn {
    ($key:literal, $arity:literal, $path:expr) => {
        |args: &[::pyre_object::PyObjectRef]| -> ::std::result::Result<
                            ::pyre_object::PyObjectRef,
                            $crate::PyError,
                        > {
                            $crate::gateway::check_declared_positional_arity($key, $arity, args)?;
                            // The annotation is what gives a bare `|args| ...` body its
                            // parameter type; the coercion to a pointer is a no-op.
                            let __pyre_body: $crate::BuiltinCodeFn = $path;
                            __pyre_body(args)
                        }
    };
}

/// Helper for `py_module!`'s `module_functions:` arm — same shape as
/// `py_module_fn!` but emits the module-builtin variant (no `self`
/// binding when stored on a class).
#[macro_export]
macro_rules! py_module_module_fn {
    ($key:literal, *, $path:expr) => {
        $crate::make_module_builtin_function($key, $path)
    };
    ($key:literal, $arity:literal, $path:expr) => {
        $crate::make_module_builtin_function_with_arity(
            $key,
            $crate::py_checked_arity_fn!($key, $arity, $path),
            $arity,
        )
    };
}

/// Declare the standard `pub mod interp_X; pub use init` pair for a
/// module whose body is too large to inline into `py_module!`.  Matches
/// PyPy's split between `moduledef.py` (declarative table) and
/// `interp_<name>.py` (implementations).
///
/// ```ignore
/// pyre_module_init!(interp_socket);
/// ```
///
/// expands to
///
/// ```ignore
/// pub mod interp_socket;
/// pub use interp_socket::register_module as init;
/// ```
#[macro_export]
macro_rules! pyre_module_init {
    ($interp_mod:ident) => {
        pub mod $interp_mod;
        pub use $interp_mod::register_module as init;
    };
}

/// `space.newlist([…])` builder.  Each item is wrapped via the
/// `pywrap!` per-item rule (literal kind ↦ `w_int_new` / `w_float_new`
/// / `w_str_new` / `w_bool_from` / passthrough for already-wrapped
/// `PyObjectRef`).  Mirrors PyPy `space.newlist([space.newint(1),
/// space.newtext("abc")])` where the boilerplate per-item wrap is
/// implicit in the helper rather than spelled out at every call site.
///
/// ```ignore
/// pylist![1i64, "abc", 3.14, py_obj]        // → list [1, "abc", 3.14, py_obj]
/// pytuple![1i64, "abc"]                     // → tuple (1, "abc")
/// pydict! { "k1" => 1i64, "k2" => 3.14 }    // → {"k1": 1, "k2": 3.14}
/// pyset! { 1i64, 2i64, 3i64 }               // → {1, 2, 3}
/// ```
///
/// Mixing already-wrapped `PyObjectRef` with literals works because the
/// passthrough `impl PywrapKind for PyObjectRef` returns the value
/// verbatim.
#[macro_export]
macro_rules! pylist {
    ( $($e:expr),* $(,)? ) => {
        ::pyre_object::w_list_new(vec![ $( $crate::PywrapKind::into_py($e) ),* ])
    };
}

#[macro_export]
macro_rules! pytuple {
    ( $($e:expr),* $(,)? ) => {
        ::pyre_object::w_tuple_new(vec![ $( $crate::PywrapKind::into_py($e) ),* ])
    };
}

#[macro_export]
macro_rules! pydict {
    ( $($k:expr => $v:expr),* $(,)? ) => {{
        let __d = ::pyre_object::w_dict_new();
        $(
            unsafe {
                ::pyre_object::w_dict_store(
                    __d,
                    $crate::PywrapKind::into_py($k),
                    $crate::PywrapKind::into_py($v),
                );
            }
        )*
        __d
    }};
}

#[macro_export]
macro_rules! pyset {
    ( $($e:expr),* $(,)? ) => {
        ::pyre_object::w_set_from_items(&[ $( $crate::PywrapKind::into_py($e) ),* ])
    };
}

/// Per-type wrap trait consumed by `pylist!` / `pytuple!` / `pydict!`
/// / `pyset!`.  Each `impl` covers one literal kind; the `PyObjectRef`
/// passthrough impl lets users mix already-wrapped values with
/// literals (`pylist![1i64, w_int_new(2), "abc"]`).
pub trait PywrapKind {
    fn into_py(self) -> ::pyre_object::PyObjectRef;
}

impl PywrapKind for i64 {
    #[inline]
    fn into_py(self) -> ::pyre_object::PyObjectRef {
        ::pyre_object::w_int_new(self)
    }
}
impl PywrapKind for i32 {
    #[inline]
    fn into_py(self) -> ::pyre_object::PyObjectRef {
        ::pyre_object::w_int_new(self as i64)
    }
}
impl PywrapKind for u32 {
    #[inline]
    fn into_py(self) -> ::pyre_object::PyObjectRef {
        ::pyre_object::w_int_new(self as i64)
    }
}
impl PywrapKind for usize {
    #[inline]
    fn into_py(self) -> ::pyre_object::PyObjectRef {
        ::pyre_object::w_int_new(self as i64)
    }
}
impl PywrapKind for f64 {
    #[inline]
    fn into_py(self) -> ::pyre_object::PyObjectRef {
        ::pyre_object::w_float_new(self)
    }
}
impl PywrapKind for bool {
    #[inline]
    fn into_py(self) -> ::pyre_object::PyObjectRef {
        ::pyre_object::w_bool_from(self)
    }
}
impl PywrapKind for &str {
    #[inline]
    fn into_py(self) -> ::pyre_object::PyObjectRef {
        ::pyre_object::w_str_new(self)
    }
}
impl PywrapKind for String {
    #[inline]
    fn into_py(self) -> ::pyre_object::PyObjectRef {
        ::pyre_object::w_str_new(&self)
    }
}
impl PywrapKind for ::pyre_object::PyObjectRef {
    #[inline]
    fn into_py(self) -> ::pyre_object::PyObjectRef {
        self
    }
}

/// `raise oefmt(space.w_ValueError, "fmt", args)` equivalent.  Each
/// `bail_*_error!` expands to `return Err(crate::PyError::*_error(
/// format!(...)))`, mirroring PyPy's `oefmt` short-circuit that
/// constructs an `OperationError` and raises it in one line.
///
/// ```ignore
/// bail_value_error!("expected positive int, got {n}");
/// bail_type_error!("expected str, got {}", typename);
/// ```
#[macro_export]
macro_rules! bail_value_error {
    ($($t:tt)+) => { return ::std::result::Result::Err($crate::PyError::value_error(format!($($t)+))) };
}
#[macro_export]
macro_rules! bail_type_error {
    ($($t:tt)+) => { return ::std::result::Result::Err($crate::PyError::type_error(format!($($t)+))) };
}
#[macro_export]
macro_rules! bail_attr_error {
    ($($t:tt)+) => { return ::std::result::Result::Err($crate::PyError::attribute_error(format!($($t)+))) };
}
#[macro_export]
macro_rules! bail_key_error {
    ($($t:tt)+) => { return ::std::result::Result::Err($crate::PyError::key_error(format!($($t)+))) };
}
#[macro_export]
macro_rules! bail_index_error {
    ($($t:tt)+) => { return ::std::result::Result::Err($crate::PyError::index_error(format!($($t)+))) };
}
#[macro_export]
macro_rules! bail_runtime_error {
    ($($t:tt)+) => { return ::std::result::Result::Err($crate::PyError::runtime_error(format!($($t)+))) };
}
#[macro_export]
macro_rules! bail_os_error {
    ($($t:tt)+) => { return ::std::result::Result::Err($crate::PyError::os_error(format!($($t)+))) };
}
#[macro_export]
macro_rules! bail_overflow_error {
    ($($t:tt)+) => { return ::std::result::Result::Err($crate::PyError::overflow_error(format!($($t)+))) };
}
#[macro_export]
macro_rules! bail_zero_division {
    ($($t:tt)+) => { return ::std::result::Result::Err($crate::PyError::zero_division(format!($($t)+))) };
}
#[macro_export]
macro_rules! bail_not_implemented {
    ($($t:tt)+) => { return ::std::result::Result::Err($crate::PyError::not_implemented(format!($($t)+))) };
}

// ── Re-exports ──
pub use baseobjspace::*;
pub use builtins::*;
pub use display::*;
pub use error::*;
pub use executioncontext::*;
pub use function::*;
pub use gateway::{
    BUILTIN_CODE_TYPE, BuiltinCode, BuiltinCodeFn, FLATPYCALL, HOPELESS, MethodOwner,
    PASSTHROUGHARGS1, Signature, SignatureBuilder, builtin_code_call, builtin_code_get,
    builtin_code_get_fast_natural_arity, builtin_code_get_signature, builtin_code_name,
    builtin_code_new, builtin_code_new_passthrough_args1, builtin_code_new_with_arity,
    builtin_code_new_with_signature, builtin_code_no_keyword_arguments, is_builtin_code,
    make_builtin_function, make_builtin_function_as_builtin_with_signature,
    make_builtin_function_maybe_sig, make_builtin_function_passthrough_args1,
    make_builtin_function_with_arity, make_builtin_function_with_arity_and_maybe_sig,
    make_builtin_function_with_signature, make_method_descriptor_with_arity,
    make_module_builtin_function, make_module_builtin_function_with_arity,
    make_module_builtin_function_with_arity_and_maybe_sig, make_module_builtin_function_with_doc,
    make_slot_wrapper, make_slot_wrapper_with_arity,
};
pub use jit_fnaddr::*;
pub use majit_rlib::rbigint::RBigInt as PyBigInt;
pub use opcode_ops::*;
pub use pycode::*;
pub use pyframe::*;
pub use pyopcode::*;
pub use pytraceback::*;
pub use runtime_ops::*;
pub use shared_opcode::*;

/// RustPython's compiler and marshal crates expose their serialized integer
/// constants as `malachite_bigint::BigInt`.  Convert at that fixed API seam;
/// interpreter objects and every arithmetic operation use RPython's rbigint.
#[majit_macros::dont_look_inside]
pub(crate) fn compiler_bigint_to_rbigint(value: &malachite_bigint::BigInt) -> PyBigInt {
    let (sign, bytes) = value.to_bytes_le();
    let sign = match sign {
        malachite_bigint::Sign::Minus => majit_rlib::rbigint::RBigIntSign::Minus,
        malachite_bigint::Sign::NoSign => majit_rlib::rbigint::RBigIntSign::NoSign,
        malachite_bigint::Sign::Plus => majit_rlib::rbigint::RBigIntSign::Plus,
    };
    PyBigInt::from_bytes_le(sign, &bytes)
}

/// Pointer-ABI residual for the immutable compiler-constant seam above.
///
/// The code object's compiler BigInt is green/stable and the conversion is a
/// pure function of its bytes.  RPython's LOAD_CONST reads a prebuilt bigint
/// reference; in pyre's current compiler API shape, CALL_PURE/CSE of this
/// residual supplies the equivalent single GC reference without tracing into
/// Malachite internals.
#[majit_macros::elidable_or_memerror]
pub extern "C" fn jit_compiler_bigint_to_rbigint(value: i64) -> *mut PyBigInt {
    let value = value as *const malachite_bigint::BigInt;
    let converted = unsafe { compiler_bigint_to_rbigint(&*value) };
    pyre_object::longobject::alloc_bigint_nursery_collecting(converted)
}

pub(crate) fn rbigint_to_compiler_bigint(value: &PyBigInt) -> malachite_bigint::BigInt {
    let sign = match value.sign() {
        majit_rlib::rbigint::RBigIntSign::Minus => malachite_bigint::Sign::Minus,
        majit_rlib::rbigint::RBigIntSign::NoSign => malachite_bigint::Sign::NoSign,
        majit_rlib::rbigint::RBigIntSign::Plus => malachite_bigint::Sign::Plus,
    };
    let magnitude = value.abs();
    let nbytes = (magnitude
        .bit_length()
        .expect("an allocated 64-bit RBigInt cannot overflow Signed bit_length")
        + 7)
        / 8;
    let bytes = magnitude
        .tobytes(nbytes, "little", false)
        .expect("unsigned magnitude always fits its exact byte length");
    malachite_bigint::BigInt::from_bytes_le(sign, &bytes)
}

/// PyPy `@unwrap_spec(...)` equivalent.  See `pyre-macros/src/lib.rs`.
pub use pyre_macros::pyre_function;

/// PyPy `class W_X(W_Root)` + `TypeDef(...)` equivalent: derives the
/// PyType static, GC type-id constants, GC pointer-offsets table, and
/// `from_obj` / `allocate` helpers from a `#[repr(C)]` struct with a
/// `pub ob: PyObject` header (auto-prepended if absent).
pub use pyre_macros::pyre_class;

/// PyPy `interp2app(W_X.method)` equivalent attached to an `impl
/// W_X { ... }` block: every typed method gains an `args: &[PyObjectRef]`
/// wrapper that downcasts `args[0]` to `&mut Self` via `from_obj`,
/// unwraps the remaining args (same engine as [`pyre_function`]), calls
/// the typed body, and re-wraps the return value.  A `pub fn
/// type_object()` accessor is generated alongside, ready to drop into
/// `py_module! { interpleveldefs: { "X" => type_object() } }`.
pub use pyre_macros::pyre_methods;

/// Re-export of [`pyre_object::lltype::PyreClassPyTypeOf`] so
/// `py_class_typed!` can name it via `$crate::PyreClassPyTypeOf` from
/// downstream module crates.
pub use pyre_object::lltype::PyreClassPyTypeOf;

/// Every interpreter-level `PyType` static that represents a
/// `PyObject`-layout type (instances carry `ob_type` at offset 0,
/// matching `rclass.OBJECT`), paired with its parent class.
///
/// Same shape as `pyre_object::pyobject::all_foreign_pytypes`: each
/// entry is a `(type, parent)` tuple consumed by the JIT registration
/// loop in `pyre/pyre-jit/src/eval.rs`. The parent feeds
/// `TypeInfo::object_subclass` so `assign_inheritance_ids`
/// (normalizecalls.py:373-389) computes the right preorder bounds.
///
/// These live here rather than in `pyre_object::pyobject` because
/// `pyre-object` cannot depend on `pyre-interpreter`.
///
/// `BUILTIN_CODE_TYPE`, `FUNCTION_TYPE`, `BUILTIN_FUNCTION_TYPE` are
/// intentionally absent: they get their own ids
/// (`BUILTIN_CODE_GC_TYPE_ID`, `FUNCTION_GC_TYPE_ID`) because the GC
/// needs the actual payload size and inline `PyObjectRef` field
/// offsets, neither of which the foreign-pytype loop can derive from
/// `sizeof(PyObject)`. Future types whose instances reach the GC
/// nursery should follow the same pre-registration pattern (see
/// `eval.rs` BuiltinCode / Function blocks).
pub fn all_foreign_pytypes() -> &'static [(
    &'static pyre_object::pyobject::PyType,
    &'static pyre_object::pyobject::PyType,
)] {
    static PYTYPES: &[(
        &pyre_object::pyobject::PyType,
        &pyre_object::pyobject::PyType,
    )] = &[
        (&crate::pycode::CODE_TYPE, &pyre_object::INSTANCE_TYPE),
        (
            &crate::pytraceback::PYTRACEBACK_TYPE,
            &pyre_object::INSTANCE_TYPE,
        ),
    ];
    PYTYPES
}

/// Managed `#[pyre_class]` types whose only inline GC edge is the header's
/// `w_class`, in the order `build_gc` must register them.  `allocate_stable`
/// puts their instances in the old generation, so the marker needs a type id
/// carrying that offset — otherwise a `class L(_thread.LockType)` instance
/// leaves its heap type unreachable.  The tail order pins the type ids the
/// alias census below and `SUBCLASS_RANGE_HIERARCHY` assert.
pub fn all_w_class_only_descriptors() -> Vec<&'static pyre_object::lltype::PyreClassDescriptor> {
    use pyre_object::lltype::PyreClassPyTypeOf;
    vec![
        <crate::module::thread::W_Lock as PyreClassPyTypeOf>::DESCRIPTOR,
        <crate::module::thread::W_RLock as PyreClassPyTypeOf>::DESCRIPTOR,
        <crate::module::thread::W_ThreadHandle as PyreClassPyTypeOf>::DESCRIPTOR,
    ]
}

/// The same edge for `#[pyre_class]` types allocated through the immortal
/// `allocate`: the collector never walks them, so their `w_class` is reached
/// by the interpreter's immortal-root walker instead of a type id, and they
/// take no place in the type-id censuses.  Each is behind a target/feature
/// gate this crate spells exactly.
pub fn all_immortal_w_class_only_descriptors()
-> Vec<&'static pyre_object::lltype::PyreClassDescriptor> {
    #[allow(unused_imports)]
    use pyre_object::lltype::PyreClassPyTypeOf;
    vec![
        // `select` is compiled out of a sandbox build (`module/mod.rs:93`), so
        // its descriptors carry that gate too.
        #[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
        <crate::module::select::interp_select::Poll as PyreClassPyTypeOf>::DESCRIPTOR,
        #[cfg(all(target_os = "macos", feature = "host_env", not(feature = "sandbox")))]
        <crate::module::select::interp_kqueue::W_Kqueue as PyreClassPyTypeOf>::DESCRIPTOR,
        #[cfg(all(target_os = "macos", feature = "host_env", not(feature = "sandbox")))]
        <crate::module::select::interp_kevent::W_Kevent as PyreClassPyTypeOf>::DESCRIPTOR,
    ]
}

/// Interpreter-owned PyType aliases in the shared GC inheritance census.
/// `pyre-object::pyobject::all_subclass_range_aliases` supplies the object
/// layer; `init_typeobjects` passes both slices to the common numbering
/// writer.
pub fn all_subclass_range_aliases() -> Vec<pyre_object::pyobject::SubclassRangeAlias> {
    use pyre_object::lltype::PyreClassPyTypeOf;
    use pyre_object::pyobject::subclass_range_alias;

    fn typed<T: PyreClassPyTypeOf>() -> &'static pyre_object::PyType {
        // Every `#[pyre_class]` descriptor points at its macro-emitted static
        // PyType for the program lifetime.
        unsafe { &*T::PYTYPE }
    }

    vec![
        subclass_range_alias(13, &crate::gateway::BUILTIN_CODE_TYPE),
        subclass_range_alias(14, &crate::function::FUNCTION_TYPE),
        subclass_range_alias(14, &crate::function::BUILTIN_FUNCTION_TYPE),
        subclass_range_alias(14, &crate::function::METHOD_DESCRIPTOR_TYPE),
        subclass_range_alias(14, &crate::function::SLOT_WRAPPER_TYPE),
        subclass_range_alias(37, &crate::pyframe::FRAME_TYPE),
        subclass_range_alias(43, &crate::pycode::CODE_TYPE),
        subclass_range_alias(44, &crate::pytraceback::PYTRACEBACK_TYPE),
        subclass_range_alias(56, typed::<crate::module::_random::W_Random>()),
        subclass_range_alias(89, typed::<crate::module::_pickle::W_Pickler>()),
        subclass_range_alias(90, typed::<crate::module::_pickle::W_Unpickler>()),
        subclass_range_alias(91, typed::<crate::module::__pypy__::W_PickleBuffer>()),
        subclass_range_alias(92, typed::<crate::module::_pickle::PicklerMemoProxy>()),
        subclass_range_alias(93, typed::<crate::module::_pickle::UnpicklerMemoProxy>()),
        // `collections.deque` W_Deque — auto-id registered at the tail of the
        // GC type chain (`build_gc`), after the coroutine / dict-view-iterator
        // slots, so its vtable alias lands at the current max tid.
        subclass_range_alias(116, typed::<crate::module::_collections::W_Deque>()),
        // Formerly-immortal iterators converted to `allocate_stable` (managed),
        // registered at the tail of the JIT `register_pyre_class` chain after
        // the pyre-object iterators (W_Struct = 125 .. W_TokenizerIter = 129).
        subclass_range_alias(125, typed::<crate::module::r#struct::W_Struct>()),
        subclass_range_alias(
            126,
            typed::<crate::module::r#struct::unpack_iter::W_UnpackIter>(),
        ),
        subclass_range_alias(127, typed::<crate::module::_collections::W_DequeIter>()),
        subclass_range_alias(128, typed::<crate::module::_collections::W_DequeRevIter>()),
        subclass_range_alias(129, typed::<crate::module::_tokenize::W_TokenizerIter>()),
        subclass_range_alias(
            130,
            typed::<crate::pyframe::frame_locals_proxy::FrameLocalsProxy>(),
        ),
        subclass_range_alias(135, typed::<crate::module::_io::W_BufferedReader>()),
        subclass_range_alias(136, typed::<crate::module::_io::W_BufferedWriter>()),
        subclass_range_alias(137, typed::<crate::module::_io::W_BufferedRWPair>()),
        subclass_range_alias(138, typed::<crate::module::_io::W_BufferedRandom>()),
        subclass_range_alias(139, typed::<crate::module::_io::W_TextIOWrapper>()),
        subclass_range_alias(140, typed::<crate::module::thread::W_Local>()),
        // `all_w_class_only_descriptors` order, registered at the absolute
        // tail of `build_gc` after `W_DequeBlock` and `W_BufferWrapper`.
        subclass_range_alias(153, typed::<crate::module::thread::W_Lock>()),
        subclass_range_alias(154, typed::<crate::module::thread::W_RLock>()),
        subclass_range_alias(155, typed::<crate::module::thread::W_ThreadHandle>()),
        // `functools.KeyWrapper` follows them at the append-only AUTO-ID
        // registration tail.
        subclass_range_alias(156, typed::<crate::module::_functools::W_KeyWrapper>()),
        // `unicodedata.UCD` and `__pypy__.Bufferable` close that tail in the
        // order `build_gc` registers them.
        subclass_range_alias(157, typed::<crate::module::unicodedata::W_UCD>()),
        subclass_range_alias(
            158,
            typed::<crate::module::__pypy__::interp_buffer::bufferable_impl::W_Bufferable>(),
        ),
        // `_io.BytesIO` registers after the `rbigint` result pair, which takes
        // 159 as a bare `with_gc_ptrs` id and carries no vtable of its own.
        subclass_range_alias(160, typed::<crate::module::_io::W_BytesIO>()),
        subclass_range_alias(161, typed::<crate::module::_io::W_StringIO>()),
        // `_json.Scanner` and `_json.Encoder` extend the append-only managed
        // payload tail without renumbering an established class.
        subclass_range_alias(162, typed::<crate::module::_json::W_Scanner>()),
        subclass_range_alias(163, typed::<crate::module::_json::W_Encoder>()),
        // `_hashlib`'s per-object digest/HMAC contexts follow their Python
        // owners and have sweep-time native-state destructors in build_gc.
        subclass_range_alias(164, typed::<crate::module::_hashlib::W_HashState>()),
        subclass_range_alias(165, typed::<crate::module::_hashlib::W_Hmac>()),
        // `gc.GcRef` keeps its raw referent as a traced wrapper field.
        // referent field is traced on the wrapper itself, as in
        // `pypy/module/gc/referents.py`.
        subclass_range_alias(166, typed::<crate::module::gc::gcref::W_GcRef>()),
        // `gc.hooks` owns its three callback references directly, matching
        // W_AppLevelHooks in pypy/module/gc/hook.py.
        subclass_range_alias(167, typed::<crate::module::gc::hook::W_AppLevelHooks>()),
        // `gc._get_stats()` returns referents.py's native W_GcStats owner.
        subclass_range_alias(168, typed::<crate::module::gc::stats::W_GcStats>()),
        // PyPy zlib stream wrappers own their native stream and lock directly.
        // Keep these unconditional entries ahead of target-gated native types.
        subclass_range_alias(169, typed::<crate::module::zlib::W_Compress>()),
        subclass_range_alias(170, typed::<crate::module::zlib::W_Decompress>()),
        subclass_range_alias(171, typed::<crate::module::zlib::W_ZlibDecompressor>()),
        // `_bz2`'s two stream objects own their libbz2 state and per-object
        // lock.  Unconditional, so they stay ahead of the target-gated types.
        subclass_range_alias(172, typed::<crate::module::_bz2::W_BZ2Compressor>()),
        subclass_range_alias(173, typed::<crate::module::_bz2::W_BZ2Decompressor>()),
        // `_lzma`'s two stream objects own their liblzma coder, unconditional
        // for the same reason.
        subclass_range_alias(174, typed::<crate::module::_lzma::W_LZMACompressor>()),
        subclass_range_alias(175, typed::<crate::module::_lzma::W_LZMADecompressor>()),
        // `_lsprof`'s profiler and stats result owners are unconditional.
        subclass_range_alias(176, typed::<crate::module::_lsprof::W_Profiler>()),
        subclass_range_alias(177, typed::<crate::module::_lsprof::W_StatsEntry>()),
        subclass_range_alias(178, typed::<crate::module::_lsprof::W_StatsSubEntry>()),
        // `posix.DirEntry` follows the unconditional native owners.
        #[cfg(not(target_arch = "wasm32"))]
        subclass_range_alias(179, typed::<crate::module::posix::W_DirEntry>()),
        // rustls-backed `_ssl` native payloads.  They are appended after the
        // last pre-existing native class in the same order `build_gc`
        // registers them, so no established type id moves.
        #[cfg(all(not(target_arch = "wasm32"), not(feature = "sandbox")))]
        subclass_range_alias(180, typed::<crate::module::_ssl::W_SSLContext>()),
        #[cfg(all(not(target_arch = "wasm32"), not(feature = "sandbox")))]
        subclass_range_alias(181, typed::<crate::module::_ssl::W_MemoryBIO>()),
        #[cfg(all(not(target_arch = "wasm32"), not(feature = "sandbox")))]
        subclass_range_alias(182, typed::<crate::module::_ssl::W_SSLSession>()),
        #[cfg(all(not(target_arch = "wasm32"), not(feature = "sandbox")))]
        subclass_range_alias(183, typed::<crate::module::_ssl::W_SSLSocket>()),
        #[cfg(all(not(target_arch = "wasm32"), not(feature = "sandbox")))]
        subclass_range_alias(184, typed::<crate::module::_ssl::W_Certificate>()),
        // `mmap.mmap` follows the optional SSL tail on ordinary Unix builds.
        // A sandbox build has no `mmap` module at all (`module/mod.rs`), so it
        // contributes no alias rather than sliding into the vacated SSL slot.
        #[cfg(all(any(unix, windows), not(feature = "sandbox")))]
        subclass_range_alias(185, typed::<crate::module::mmap::W_MMap>()),
        // Windows asyncio's Overlapped owner follows mmap at the native tail.
        // It is a non-subclassable builtin in Python, but still participates
        // in the rclass hierarchy because its managed header and retained
        // buffer/result fields are traced by the ordinary object marker.
        #[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
        subclass_range_alias(186, typed::<crate::module::_overlapped::W_Overlapped>()),
        // `_winapi.Overlapped` follows it: a second record of the same kind,
        // owning its own event and transfer buffer rather than retained
        // Python objects, so nothing of it is traced beyond the header.
        #[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
        subclass_range_alias(
            187,
            typed::<crate::module::_winapi::overlapped::W_Overlapped>(),
        ),
    ]
}

/// The rclass hierarchy present in this interpreter configuration.
///
/// `_ssl` owns five native hierarchy slots and `mmap` owns one behind them
/// wherever it is compiled.  A sandbox build has neither module, and both sit
/// at the tail of `SUBCLASS_RANGE_HIERARCHY`, so dropping that many trailing
/// entries leaves exactly the ids such a build registers.
pub fn active_subclass_range_hierarchy() -> &'static [(u32, Option<u32>)] {
    let hierarchy = pyre_object::pyobject::SUBCLASS_RANGE_HIERARCHY;
    #[cfg(all(not(target_arch = "wasm32"), feature = "sandbox"))]
    {
        const SSL_HIERARCHY_SLOTS: usize = 5;
        #[cfg(any(unix, windows))]
        const MMAP_HIERARCHY_SLOTS: usize = 1;
        #[cfg(not(any(unix, windows)))]
        const MMAP_HIERARCHY_SLOTS: usize = 0;
        // `_overlapped.Overlapped` and `_winapi.Overlapped`, both of which a
        // sandbox build leaves out.
        #[cfg(windows)]
        const OVERLAPPED_HIERARCHY_SLOTS: usize = 2;
        #[cfg(not(windows))]
        const OVERLAPPED_HIERARCHY_SLOTS: usize = 0;
        &hierarchy[..hierarchy.len()
            - SSL_HIERARCHY_SLOTS
            - MMAP_HIERARCHY_SLOTS
            - OVERLAPPED_HIERARCHY_SLOTS]
    }
    #[cfg(not(all(not(target_arch = "wasm32"), feature = "sandbox")))]
    {
        hierarchy
    }
}

// ── Print / stderr hooks for wasm (fd-1 / fd-2 capture) ──
//
// An embedder installs these to receive everything the interpreter writes to
// fd 1 and fd 2. The sink belongs to the *process*, not to whichever thread
// installed it: a traceback or warning raised on any interpreter thread has to
// reach the same embedder, and on wasm32 `std::io::stderr().write_all`
// discards the bytes outright, so a thread that saw no hook would lose them.
// Both are plain `fn` pointers, so one atomic word each holds them and the
// write path takes no lock.
use std::sync::atomic::{AtomicUsize, Ordering};

static PRINT_HOOK: AtomicUsize = AtomicUsize::new(0);
static STDERR_HOOK: AtomicUsize = AtomicUsize::new(0);

fn store_hook(slot: &AtomicUsize, hook: fn(&[u8])) {
    slot.store(hook as usize, Ordering::Release);
}

fn load_hook(slot: &AtomicUsize) -> Option<fn(&[u8])> {
    match slot.load(Ordering::Acquire) {
        0 => None,
        // SAFETY: the slot only ever holds a `fn(&[u8])` written by
        // `store_hook`, and a function pointer is pointer-sized.
        raw => Some(unsafe { std::mem::transmute::<usize, fn(&[u8])>(raw) }),
    }
}

/// Set a hook that receives all fd-1 output instead of stdout.
///
/// The hook takes bytes rather than `&str` so a write the filesystem or a
/// `sys.stdout.buffer` caller made is handed over unmodified; decoding it is
/// the embedder's decision, not a lossy conversion applied on the way out.
pub fn set_print_hook(hook: fn(&[u8])) {
    store_hook(&PRINT_HOOK, hook);
}

/// Offer already-encoded `bytes` to the print hook. Returns whether a hook
/// consumed them; `false` leaves the caller on its own descriptor path.
pub fn print_hook_emit_bytes(bytes: &[u8]) -> bool {
    match load_hook(&PRINT_HOOK) {
        Some(hook) => {
            hook(bytes);
            true
        }
        None => false,
    }
}

/// [`print_hook_emit_bytes`] for callers holding a `str`.
pub fn print_hook_emit(s: &str) -> bool {
    print_hook_emit_bytes(s.as_bytes())
}

/// Apply the named standard stream's newline translation to text going to
/// fd 1/2.
///
/// The substitution is what `TextIOWrapper.write` would make
/// (`module/_io/textio.rs` `write_newline`), which neither path that actually
/// reaches fd 1/2 goes through — `sys.stdout.write` is an instance builtin
/// that encodes straight to the descriptor, and diagnostics (tracebacks,
/// warnings, the displayhook) arrive at `host_seam::emit_stdout` /
/// `emit_stderr` — so each applies it itself.  It cannot move down to the
/// descriptor write: `sys.stdout.buffer.write` is binary and stays
/// untranslated.
///
/// A plain substitution, `'\r'` included: `TextIOWrapper` runs
/// `text.replace('\n', writenl)`, so `'a\r\n'` really does come out `'a\r\r\n'`.
pub fn stdio_line_endings<'a>(text: &'a str, stream_name: &str) -> std::borrow::Cow<'a, str> {
    match stdio_line_endings_bytes(text.as_bytes(), stream_name) {
        std::borrow::Cow::Borrowed(_) => std::borrow::Cow::Borrowed(text),
        // SAFETY: the substitution replaces an ASCII '\n' with ASCII text, so
        // well-formed utf-8 stays well-formed.
        std::borrow::Cow::Owned(bytes) => {
            std::borrow::Cow::Owned(unsafe { String::from_utf8_unchecked(bytes) })
        }
    }
}

/// [`stdio_line_endings`] for output already encoded.  Only for the diagnostic
/// seam, whose bytes this interpreter produced as utf-8: a byte-level
/// substitution is not safe for an arbitrary stdio encoding — utf-16 spells
/// every ASCII character with a 0x00 beside it — which is why
/// `sys.stdout.write` translates its text before encoding instead.
pub fn stdio_line_endings_bytes<'a>(
    bytes: &'a [u8],
    stream_name: &str,
) -> std::borrow::Cow<'a, [u8]> {
    // The mode is read off the live stream, so the cheap test comes first:
    // output carrying no newline needs no mode and asks `sys` for nothing.
    if !bytes.contains(&b'\n') {
        return std::borrow::Cow::Borrowed(bytes);
    }
    let Some(newline) = stdio_newline(stream_name).filter(|nl| *nl != "\n") else {
        return std::borrow::Cow::Borrowed(bytes);
    };
    let mut out = Vec::with_capacity(bytes.len() + 8);
    for &byte in bytes {
        if byte == b'\n' {
            out.extend_from_slice(newline.as_bytes());
        } else {
            out.push(byte);
        }
    }
    std::borrow::Cow::Owned(out)
}

/// The substitution the named standard stream's own `write` would make.
///
/// Read from the live stream rather than from the platform, so that
/// `sys.stdout.reconfigure(newline=...)` reaches the paths that do not go
/// through `TextIOWrapper.write`.
///
/// These paths write to the descriptor whatever `sys` holds, so anything that
/// states no mode of its own — the seam runs before `sys` exists and after it
/// is torn down, and the name may have been rebound to something that is not a
/// stream at all — leaves them on the mode `allocate_stdio` builds with.
pub fn stdio_newline(stream_name: &str) -> Option<&'static str> {
    let platform_default = if cfg!(windows) { Some("\r\n") } else { None };
    let stated = crate::importing::get_sys_module("sys")
        .and_then(|sys| crate::baseobjspace::getattr_str(sys, stream_name).ok())
        .and_then(crate::module::_io::W_TextIOWrapper::stdio_write_newline);
    stated.unwrap_or(platform_default)
}

/// Write a string through the print hook (if set) or stdout.
pub fn print_output(s: &str) {
    // `print()` writing to the unmodified `sys.stdout` short-circuits to here
    // instead of calling its `write`, so the standard stream's newline
    // translation has to be applied on this path too.
    let s = stdio_line_endings(s, "stdout");
    let s = s.as_ref();
    if print_hook_emit(s) {
        return;
    }
    // Under sandbox fd 1 is the marshalling pipe, so route program
    // output through ll_os_write(1,…) for the controller to relay; a
    // raw `print!` would corrupt the protocol stream.
    #[cfg(all(unix, feature = "sandbox"))]
    let _ = crate::host_seam::ops::write(1, s.as_bytes());
    #[cfg(not(all(unix, feature = "sandbox")))]
    print!("{s}");
    crate::host_seam::flush_stdout_when_unbuffered();
}

/// Set a hook that receives everything the interpreter writes to fd 2 —
/// `sys.stderr.write`, tracebacks, warnings — instead of the real descriptor.
///
/// The wasm32 target has no descriptors: `std::io::stderr().write_all` there
/// discards the bytes, so without a hook a traceback simply vanishes. The
/// stdout twin is [`set_print_hook`].
pub fn set_stderr_hook(hook: fn(&[u8])) {
    store_hook(&STDERR_HOOK, hook);
}

/// Offer `bytes` to the stderr hook. Returns whether a hook consumed them;
/// `false` leaves the caller on its own descriptor path.
pub fn stderr_hook_emit(bytes: &[u8]) -> bool {
    match load_hook(&STDERR_HOOK) {
        Some(hook) => {
            hook(bytes);
            true
        }
        None => false,
    }
}

#[cfg(test)]
mod bigint_seam_tests {
    use super::*;

    #[test]
    fn compiler_bigint_byte_seam_round_trips_sign_and_large_magnitude() {
        for source in [
            "0",
            "1",
            "-1",
            "9223372036854775808",
            "-9223372036854775809",
            "12345678901234567890123456789012345678901234567890",
        ] {
            let compiler: malachite_bigint::BigInt = source.parse().unwrap();
            let rbigint = compiler_bigint_to_rbigint(&compiler);
            assert_eq!(rbigint.str(0).unwrap(), source);
            assert_eq!(rbigint_to_compiler_bigint(&rbigint), compiler);
        }
    }
}

// baseobjspace call helpers are re-exported from `baseobjspace`.
