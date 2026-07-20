#![allow(
    ambiguous_glob_reexports,
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
pub mod baseobjspace;
pub mod builtins;
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

// On non-unix targets (wasm) the seam is configured out, but the diagnostic
// stdio emitters need neither libc nor the sandbox trampoline (sandbox is
// unix-only). Provide them so the shared `crate::host_seam::emit_*` call sites
// resolve everywhere; the bodies mirror the non-sandbox emit path.
#[cfg(not(unix))]
pub mod host_seam {
    /// Emit bytes to the interpreter's stdout (fd 1).
    pub fn emit_stdout(bytes: &[u8]) {
        use std::io::Write;
        let _ = std::io::stdout().write_all(bytes);
    }

    /// Emit bytes to the interpreter's stderr (fd 2).
    pub fn emit_stderr(bytes: &[u8]) {
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
pub mod listobject;
pub mod opcode_ops;
pub mod pycode;
pub mod pyopcode;
pub mod pytraceback;
pub mod reduce_protocol;
pub mod runtime_ops;
pub mod shared_opcode;
pub mod sliceobject;
pub mod stack_check;
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
                        $crate::make_module_builtin_function_with_arity_and_maybe_sig(
                            stringify!($ifn_name),
                            $ifn_name,
                            $crate::pyre_count_typed_args!($($ifn_args)*) as u16,
                            ::paste::paste! { [<$ifn_name _pyre_sig>]() },
                        ),
                    );
                }
            )*)?
            $($(
                $crate::module_ns_store(
                    ns, $fn_key,
                    $crate::py_module_fn!($fn_key, $fn_arity, $fn_path),
                );
            )*)?
            $($(
                $crate::module_ns_store(
                    ns, $mfn_key,
                    $crate::py_module_module_fn!($mfn_key, $mfn_arity, $mfn_path),
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

/// Count typed `name: ty` arguments in a `fn` parameter list.  Used by
/// `py_module!`'s `inline_functions:` arm to derive arity from the
/// signature.  Treats `&[PyObjectRef]` varargs as zero (caller should
/// route those through path-ref `functions:` form instead).  Each
/// parameter may carry leading attributes (`#[default(...)]`,
/// `#[kwonly]`, `#[kwargs]`) — they are consumed and ignored here; the
/// `#[pyre_function]` expansion strips them from the emitted fn.
#[macro_export]
macro_rules! pyre_count_typed_args {
    () => { 0usize };
    ( $(#[$m:meta])* $a:ident : $t:ty ) => { 1usize };
    ( $(#[$m:meta])* $a:ident : $t:ty, $($rest:tt)* ) => {
        1usize + $crate::pyre_count_typed_args!($($rest)*)
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
///     thread_local! { static CELL: ... = const { ... }; }
///     CELL.with(|c| *c.get_or_init(|| {
///         let tp = crate::typedef::make_builtin_type("_random.Random", |ns| {
///             #[crate::pyre_function]
///             fn __init__(self_obj: PyObjectRef, seed: i64) -> Result<(), crate::PyError> { ... }
///             crate::dict_storage_store(ns, "__init__",
///                 crate::make_builtin_function_with_arity("__init__", __init__, 2));
///             // ... more methods
///         });
///         unsafe { ::pyre_object::typeobject::w_type_set_hasdict(tp, true) };
///         tp
///     }))
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
            thread_local! {
                static CELL: ::std::cell::OnceCell<::pyre_object::PyObjectRef>
                    = const { ::std::cell::OnceCell::new() };
            }
            CELL.with(|c| {
                *c.get_or_init(|| {
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
                    tp
                })
            })
        }
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
            thread_local! {
                static CELL: ::std::cell::OnceCell<::pyre_object::PyObjectRef>
                    = const { ::std::cell::OnceCell::new() };
            }
            CELL.with(|c| {
                *c.get_or_init(|| {
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
                    tp
                })
            })
        }
    };
}

/// Helper for `py_module!`'s `functions:` arm.  `*` → varargs
/// (`make_module_builtin_function`); numeric arity →
/// `make_module_builtin_function_with_arity`.  Functions directly in a
/// mixed-module are non-descriptors (`mixedmodule.py:_load_lazily` converts
/// every mixed-module function to `BuiltinFunction`, whose typedef omits
/// `__get__`), so storing one on a user class must not synthesize a bound
/// method — identical to the `module_functions:` arm.
#[macro_export]
macro_rules! py_module_fn {
    ($key:literal, *, $path:expr) => {
        $crate::make_module_builtin_function($key, $path)
    };
    ($key:literal, $arity:literal, $path:expr) => {
        $crate::make_module_builtin_function_with_arity($key, $path, $arity)
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
        $crate::make_module_builtin_function_with_arity($key, $path, $arity)
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
    BUILTIN_CODE_TYPE, BuiltinCode, BuiltinCodeFn, FLATPYCALL, HOPELESS, PASSTHROUGHARGS1,
    Signature, SignatureBuilder, builtin_code_get, builtin_code_get_fast_natural_arity,
    builtin_code_get_signature, builtin_code_name, builtin_code_new,
    builtin_code_new_passthrough_args1, builtin_code_new_with_arity,
    builtin_code_new_with_signature, is_builtin_code, make_builtin_function,
    make_builtin_function_maybe_sig, make_builtin_function_passthrough_args1,
    make_builtin_function_with_arity, make_builtin_function_with_arity_and_maybe_sig,
    make_builtin_function_with_signature, make_module_builtin_function,
    make_module_builtin_function_with_arity, make_module_builtin_function_with_arity_and_maybe_sig,
};
pub use jit_fnaddr::*;
pub use malachite_bigint::BigInt as PyBigInt;
pub use opcode_ops::*;
pub use pycode::*;
pub use pyframe::*;
pub use pyopcode::*;
pub use pytraceback::*;
pub use runtime_ops::*;
pub use shared_opcode::*;

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
    ]
}

// ── Print hook for wasm (stdout capture) ──
use std::cell::RefCell;
thread_local! {
    static PRINT_HOOK: RefCell<Option<fn(&str)>> = RefCell::new(None);
}

/// Set a hook that receives all `print()` output instead of stdout.
pub fn set_print_hook(hook: fn(&str)) {
    PRINT_HOOK.with(|h| *h.borrow_mut() = Some(hook));
}

/// Write a string through the print hook (if set) or stdout.
pub fn print_output(s: &str) {
    PRINT_HOOK.with(|h| {
        if let Some(hook) = *h.borrow() {
            hook(s);
        } else {
            // Under sandbox fd 1 is the marshalling pipe, so route program
            // output through ll_os_write(1,…) for the controller to relay; a
            // raw `print!` would corrupt the protocol stream.
            #[cfg(all(unix, feature = "sandbox"))]
            let _ = crate::host_seam::ops::write(1, s.as_bytes());
            #[cfg(not(all(unix, feature = "sandbox")))]
            print!("{s}");
        }
    });
}

// baseobjspace call helpers are re-exported from `baseobjspace`.
