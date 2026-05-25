#![allow(ambiguous_glob_reexports, dead_code, unused_assignments, unused_unsafe)]

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
pub mod argument;
pub mod baseobjspace;
pub mod builtins;
pub mod display;
pub mod error;
pub mod exception_table;
pub mod executioncontext;
pub mod frame_array;
pub mod function;
pub mod gateway;
pub mod jit_fnaddr;
pub mod listobject;
pub mod opcode_ops;
pub mod pycode;
pub mod pyopcode;
pub mod pytraceback;
pub mod runtime_ops;
pub mod sandbox;
pub mod shared_opcode;
pub mod sliceobject;
pub mod stack_check;
pub mod structseq;
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

// ── Declarative builtin-module registration ──
//
// `pypy/module/<name>/moduledef.py` declares its surface as a dict
// literal — `interpleveldefs = { 'name': 'interp_x.func', ... }` — and
// PyPy's MixedModule machinery walks the dict at import time.  Pyre
// mirrors that with the `py_module!` macro below: each call expands to
// a `pub fn init(ns: &mut DictStorage)` that stores every entry via
// `dict_storage_store`.  The previous one-line `moduledef.rs` shim
// (which did nothing but `super::interp_x::register_module(ns)`) has
// been retired across every builtin module.
//
// The macro is intentionally minimal at the value layer — each entry's
// RHS is just a `PyObjectRef` expression — so call-site code stays the
// same as the hand-written `dict_storage_store` calls it replaces.  An
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
///         "getweakrefcount" / 1 = interp_weakref::getweakrefcount,
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
        pub fn init(ns: &mut $crate::DictStorage) {
            let _name = $name;
            $($(
                $crate::dict_storage_store(ns, $key, $value);
            )*)?
            // inline_functions: `#[pyre_function]` typed defs whose name +
            // arity are derived from the signature.  Replaces the
            // separate `#[pyre_function] fn X` + `"X" / N = X` pair.
            $($(
                {
                    #[$crate::pyre_function]
                    fn $ifn_name ( $($ifn_args)* ) $(-> $ifn_ret)? $ifn_body
                    $crate::dict_storage_store(
                        ns,
                        stringify!($ifn_name),
                        $crate::make_builtin_function_with_arity(
                            stringify!($ifn_name),
                            $ifn_name,
                            $crate::pyre_count_typed_args!($($ifn_args)*) as u16,
                        ),
                    );
                }
            )*)?
            $($(
                $crate::dict_storage_store(
                    ns, $fn_key,
                    $crate::py_module_fn!($fn_key, $fn_arity, $fn_path),
                );
            )*)?
            $($(
                $crate::dict_storage_store(
                    ns, $mfn_key,
                    $crate::py_module_module_fn!($mfn_key, $mfn_arity, $mfn_path),
                );
            )*)?
            $(
                {
                    let $ns: &mut $crate::DictStorage = ns;
                    $body
                }
            )?
        }
    };
}

/// Count typed `name: ty` arguments in a `fn` parameter list.  Used by
/// `py_module!`'s `inline_functions:` arm to derive arity from the
/// signature.  Treats `&[PyObjectRef]` varargs as zero (caller should
/// route those through path-ref `functions:` form instead).
#[macro_export]
macro_rules! pyre_count_typed_args {
    () => { 0usize };
    ($a:ident : $t:ty) => { 1usize };
    ($a:ident : $t:ty, $($rest:tt)*) => {
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
///             crate::baseobjspace::setattr(self_obj, "_state", ::pyre_object::w_int_new(seed))?;
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
                                $crate::dict_storage_store(
                                    ns,
                                    stringify!($mname),
                                    $crate::make_builtin_function(stringify!($mname), $mname),
                                );
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
                                $crate::dict_storage_store(
                                    ns,
                                    stringify!($pname),
                                    $crate::typedef::make_getset_descriptor_named(
                                        $crate::make_builtin_function(stringify!($pname), $pname),
                                        stringify!($pname),
                                    ),
                                );
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

/// Helper for `py_module!`'s `functions:` arm.  `*` → varargs
/// (`make_builtin_function`); numeric arity → `make_builtin_function_with_arity`.
#[macro_export]
macro_rules! py_module_fn {
    ($key:literal, *, $path:expr) => {
        $crate::make_builtin_function($key, $path)
    };
    ($key:literal, $arity:literal, $path:expr) => {
        $crate::make_builtin_function_with_arity($key, $path, $arity)
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

// ── Re-exports ──
pub use baseobjspace::*;
pub use builtins::*;
pub use display::*;
pub use error::*;
pub use executioncontext::*;
pub use function::*;
pub use gateway::{
    BUILTIN_CODE_TYPE, BuiltinCode, BuiltinCodeFn, FLATPYCALL, HOPELESS, PASSTHROUGHARGS1,
    builtin_code_get, builtin_code_get_fast_natural_arity, builtin_code_name, builtin_code_new,
    builtin_code_new_passthrough_args1, builtin_code_new_with_arity, is_builtin_code,
    make_builtin_function, make_builtin_function_passthrough_args1,
    make_builtin_function_with_arity, make_module_builtin_function,
    make_module_builtin_function_with_arity,
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
            print!("{s}");
        }
    });
}

// baseobjspace call helpers are re-exported from `baseobjspace`.
