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
