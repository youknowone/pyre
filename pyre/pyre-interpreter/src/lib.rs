//! pyre-interpreter — the Python interpreter.
//!
//! PyPy equivalent: pypy/interpreter/
//!
//! Contains the object space (space.rs), bytecode evaluation (eval.rs),
//! frame management (frame.rs), function call dispatch (call.rs),
//! import machinery (importing.rs), builtin functions (builtins.rs),
//! type definitions (typedef.rs), and builtin modules (module/).

// ── Former pyre-runtime modules ──
pub mod builtinfunc;
pub mod builtins;
pub mod codeobject;
pub mod display;
pub mod error;
pub mod executioncontext;
pub mod frame_array;
pub mod funcobject;
pub mod opcode_ops;
pub mod opcode_step;
pub mod runtime_ops;
pub mod shared_opcode;
pub mod space;
pub mod type_methods;
pub mod typedef;

// ── Former pyre-interp modules ──
pub mod call;
pub mod eval;
pub mod frame;
pub mod importing;
pub mod module;

// ── Re-exports ──
pub use builtinfunc::*;
pub use builtins::*;
pub use codeobject::*;
pub use display::*;
pub use error::*;
pub use executioncontext::*;
pub use frame_array::*;
pub use funcobject::*;
pub use malachite_bigint::BigInt as PyBigInt;
pub use opcode_ops::*;
pub use opcode_step::*;
pub use runtime_ops::*;
pub use shared_opcode::*;
pub use space::*;

// ── space.call_function ──────────────────────────────────────────────
//
// PyPy: baseobjspace.py call_function — unified callable dispatch.
// Now a direct function (no callback needed — interpreter is in the same crate).

/// Call a Python callable with the given arguments.
///
/// PyPy: `space.call_function(w_callable, *args_w)`
///
/// Dispatches to builtins, user functions, and type objects.
pub fn space_call_function(
    callable: pyre_object::PyObjectRef,
    args: &[pyre_object::PyObjectRef],
) -> pyre_object::PyObjectRef {
    call::space_call_function_impl(callable, args)
}

/// Try calling `obj.__dunder__()`, return obj itself if dunder not found.
pub fn space_call_function_or_identity(
    obj: pyre_object::PyObjectRef,
    dunder: &str,
) -> pyre_object::PyObjectRef {
    unsafe {
        if pyre_object::is_instance(obj) {
            let w_type = pyre_object::w_instance_get_type(obj);
            if let Some(method) = space::lookup_in_type_mro_pub(w_type, dunder) {
                return space_call_function(method, &[obj]);
            }
        }
    }
    obj
}
