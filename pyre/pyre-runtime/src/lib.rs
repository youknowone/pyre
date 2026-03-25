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
pub mod typedef;

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

// ── space.call_function — PyPy baseobjspace.py parity ────────────────
//
// Single callback from pyre-runtime → pyre-interp for calling any
// Python callable (user function, builtin, type) with arbitrary args.
// PyPy equivalent: `space.call_function(w_callable, *args_w)`.
//
// All other callbacks (DUNDER_CALLER, PROPERTY_CALLER, BUILD_CLASS_IMPL,
// DUNDER_BINOP_CALLER, DUNDER_UNARY_CALLER) are removed in favor of
// this single entry point.

use std::sync::OnceLock;

/// `space.call_function(callable, *args)` → PyObjectRef
///
/// Registered once at startup by pyre-interp. Handles dispatch to
/// builtin functions, user functions (frame creation + eval), and
/// type objects (instance creation + __init__).
pub type SpaceCallFn =
    fn(pyre_object::PyObjectRef, &[pyre_object::PyObjectRef]) -> pyre_object::PyObjectRef;

static SPACE_CALL_FUNCTION: OnceLock<SpaceCallFn> = OnceLock::new();

pub fn register_space_call_function(f: SpaceCallFn) {
    let _ = SPACE_CALL_FUNCTION.set(f);
}

/// Call a Python callable with the given arguments.
///
/// PyPy: `space.call_function(w_callable, *args_w)`
///
/// Falls back to builtin dispatch if the interp callback is not yet registered.
pub fn space_call_function(
    callable: pyre_object::PyObjectRef,
    args: &[pyre_object::PyObjectRef],
) -> pyre_object::PyObjectRef {
    if let Some(f) = SPACE_CALL_FUNCTION.get() {
        return f(callable, args);
    }
    // Fallback: try builtin dispatch only
    unsafe {
        if is_builtin_func(callable) {
            let func = w_builtin_func_get(callable);
            return func(args);
        }
    }
    panic!("space_call_function: no implementation registered and callable is not a builtin");
}

/// Try calling `obj.__dunder__()`, return obj itself if dunder not found.
///
/// Used by operator.index() etc.
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
