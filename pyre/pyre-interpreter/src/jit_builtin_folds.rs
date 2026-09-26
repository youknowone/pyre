//! Builtin identities the JIT walker's remaining folds recognize by the
//! `BuiltinCodeFn` a callable wraps rather than by the name it is reachable
//! under.

use pyre_object::PyObjectRef;

/// Whether `callable` is the `repr` builtin itself.
///
/// `repr(i)` and `str(i)` on an exact `int` render the same decimal text, so
/// the walker's int-render fold answers for both; this names the second
/// callable by identity rather than by the name it is reachable under, so a
/// rebound global keeps the residual.
pub fn is_repr_builtin(callable: PyObjectRef) -> bool {
    unsafe {
        builtin_code_fn_of(callable).is_some_and(|found| {
            crate::gateway::builtin_code_fn_eq(found, crate::builtins::builtin_repr)
        })
    }
}

/// The `BuiltinCodeFn` a callable's wrapped code holds, or `None` when the
/// callable is not a builtin-code function at all.
pub unsafe fn builtin_code_fn_of(callable: PyObjectRef) -> Option<crate::gateway::BuiltinCodeFn> {
    unsafe {
        if callable.is_null() || !crate::is_function(callable) {
            return None;
        }
        let code = crate::function_get_code(callable) as PyObjectRef;
        if code.is_null() || !crate::gateway::is_builtin_code(code) {
            return None;
        }
        Some(crate::gateway::builtin_code_get(code))
    }
}
