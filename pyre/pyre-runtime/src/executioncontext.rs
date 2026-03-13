use std::collections::HashMap;

use pyre_object::PyObjectRef;

use crate::{install_default_builtins, new_builtin_namespace};

/// Name-based Python namespace used by the current interpreter subset.
pub type PyNamespace = HashMap<String, PyObjectRef>;

/// Shared execution context for all frames in one interpreter run.
///
/// This is the first step toward PyPy's `ExecutionContext` layering:
/// frame creation, builtin seeding, and per-run interpreter state live here
/// instead of inside `PyFrame`.
#[derive(Clone)]
pub struct PyExecutionContext {
    builtins: PyNamespace,
}

impl Default for PyExecutionContext {
    fn default() -> Self {
        Self::new()
    }
}

impl PyExecutionContext {
    pub fn new() -> Self {
        Self {
            builtins: new_builtin_namespace(),
        }
    }

    /// Create a fresh module/global namespace for a top-level frame.
    pub fn fresh_namespace(&self) -> PyNamespace {
        self.builtins.clone()
    }

    /// Inherit the caller namespace for a function call and ensure builtins exist.
    pub fn inherit_namespace(&self, caller_namespace: &PyNamespace) -> PyNamespace {
        let mut namespace = caller_namespace.clone();
        self.ensure_builtins(&mut namespace);
        namespace
    }

    /// Ensure default builtins exist in the given namespace.
    pub fn ensure_builtins(&self, namespace: &mut PyNamespace) {
        for (name, &value) in &self.builtins {
            namespace.entry(name.clone()).or_insert(value);
        }
        install_default_builtins(namespace);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::is_builtin_func;

    #[test]
    fn test_fresh_namespace_starts_with_builtins() {
        let ctx = PyExecutionContext::new();
        let namespace = ctx.fresh_namespace();

        let print = *namespace.get("print").unwrap();
        let range = *namespace.get("range").unwrap();

        unsafe {
            assert!(is_builtin_func(print));
            assert!(is_builtin_func(range));
        }
    }

    #[test]
    fn test_inherit_namespace_keeps_user_bindings() {
        let ctx = PyExecutionContext::new();
        let mut caller_namespace = ctx.fresh_namespace();
        caller_namespace.insert("answer".to_string(), pyre_object::w_int_new(42));

        let namespace = ctx.inherit_namespace(&caller_namespace);

        unsafe {
            let answer = *namespace.get("answer").unwrap();
            assert_eq!(pyre_object::w_int_get_value(answer), 42);
            assert!(is_builtin_func(*namespace.get("print").unwrap()));
        }
    }
}
