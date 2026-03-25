use pyre_object::{PYOBJECT_ARRAY_LEN_OFFSET, PyObjectArray, PyObjectRef};

use crate::new_builtin_namespace;

/// Byte offset of the value-array storage inside `PyNamespace`.
pub const PYNAMESPACE_VALUES_OFFSET: usize = std::mem::offset_of!(PyNamespace, values);

/// Byte offset of the live namespace slot count inside `PyNamespace`.
pub const PYNAMESPACE_VALUES_LEN_OFFSET: usize =
    PYNAMESPACE_VALUES_OFFSET + PYOBJECT_ARRAY_LEN_OFFSET;

/// Name-based Python namespace used by the current interpreter subset.
///
/// Names are stored in insertion order and values live in a pointer-backed
/// `PyObjectArray`, giving the JIT a stable slot model for hot global loads and
/// stores.
#[repr(C)]
pub struct PyNamespace {
    names: Vec<String>,
    values: PyObjectArray,
    /// JIT invalidation watchers. When a namespace slot is overwritten,
    /// all watchers are notified (AtomicBool set to true) so that
    /// compiled code using constant-folded globals is invalidated.
    /// RPython parity: quasi-immutable field invalidation for module dicts.
    watchers: Vec<std::sync::Weak<std::sync::atomic::AtomicBool>>,
}

impl Clone for PyNamespace {
    fn clone(&self) -> Self {
        let mut namespace = Self {
            names: self.names.clone(),
            values: PyObjectArray::from_vec(self.values.to_vec()),
            watchers: Vec::new(), // cloned namespace is a new identity
        };
        namespace.fix_ptr();
        namespace
    }
}

impl Default for PyNamespace {
    fn default() -> Self {
        Self::new()
    }
}

impl PyNamespace {
    pub fn new() -> Self {
        let mut namespace = Self {
            names: Vec::new(),
            values: PyObjectArray::from_vec(Vec::new()),
            watchers: Vec::new(),
        };
        namespace.fix_ptr();
        namespace
    }

    #[inline]
    pub fn fix_ptr(&mut self) {
        self.values.fix_ptr();
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.names.len()
    }

    #[inline]
    pub fn slot_of(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|candidate| candidate == name)
    }

    #[inline]
    pub fn get(&self, name: &str) -> Option<&PyObjectRef> {
        self.slot_of(name).map(|idx| &self.values[idx])
    }

    #[inline]
    pub fn get_slot(&self, idx: usize) -> Option<PyObjectRef> {
        self.values.as_slice().get(idx).copied()
    }

    pub fn get_or_insert_with(
        &mut self,
        name: &str,
        make: impl FnOnce() -> PyObjectRef,
    ) -> PyObjectRef {
        if let Some(idx) = self.slot_of(name) {
            return self.values[idx];
        }
        let value = make();
        self.names.push(name.to_string());
        self.values.push(value);
        value
    }

    pub fn insert(&mut self, name: String, value: PyObjectRef) -> Option<PyObjectRef> {
        if let Some(idx) = self.slot_of(&name) {
            let old = self.values[idx];
            self.values[idx] = value;
            if old != value {
                self.notify_watchers();
            }
            Some(old)
        } else {
            self.names.push(name);
            self.values.push(value);
            None
        }
    }

    #[inline]
    pub fn set_slot(&mut self, idx: usize, value: PyObjectRef) -> bool {
        let slice = self.values.as_mut_slice();
        let Some(slot) = slice.get_mut(idx) else {
            return false;
        };
        let old = *slot;
        *slot = value;
        if old != value {
            self.notify_watchers();
        }
        true
    }

    /// Register a JIT invalidation watcher. When any namespace slot
    /// is overwritten, the flag is set to true, causing
    /// GUARD_NOT_INVALIDATED to fail in compiled code.
    pub fn register_watcher(&mut self, flag: &std::sync::Arc<std::sync::atomic::AtomicBool>) {
        self.watchers.push(std::sync::Arc::downgrade(flag));
    }

    fn notify_watchers(&mut self) {
        if self.watchers.is_empty() {
            return;
        }
        self.watchers.retain(|w| {
            if let Some(flag) = w.upgrade() {
                flag.store(true, std::sync::atomic::Ordering::Release);
                true
            } else {
                false
            }
        });
    }

    #[inline]
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.names.iter()
    }
}

/// Shared execution context for all frames in one interpreter run.
///
/// Holds the builtin namespace seed.  Module-level frames call
/// `fresh_namespace()` once to create a leaked globals dict;
/// function calls share the globals pointer without cloning.
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

    /// Create a fresh module/global namespace seeded with builtins.
    ///
    /// The caller is responsible for leaking it via `Box::into_raw`
    /// so it can be shared across frames as a raw pointer.
    pub fn fresh_namespace(&self) -> PyNamespace {
        self.builtins.clone()
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
    fn test_namespace_slots_stay_stable_when_appending_names() {
        let mut namespace = PyNamespace::new();
        namespace.insert("x".to_string(), pyre_object::w_int_new(1));
        assert_eq!(namespace.slot_of("x"), Some(0));

        namespace.insert("y".to_string(), pyre_object::w_int_new(2));
        assert_eq!(namespace.slot_of("x"), Some(0));
        assert_eq!(namespace.slot_of("y"), Some(1));
    }
}
