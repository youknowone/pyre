//! `pypy/interpreter/module.py` — Python `module` type.
//!
//! A module holds a name (str) and its backing dict object.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::pyobject::*;

/// Python module object.
///
/// Layout: `[ob_type | name | w_dict]`
///
/// `w_dict` mirrors PyPy `module.py:20 self.w_dict = w_dict` — every
/// Module owns a non-null `W_DictObject` (or dict subclass instance
/// for the user-supplied wrap case at `moduledef.py:102-103`).  For
/// ordinary Modules pyre constructs a `W_ModuleDictObject`, so
/// `getdict(space)` returns a stable identity across calls. For the user-supplied
/// case `w_dict` is the caller's object directly, preserving subclass
/// identity for `space.finditem_str` dispatch.
#[repr(C)]
pub struct Module {
    pub ob_header: PyObject,
    /// Heap-allocated module name string.
    pub name: *mut String,
    /// Authoritative dict object (`PyPy module.w_dict`).  Always non-null
    /// after construction.
    pub w_dict: PyObjectRef,
}

/// GC type id assigned to `Module` at JitDriver init time.
pub const W_MODULE_GC_TYPE_ID: u32 = 36;

/// Fixed payload size (`framework.py:811`).
pub const W_MODULE_OBJECT_SIZE: usize = std::mem::size_of::<Module>();

/// Byte offset of the inline `w_dict: PyObjectRef` slot — the GC must
/// trace the aliased `W_DictObject` (`pypy/interpreter/module.py:22
/// self.w_dict = w_dict`) so a Module surviving a minor collection
/// keeps the user-supplied dict alive.  `name`/`dict` are non-PyObject
/// raw heap pointers and are intentionally absent; they are owned via
/// `lltype::malloc_raw` and traced through their own type ids.
pub const W_MODULE_GC_PTR_OFFSETS: [usize; 1] = [std::mem::offset_of!(Module, w_dict)];

impl crate::lltype::GcType for Module {
    fn type_id() -> u32 {
        W_MODULE_GC_TYPE_ID
    }
    const SIZE: usize = W_MODULE_OBJECT_SIZE;
}

/// Allocate a new Module backed by a fresh `W_ModuleDictObject`. Use this
/// for `space.builtin`, freshly-imported modules, REPL `__main__`, and
/// other Modules whose authoritative namespace is their dict object.
///
/// `module.py:24` — `if w_name is not None: setitem(w_dict, '__name__',
/// w_name)`. Pyre seeds `__name__` through `w_dict_setitem_str`.
///
/// `name` — the module name (e.g. "math", "os.path"); empty string is
///   the anonymous-name sentinel for `pick_builtin`'s default Module
///   case (`moduledef.py:106-108`, PyPy `Module(space, None, ...)`)
///   in which `Module.__init__` skips the `__name__` setitem.
pub fn w_module_new(name: &str) -> PyObjectRef {
    // `pypy/interpreter/module.py:18 Module.__init__` opens
    // `w_dict = space.newdict(module=True)` per `dictmultiobject.py:440-451
    // _newdict(module=True)`, which lands on `W_ModuleDictObject`
    // (ModuleDictStrategy + cell-cache). Pyre routes through
    // `w_module_dict_new`; `pypy/objspace/std/celldict.py` strategy semantics
    // (`get_global_cache`, `invalidate_caches`,
    // `switch_to_object_strategy`) cover the module surface.
    let name_box = crate::lltype::malloc_raw(name.to_string());
    let w_dict = crate::dictmultiobject::w_module_dict_new();
    if !name.is_empty() {
        unsafe {
            crate::dictmultiobject::w_dict_setitem_str(w_dict, "__name__", crate::w_str_new(name));
        }
    }
    crate::lltype::malloc_typed(Module {
        ob_header: PyObject {
            ob_type: &MODULE_TYPE as *const PyType,
            w_class: get_instantiate(&MODULE_TYPE),
        },
        name: name_box,
        w_dict,
    }) as PyObjectRef
}

/// Allocate a `Module` aliasing a user-supplied `W_DictObject`.
/// Mirrors `pypy/module/__builtin__/moduledef.py:102-103
/// module.Module(space, None, w_builtin)`: the Module's dict identity
/// IS the user dict (PyPy `module.w_dict = w_builtin`).
///
/// This is the direct port of PyPy's `Module(space, None, w_builtin)` for dict subclasses:
/// `LOAD_GLOBAL` falls through to `space.finditem_str(module.w_dict,
/// name)` so subclass `__getitem__` overrides are not bypassed.
///
/// `name` seeding (`pypy/interpreter/module.py:24`): when `name` is a
/// non-empty string, set `w_dict["__name__"] = name` so
/// `module.__name__` resolves and `from module import *`,
/// `import_from` submodule fallback work.  PyPy's
/// `Module.__init__(space, w_name, w_dict)` does `space.setitem(w_dict,
/// space.newtext("__name__"), w_name)` when `w_name is not None`; pyre
/// honours the same contract here so every caller gets `__name__`
/// without duplicating the seeding step at each callsite.  When
/// `w_dict` is a non-`W_DictObject` (dict subclass instance), the
/// setitem is skipped — the subclass's own `__init__` is responsible
/// for seeding `__name__` (matching PyPy `moduledef.py:102-103
/// Module(space, None, w_builtin)` where `w_name=None`).
///
/// `#[dont_look_inside]` (`@jit.dont_look_inside`, `rlib/jit.py:139`):
/// the body performs an unported `lltype::malloc_typed` NewWithVtable
/// (`Module`) that survives `fuse_boxing_alloc` unfused, so the JIT
/// residualises the whole call to a stable runtime fnaddr instead of
/// tracing the allocation. The `-> PyObjectRef` result is a plain GCREF with no
/// discriminant to erase.
#[majit_macros::dont_look_inside]
pub fn w_module_new_aliasing_dict(name: &str, w_dict_object: PyObjectRef) -> PyObjectRef {
    if !name.is_empty() && !w_dict_object.is_null() && unsafe { crate::is_dict(w_dict_object) } {
        unsafe {
            crate::dictmultiobject::w_dict_setitem_str(
                w_dict_object,
                "__name__",
                crate::w_str_new(name),
            );
        }
    }
    let name = crate::lltype::malloc_raw(name.to_string());
    crate::lltype::malloc_typed(Module {
        ob_header: PyObject {
            ob_type: &MODULE_TYPE as *const PyType,
            w_class: get_instantiate(&MODULE_TYPE),
        },
        name,
        w_dict: w_dict_object,
    }) as PyObjectRef
}

/// Get the module name.
///
/// # Safety
/// `obj` must point to a valid `Module`.
pub unsafe fn w_module_get_name(obj: PyObjectRef) -> &'static str {
    let module = &*(obj as *const Module);
    &*module.name
}

/// Replace the module name (`module.py:24` re-seeding).  Used by
/// `module.__init__(name, doc)` after `module.__new__` allocates an
/// anonymous module.  The previous (immortal) name string is leaked.
///
/// # Safety
/// `obj` must point to a valid `Module`.
pub unsafe fn w_module_set_name(obj: PyObjectRef, name: &str) {
    let module = &mut *(obj as *mut Module);
    module.name = crate::lltype::malloc_raw(name.to_string());
}

/// Get the aliased `W_DictObject` (`PY_NULL` when storage-only).
///
/// # Safety
/// `obj` must point to a valid `Module`.
pub unsafe fn w_module_get_w_dict(obj: PyObjectRef) -> PyObjectRef {
    let module = &*(obj as *const Module);
    module.w_dict
}

/// pypy/interpreter/module.py:Module.getdictvalue —
/// `space.finditem_str(self.w_dict, attr)`.  When `w_dict` is a real
/// `W_DictObject` pyre routes through `w_dict_getitem_str` (which
/// honours the storage-proxy read-through, so storage-only Modules
/// surface storage entries via the same call).  When `w_dict` is a
/// dict subclass instance the caller must take the
/// `space.finditem_str` dispatch path itself (subclass `__getitem__`
/// override) — pyre-object can't reach the interpreter's dispatcher,
/// so we return `None` and rely on the storage fallback at the
/// caller (`eval.rs:load_global_value`).
///
/// # Safety
/// `obj` must point to a valid `Module`.
pub unsafe fn w_module_alias_getitem_str(obj: PyObjectRef, name: &str) -> Option<PyObjectRef> {
    let module = &*(obj as *const Module);
    if module.w_dict.is_null() {
        return None;
    }
    // `W_ModuleDictObject` (`module.py:18 newdict(module=True)`) joins
    // `W_DictObject` here so `w_dict_getitem_str` (which dispatches via
    // the strategy slot) reaches both module-strategy and object-strategy
    // backings.  Subclass instances still fall through to None so the
    // caller (`eval.rs:load_global_value`) takes the
    // `space.finditem_str` dispatch path with the subclass's own
    // `__getitem__`.
    if !crate::is_dict(module.w_dict) && !crate::dictmultiobject::is_module_dict(module.w_dict) {
        return None;
    }
    crate::dictmultiobject::w_dict_getitem_str(module.w_dict, name)
}

/// Check if an object is a module.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_module(obj: PyObjectRef) -> bool {
    py_type_check(obj, &MODULE_TYPE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_create_and_check() {
        let obj = w_module_new("test_mod");
        unsafe {
            assert!(is_module(obj));
            assert!(!is_int(obj));
            assert_eq!(w_module_get_name(obj), "test_mod");
        }
    }
}
