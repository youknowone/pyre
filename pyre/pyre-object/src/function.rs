//! `pypy/interpreter/function.py` method descriptor ports.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::pyobject::*;
use pyre_macros::pyre_class;

// ── Method ───────────────────────────────────────────────────────────
// PyPy: pypy/interpreter/function.py Method

/// Python bound method wrapper.
#[pyre_class("method", type_id = 16, static_name = "METHOD")]
pub struct Method {
    pub w_function: PyObjectRef,
    pub w_self: PyObjectRef,
    pub w_class: PyObjectRef,
    /// CPython 3.14 `PyCFunctionObject.m_module` for builtin-bound methods.
    ///
    /// PyPy's `_Method` has no public `__module__` storage of its own; pyre
    /// keeps this version-selected field on the same bound object rather than
    /// in a side table. `PY_NULL` for ordinary Python methods, `None` initially
    /// for a bound builtin method.
    pub w_module: PyObjectRef,
}

/// Field offsets of inline `PyObjectRef` slots within `Method`.
/// Consumed by `pyre-jit-trace/src/descr.rs` to emit field-access IR;
/// the macro's own `W_METHOD_GC_PTR_OFFSETS` aggregate is independent
/// and does not depend on these per-field consts.
pub const METHOD_W_FUNCTION_OFFSET: usize = std::mem::offset_of!(Method, w_function);
pub const METHOD_W_SELF_OFFSET: usize = std::mem::offset_of!(Method, w_self);
pub const METHOD_W_CLASS_OFFSET: usize = std::mem::offset_of!(Method, w_class);
pub const METHOD_W_MODULE_OFFSET: usize = std::mem::offset_of!(Method, w_module);

pub fn w_method_new(
    w_function: PyObjectRef,
    w_self: PyObjectRef,
    w_class: PyObjectRef,
) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`): pin the
    // three members across the GC malloc and re-read their relocated
    // addresses afterwards (a minor collection inside the malloc may move
    // them). A bound method whose `w_function`/`w_self`/`w_class` is
    // reachable only through it must be GC-traced; a `malloc_typed` method
    // is invisible to mark-sweep, whereas `register_pyre_class` registers
    // this layout's `ptr_offsets`, so mark-sweep follows the members. The
    // write barrier below keeps the old-gen method in the remembered set so
    // young members survive a later minor collection.
    let _roots = crate::gc_roots::push_roots();
    let save_point = crate::gc_roots::shadow_stack_len();
    crate::gc_roots::pin_root(w_function);
    crate::gc_roots::pin_root(w_self);
    crate::gc_roots::pin_root(w_class);
    let header = PyObject {
        ob_type: &METHOD_TYPE as *const PyType,
        w_class: get_instantiate(&METHOD_TYPE),
    };
    let raw = crate::gc_hook::try_gc_alloc_stable_raw(W_METHOD_GC_TYPE_ID, W_METHOD_OBJECT_SIZE);
    let w_module = PY_NULL;
    if !raw.is_null() {
        // Remember the old-gen shell before publishing nursery members. The
        // barrier may park behind a collection; running it after the stores
        // would leave a movable `w_self` (notably a list) untraced during that
        // window. This is the same pre-store shape as tupleobject's stable
        // shell plus young items block.
        crate::gc_hook::try_gc_write_barrier_managed(raw);
        for slot in save_point..save_point + 3 {
            let root = crate::gc_roots::shadow_stack_get(slot);
            let current =
                crate::gc_hook::try_gc_current_object_address(root as *mut u8) as PyObjectRef;
            crate::gc_roots::shadow_stack_set(slot, current);
        }
        let w_function = crate::gc_roots::shadow_stack_get(save_point);
        let w_self = crate::gc_roots::shadow_stack_get(save_point + 1);
        let w_class = crate::gc_roots::shadow_stack_get(save_point + 2);
        unsafe {
            std::ptr::write(
                raw as *mut Method,
                Method {
                    ob: header,
                    w_function,
                    w_self,
                    w_class,
                    w_module,
                },
            );
        }
        return raw as PyObjectRef;
    }
    let w_function = crate::gc_roots::shadow_stack_get(save_point);
    let w_self = crate::gc_roots::shadow_stack_get(save_point + 1);
    let w_class = crate::gc_roots::shadow_stack_get(save_point + 2);
    Method::allocate(Method {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        w_function,
        w_self,
        w_class,
        w_module,
    })
}

/// Select a CPython-compatible public type for a Method-layout object while
/// preserving PyPy's raw `METHOD_TYPE` payload and GC descriptor.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_method_set_public_class(obj: PyObjectRef, w_class: PyObjectRef) {
    crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
    (*obj).w_class = w_class;
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_method_get_module(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const Method)).w_module
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_method_set_module(obj: PyObjectRef, w_module: PyObjectRef) {
    crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
    (*(obj as *mut Method)).w_module = w_module;
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_method(obj: PyObjectRef) -> bool {
    py_type_check(obj, &METHOD_TYPE)
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_method_get_func(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const Method)).w_function
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_method_get_self(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const Method)).w_self
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_method_get_class(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const Method)).w_class
}

// ── StaticMethod ─────────────────────────────────────────────────────
// PyPy: pypy/interpreter/function.py StaticMethod
//
// __get__ returns the wrapped function unchanged (no self binding).

/// Python staticmethod descriptor.
#[pyre_class("staticmethod", type_id = 20, static_name = "STATICMETHOD")]
pub struct StaticMethod {
    pub w_function: PyObjectRef,
    /// function.py:676 `self.w_dict = None` — lazily allocated by
    /// `StaticMethod.getdict`, then populated by `descr_init` with the
    /// wrapped function's presentation attributes.
    pub w_dict: PyObjectRef,
    /// The hidden `mutate_w_function` field for `function.py:673
    /// _immutable_fields_ = ['w_function?']` — see [`crate::quasiimmut`].
    ///
    /// Holds no GC pointers, so the derived `W_STATICMETHOD_GC_PTR_OFFSETS` has
    /// nothing to walk here, and it is deliberately absent from the descr
    /// census: it is an `AtomicPtr` plus a lock rather than one pointer-shaped
    /// word, so a `Type::Ref` row would misdescribe it, and no emit allocates
    /// either wrapper, so `rewrite.py clear_gc_fields` never runs over this
    /// layout.  The allocation is [`crate::gc_hook::try_gc_alloc_stable_raw`],
    /// i.e. non-moving, which is [`crate::quasiimmut::QuasiImmutField`]'s
    /// stated precondition: the lock cannot be remapped out from under a
    /// holder.  Recording binds the instance onto the marker descr as an
    /// `Arc`, so the compiler never re-resolves this owner and a sweep
    /// mid-compile is safe; the sweep hook is `{static,class}method_destructor`
    /// in `pyre-jit/src/eval.rs`.
    pub w_function_watchers: crate::quasiimmut::QuasiImmutField,
}

/// Field offsets of the inline `PyObjectRef` slots within `StaticMethod`,
/// consumed by `pyre-jit-trace/src/descr.rs` on the same footing as the
/// `METHOD_*` consts above.
pub const STATICMETHOD_W_FUNCTION_OFFSET: usize = std::mem::offset_of!(StaticMethod, w_function);
pub const STATICMETHOD_W_DICT_OFFSET: usize = std::mem::offset_of!(StaticMethod, w_dict);

pub fn w_staticmethod_new(func: PyObjectRef) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`): pin the
    // wrapped function across the GC malloc and read its relocated address.
    let _roots = crate::gc_roots::push_roots();
    let save_point = crate::gc_roots::shadow_stack_len();
    crate::gc_roots::pin_root(func);

    let header = PyObject {
        ob_type: &STATICMETHOD_TYPE as *const PyType,
        w_class: get_instantiate(&STATICMETHOD_TYPE),
    };
    let raw = crate::gc_hook::try_gc_alloc_stable_raw(
        W_STATICMETHOD_GC_TYPE_ID,
        W_STATICMETHOD_OBJECT_SIZE,
    );
    let func = crate::gc_roots::shadow_stack_get(save_point);
    if !raw.is_null() {
        unsafe {
            std::ptr::write(
                raw as *mut StaticMethod,
                StaticMethod {
                    ob: header,
                    w_function: func,
                    w_dict: PY_NULL,
                    w_function_watchers: crate::quasiimmut::QuasiImmutField::new(),
                },
            );
        }
        crate::gc_hook::try_gc_write_barrier(raw);
        return raw as PyObjectRef;
    }
    StaticMethod::allocate(StaticMethod {
        ob: header,
        w_function: func,
        w_dict: PY_NULL,
        w_function_watchers: crate::quasiimmut::QuasiImmutField::new(),
    })
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_staticmethod_get_func(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const StaticMethod)).w_function
}

/// function.py:697 `self.w_function = w_function`.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_staticmethod_set_func(obj: PyObjectRef, func: PyObjectRef) {
    unsafe {
        // `rclass.py hook_setfield` emits `jit_force_quasi_immutable` ahead of
        // every store to a `?` field, so the wrapped callable stops being a
        // trace constant before it stops being the live value.  The hook
        // precedes the store and does not consult it, so re-initialising the
        // slot with the value it already holds invalidates as well.  Nothing
        // else revokes a fold over this wrapper: re-initialising an installed
        // descriptor changes no type's version tag, which is the only other pin
        // the folds that unwrap `w_function` hold.  The `is_installed` test is
        // `pyjitpl.py:1112`'s `mutatebox.nonnull()` — a wrapper no loop watches
        // pays one load.
        if (*(obj as *const StaticMethod))
            .w_function_watchers
            .is_installed()
        {
            crate::quasiimmut::sweep_quasi_immut_field(
                &(*(obj as *const StaticMethod)).w_function_watchers,
            );
        }
        (*(obj as *mut StaticMethod)).w_function = func;
        crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
    }
}

/// `quasiimmut.py:17-27 get_current_qmut_instance` for `function.py`'s
/// `w_function?` — resolved at RECORD time so a write reached later in the same
/// trace sees it, and handed back so the loop compiled from that trace
/// registers on it.
///
/// # Safety
/// `obj` must point to a live [`StaticMethod`].
pub unsafe fn w_staticmethod_current_w_function_qmut(
    obj: PyObjectRef,
) -> Option<std::sync::Arc<crate::quasiimmut::QuasiImmut>> {
    if obj.is_null() {
        return None;
    }
    Some(
        (*(obj as *const StaticMethod))
            .w_function_watchers
            .get_current_qmut_instance(),
    )
}

/// function.py:678-681 `StaticMethod.getdict` — allocate the instance
/// dictionary on first access and retain its identity.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_staticmethod_getdict(obj: PyObjectRef) -> PyObjectRef {
    unsafe {
        let sm = obj as *mut StaticMethod;
        if (*sm).w_dict.is_null() {
            (*sm).w_dict = crate::dictmultiobject::w_dict_new_instance();
            crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
        }
        (*sm).w_dict
    }
}

/// function.py:683-688 `StaticMethod.setdict`; the caller performs the
/// dict type check before replacing this field.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_staticmethod_setdict(obj: PyObjectRef, w_dict: PyObjectRef) {
    unsafe {
        (*(obj as *mut StaticMethod)).w_dict = w_dict;
        crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
    }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_staticmethod(obj: PyObjectRef) -> bool {
    py_type_check(obj, &STATICMETHOD_TYPE)
}

/// An exact `staticmethod`, excluding subclasses — the test a caller needs
/// before it may unwrap `w_function` in place of invoking `__get__`.
/// `descroperation.py:169-187 get_and_call_function` takes its descriptor
/// shortcut only on the exact type and routes every other one through
/// `space.get`, so a subclass that overrides `__get__` binds differently.
/// Compares the user-visible class object, as [`is_exact_tuple`] does, because
/// a subclass instance keeps the base layout in `ob_type` and retags `w_class`.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_exact_staticmethod(obj: PyObjectRef) -> bool {
    unsafe {
        is_staticmethod(obj) && std::ptr::eq((*obj).w_class, get_instantiate(&STATICMETHOD_TYPE))
    }
}

// ── ClassMethod ──────────────────────────────────────────────────────
// PyPy: pypy/interpreter/function.py ClassMethod
//
// __get__ returns a bound method with the class as first arg.

/// Python classmethod descriptor.
#[pyre_class("classmethod", type_id = 21, static_name = "CLASSMETHOD")]
pub struct ClassMethod {
    pub w_function: PyObjectRef,
    /// function.py:724 `self.w_dict = None` — a real per-wrapper field,
    /// allocated lazily by `ClassMethod.getdict`.
    pub w_dict: PyObjectRef,
    /// The hidden `mutate_w_function` field for `function.py:720
    /// _immutable_fields_ = ['w_function?']` — see [`crate::quasiimmut`].
    ///
    /// Holds no GC pointers, so the derived `W_CLASSMETHOD_GC_PTR_OFFSETS` has
    /// nothing to walk here, and it is deliberately absent from the descr
    /// census: it is an `AtomicPtr` plus a lock rather than one pointer-shaped
    /// word, so a `Type::Ref` row would misdescribe it, and no emit allocates
    /// either wrapper, so `rewrite.py clear_gc_fields` never runs over this
    /// layout.  The allocation is [`crate::gc_hook::try_gc_alloc_stable_raw`],
    /// i.e. non-moving, which is [`crate::quasiimmut::QuasiImmutField`]'s
    /// stated precondition: the lock cannot be remapped out from under a
    /// holder.  Recording binds the instance onto the marker descr as an
    /// `Arc`, so the compiler never re-resolves this owner and a sweep
    /// mid-compile is safe; the sweep hook is `{static,class}method_destructor`
    /// in `pyre-jit/src/eval.rs`.
    pub w_function_watchers: crate::quasiimmut::QuasiImmutField,
}

/// Field offsets of the inline `PyObjectRef` slots within `ClassMethod`, the
/// `classmethod` twin of the `STATICMETHOD_*` consts above.
pub const CLASSMETHOD_W_FUNCTION_OFFSET: usize = std::mem::offset_of!(ClassMethod, w_function);
pub const CLASSMETHOD_W_DICT_OFFSET: usize = std::mem::offset_of!(ClassMethod, w_dict);

pub fn w_classmethod_new(func: PyObjectRef) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`): pin the
    // wrapped function across the GC malloc and read its relocated address.
    let _roots = crate::gc_roots::push_roots();
    let save_point = crate::gc_roots::shadow_stack_len();
    crate::gc_roots::pin_root(func);

    let header = PyObject {
        ob_type: &CLASSMETHOD_TYPE as *const PyType,
        w_class: get_instantiate(&CLASSMETHOD_TYPE),
    };
    let raw = crate::gc_hook::try_gc_alloc_stable_raw(
        W_CLASSMETHOD_GC_TYPE_ID,
        W_CLASSMETHOD_OBJECT_SIZE,
    );
    let func = crate::gc_roots::shadow_stack_get(save_point);
    if !raw.is_null() {
        unsafe {
            std::ptr::write(
                raw as *mut ClassMethod,
                ClassMethod {
                    ob: header,
                    w_function: func,
                    w_dict: PY_NULL,
                    w_function_watchers: crate::quasiimmut::QuasiImmutField::new(),
                },
            );
        }
        crate::gc_hook::try_gc_write_barrier(raw);
        return raw as PyObjectRef;
    }
    ClassMethod::allocate(ClassMethod {
        ob: header,
        w_function: func,
        w_dict: PY_NULL,
        w_function_watchers: crate::quasiimmut::QuasiImmutField::new(),
    })
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_classmethod_get_func(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const ClassMethod)).w_function
}

/// function.py:752 `self.w_function = w_function`.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_classmethod_set_func(obj: PyObjectRef, func: PyObjectRef) {
    unsafe {
        // `rclass.py hook_setfield` emits `jit_force_quasi_immutable` ahead of
        // every store to a `?` field, so the wrapped callable stops being a
        // trace constant before it stops being the live value.  The hook
        // precedes the store and does not consult it, so re-initialising the
        // slot with the value it already holds invalidates as well.  Nothing
        // else revokes a fold over this wrapper: re-initialising an installed
        // descriptor changes no type's version tag, which is the only other pin
        // the folds that unwrap `w_function` hold.  The `is_installed` test is
        // `pyjitpl.py:1112`'s `mutatebox.nonnull()` — a wrapper no loop watches
        // pays one load.
        if (*(obj as *const ClassMethod))
            .w_function_watchers
            .is_installed()
        {
            crate::quasiimmut::sweep_quasi_immut_field(
                &(*(obj as *const ClassMethod)).w_function_watchers,
            );
        }
        (*(obj as *mut ClassMethod)).w_function = func;
        crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
    }
}

/// `quasiimmut.py:17-27 get_current_qmut_instance` for `function.py`'s
/// `w_function?` — resolved at RECORD time so a write reached later in the same
/// trace sees it, and handed back so the loop compiled from that trace
/// registers on it.
///
/// # Safety
/// `obj` must point to a live [`ClassMethod`].
pub unsafe fn w_classmethod_current_w_function_qmut(
    obj: PyObjectRef,
) -> Option<std::sync::Arc<crate::quasiimmut::QuasiImmut>> {
    if obj.is_null() {
        return None;
    }
    Some(
        (*(obj as *const ClassMethod))
            .w_function_watchers
            .get_current_qmut_instance(),
    )
}

/// function.py:726-729 `ClassMethod.getdict`.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_classmethod_getdict(obj: PyObjectRef) -> PyObjectRef {
    unsafe {
        let cm = obj as *mut ClassMethod;
        if (*cm).w_dict.is_null() {
            (*cm).w_dict = crate::dictmultiobject::w_dict_new_instance();
            crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
        }
        (*cm).w_dict
    }
}

/// function.py:731-736 `ClassMethod.setdict`; the object-space layer checks
/// for dict or a dict subclass before replacing the field.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_classmethod_setdict(obj: PyObjectRef, w_dict: PyObjectRef) {
    unsafe {
        (*(obj as *mut ClassMethod)).w_dict = w_dict;
        crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
    }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_classmethod(obj: PyObjectRef) -> bool {
    py_type_check(obj, &CLASSMETHOD_TYPE)
}

/// An exact `classmethod`, excluding subclasses — the `classmethod` twin of
/// [`is_exact_staticmethod`], for the same reason.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_exact_classmethod(obj: PyObjectRef) -> bool {
    unsafe {
        is_classmethod(obj) && std::ptr::eq((*obj).w_class, get_instantiate(&CLASSMETHOD_TYPE))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Guard against drift between the constant colocated with
    /// `Method` and the id that `pyre-jit/src/eval.rs` asserts at
    /// JitDriver init. Mirror of the W_CELL/FUNCTION trip-wire tests.
    #[test]
    fn w_method_gc_type_id_matches_descr() {
        assert_eq!(W_METHOD_GC_TYPE_ID, 16);
        assert_eq!(
            <Method as crate::lltype::GcType>::type_id(),
            W_METHOD_GC_TYPE_ID
        );
        assert_eq!(
            <Method as crate::lltype::GcType>::SIZE,
            W_METHOD_OBJECT_SIZE
        );
    }

    #[test]
    fn w_staticmethod_gc_type_id_matches_descr() {
        assert_eq!(W_STATICMETHOD_GC_TYPE_ID, 20);
        assert_eq!(
            <StaticMethod as crate::lltype::GcType>::type_id(),
            W_STATICMETHOD_GC_TYPE_ID
        );
        assert_eq!(
            <StaticMethod as crate::lltype::GcType>::SIZE,
            W_STATICMETHOD_OBJECT_SIZE
        );
    }

    #[test]
    fn w_classmethod_gc_type_id_matches_descr() {
        assert_eq!(W_CLASSMETHOD_GC_TYPE_ID, 21);
        assert_eq!(
            <ClassMethod as crate::lltype::GcType>::type_id(),
            W_CLASSMETHOD_GC_TYPE_ID
        );
        assert_eq!(
            <ClassMethod as crate::lltype::GcType>::SIZE,
            W_CLASSMETHOD_OBJECT_SIZE
        );
    }
}
