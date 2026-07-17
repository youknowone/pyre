//! Function object.
#![allow(non_snake_case)]
//!
//! Wraps a code object pointer, a function name, a pointer to the
//! defining module's globals namespace, and an optional closure tuple.
//! When called, the interpreter creates a new PyFrame that *shares*
//! the globals pointer (no clone).

use pyre_object::pyobject::*;

/// Type descriptor for user-defined functions.
pub static FUNCTION_TYPE: PyType = pyre_object::pyobject::new_pytype("function");
/// Type descriptor for module-level builtins.
pub static BUILTIN_FUNCTION_TYPE: PyType =
    pyre_object::pyobject::new_pytype("builtin_function_or_method");

/// User-defined function object.
///
/// Layout: `[ob_type | code | can_change_code | name_ptr | closure]`
/// - `code`: pointer to a Code object (PyCode for user funcs, BuiltinCode for builtins).
///   function.py:47 — `_immutable_fields_ = ['code?', ...]`
/// - `can_change_code`: function.py:33 — True by default; False for
///   `FunctionWithFixedCode` subclass (used by builtins).
/// - `name_ptr`: off-GC `Box<String>` containing the function name
/// - `closure`:  tuple of cell objects, or PY_NULL if no closure
/// - `w_func_globals_obj`: the module namespace dict object (`__globals__`)
#[repr(C)]
pub struct Function {
    pub ob: PyObject,
    /// Pointer to a Code object (PyCode or BuiltinCode).
    /// function.py:47 — `_immutable_fields_ = ['code?', ...]`
    pub code: *const (),
    /// function.py:33 — `can_change_code = True`
    /// False for FunctionWithFixedCode subclass.
    pub can_change_code: bool,
    /// Function name (off-GC Box<String>).
    pub name: *const String,
    /// Closure: tuple of cell objects from the enclosing scope,
    /// or PY_NULL if this function has no free variables.
    pub closure: PyObjectRef,
    /// Default argument values.
    /// PyPy: Function.defs_w
    pub defs_w: PyObjectRef,
    /// Keyword-only default values.
    /// PyPy: Function.w_kw_defs
    pub w_kw_defs: PyObjectRef,
    /// function.py:56 — `self.w_module = None`
    pub w_module: PyObjectRef,
    /// PyPy: Function.w_func_globals — the module namespace dict object.
    ///
    /// `function.py:57 self.w_func_globals = w_globals` stores the dict
    /// object directly; this is the function's sole globals carrier, so
    /// `function.__globals__` returns the same identity as the module's
    /// `__dict__` and frames built from this function share globals.
    /// `PY_NULL` for globals-less carriers (gateway builtins).
    pub w_func_globals_obj: PyObjectRef,
    /// CPython 3.14 `PyFunctionObject.func_builtins` — resolved once from
    /// globals at function construction and then exposed by the direct
    /// read-only `__builtins__` member.  This is intentionally distinct from
    /// frame builtin selection: replacing `globals['__builtins__']` later
    /// does not change `function.__builtins__`.
    pub w_builtins: PyObjectRef,
    /// `function.py:50 w_ann=None` constructor default plus
    /// `function.py:548-551 fget_func_annotations` lazy-init shape:
    /// PyPy stores the annotations dict directly on the function and
    /// allocates an empty dict on first read when none was set.
    /// Pyre mirrors that by keeping the slot `PY_NULL` until either
    /// `MAKE_FUNCTION ANNOTATIONS` flag stamps the compile-time dict
    /// (`function.py:553-559 fset_func_annotations`) or the getter
    /// lazy-fills with `w_dict_new()`.  `f.__annotations__ is
    /// f.__annotations__` identity holds because both reads see the
    /// cached slot after the first allocation.
    pub w_ann: PyObjectRef,
    /// PEP 649 `__annotate__` callable (CPython 3.14 `func_annotate`
    /// slot; `MAKE_FUNCTION` `Annotate` flag) — evaluated with
    /// `format=1` by the `__annotations__` getter when `w_ann` is
    /// still unset, then discarded in favour of the stamped dict.
    /// `PY_NULL` when the compiler emitted eager annotations or none.
    /// No PyPy counterpart (upstream function.py targets 3.11, before
    /// PEP 649); the typed slot mirrors how `w_ann` sits on the
    /// function object rather than a side table.
    pub w_annotate: PyObjectRef,
    /// `function.py:68 self.w_func_dict = None` — lazily allocated
    /// per-function attribute dictionary.  This is a field on the function,
    /// not mapdict side storage, matching PyPy's `Function.getdict`.
    pub w_func_dict: PyObjectRef,
    /// CPython 3.14 `PyFunctionObject.func_typeparams` — the declared type
    /// parameters tuple.  `PY_NULL` represents the default empty tuple.
    pub w_typeparams: PyObjectRef,
    /// `function.py:375 self.w_doc = w_doc` constructor slot plus
    /// `function.py:446-449 fget_func_doc` cache:
    ///
    /// ```python
    /// def fget_func_doc(self, space):
    ///     if self.w_doc is None:
    ///         self.w_doc = self.code.getdocstring(space)
    ///     return self.w_doc
    /// ```
    ///
    /// `PY_NULL` means "not yet resolved" (PyPy `None`); first reader
    /// caches `code.getdocstring()` here so subsequent
    /// `f.__doc__ is f.__doc__` holds.  `function_del_doc` writes
    /// `w_none()` to make the deleted state sticky against the lazy
    /// fallback (`function.py:455-457 fdel_func_doc`).
    pub w_doc: PyObjectRef,
    /// `function.py:54 self.qualname = qualname or self.name` slot —
    /// PyPy stores the qualified name as a regular field on the
    /// function object.  `PY_NULL` means "not explicitly set"; the
    /// getter falls back to `code.co_qualname` then `name` to mirror
    /// the constructor's `qualname or self.name` short-circuit when
    /// the compile path did not stamp a value at MAKE_FUNCTION time.
    pub w_qualname: PyObjectRef,
    /// `function.py:498-504 fget_func_objclass / set_objclass` slot —
    /// PyPy stores the bound class on the function for descriptor
    /// introspection (`inspect.getfullargspec` etc.).  `PY_NULL`
    /// means "not bound to a class" and the getter raises
    /// AttributeError per `function.py:500`.
    pub w_objclass: PyObjectRef,
    /// `function.py:487-496 fget_func_text_signature /
    /// fset_func_text_signature` slot — PyPy stores the text
    /// signature (the docstring-prefix `(...)` line that PEP 437
    /// describes for builtins) directly on the function.  `PY_NULL`
    /// means "no signature recorded" and the getter returns
    /// `space.w_None` per `function.py:488`.
    pub w_text_signature: PyObjectRef,
    /// `__self__` of a builtin `__new__` descriptor — the type whose
    /// `tp_new` this function wraps.  `typeobject.c add_tp_new_wrapper`
    /// binds `int.__new__.__self__ is int`, so `copyreg._reduce_ex`
    /// can ask `base.__new__.__self__ is base` to find the defining
    /// type.  `PY_NULL` for every ordinary function (a plain `def`
    /// has no `__self__`); only stamped on the per-type builtin
    /// `__new__` carriers at type-finalisation time.
    pub w_new_self: PyObjectRef,
    /// function.py:797-815 `BuiltinFunction.w_moduleobj` — the module object
    /// bound as `__self__` for an interp-level module function. PyPy stores
    /// this on each BuiltinFunction; CPython 3.14 likewise exposes the
    /// defining module object from `PyCFunction_GET_SELF`.
    pub w_moduleobj: PyObjectRef,
}

/// function.py:706 — `class BuiltinFunction(Function): can_change_code = False`
pub type BuiltinFunction = Function;
/// function.py:703 — `class FunctionWithFixedCode(Function): can_change_code = False`
pub type FunctionWithFixedCode = Function;
pub type Method = pyre_object::function::Method;
pub type StaticMethod = pyre_object::function::StaticMethod;
pub type ClassMethod = pyre_object::function::ClassMethod;

struct FrameLocalsRoot {
    slot: *mut *mut u8,
    registered: bool,
}

impl FrameLocalsRoot {
    fn new(frame: &mut crate::pyframe::PyFrame) -> Self {
        let slot = &mut frame.locals_cells_stack_w as *mut _ as *mut *mut u8;
        let registered = unsafe { pyre_object::gc_hook::try_gc_add_root(slot) };
        Self { slot, registered }
    }
}

impl Drop for FrameLocalsRoot {
    fn drop(&mut self) {
        if self.registered {
            pyre_object::gc_hook::try_gc_remove_root(self.slot);
        }
    }
}

#[inline]
fn function_write_barrier(obj: PyObjectRef) {
    pyre_object::gc_hook::try_gc_write_barrier(obj as *mut u8);
    // A Box-immortal function's children are reached only through
    // `walk_raw_function_roots`, which clean minor collections skip;
    // record every field store (gc_roots.rs prebuilt-root tracking).
    // RPython's GC transform inserts the old-to-young `write_barrier`
    // (minimark.py:1065) after such post-alloc field stores; pyre has no
    // transform pass, so callers run it by hand through this helper.
    pyre_object::gc_roots::mark_prebuilt_roots_dirty();
}

/// Field offset of `code` within `Function`, for JIT field access.
pub const FUNCTION_CODE_OFFSET: usize = std::mem::offset_of!(Function, code);
/// Field offset of `name` within `Function`.
pub const FUNCTION_NAME_OFFSET: usize = std::mem::offset_of!(Function, name);
/// Field offset of `closure` within `Function`.
pub const FUNCTION_CLOSURE_OFFSET: usize = std::mem::offset_of!(Function, closure);
/// Field offset of `defs_w` within `Function`.
pub const FUNCTION_DEFS_W_OFFSET: usize = std::mem::offset_of!(Function, defs_w);
/// Field offset of `w_kw_defs` within `Function`.
pub const FUNCTION_W_KW_DEFS_OFFSET: usize = std::mem::offset_of!(Function, w_kw_defs);
/// Field offset of `w_module` within `Function`.
pub const FUNCTION_W_MODULE_OFFSET: usize = std::mem::offset_of!(Function, w_module);
/// Field offset of `w_func_globals_obj` within `Function` — the
/// lazy-cached canonical W_DictObject for `w_func_globals`.
pub const FUNCTION_W_FUNC_GLOBALS_OBJ_OFFSET: usize =
    std::mem::offset_of!(Function, w_func_globals_obj);
/// Field offset of CPython 3.14 `func_builtins`.
pub const FUNCTION_W_BUILTINS_OFFSET: usize = std::mem::offset_of!(Function, w_builtins);
/// Field offset of `w_ann` within `Function` — the
/// `function.py:50 w_ann` annotations dict slot.
pub const FUNCTION_W_ANN_OFFSET: usize = std::mem::offset_of!(Function, w_ann);
/// Field offset of `w_annotate` within `Function` — the PEP 649
/// `__annotate__` callable slot.
pub const FUNCTION_W_ANNOTATE_OFFSET: usize = std::mem::offset_of!(Function, w_annotate);
/// Field offset of PyPy `Function.w_func_dict`.
pub const FUNCTION_W_FUNC_DICT_OFFSET: usize = std::mem::offset_of!(Function, w_func_dict);
/// Field offset of CPython 3.14 `func_typeparams`.
pub const FUNCTION_W_TYPEPARAMS_OFFSET: usize = std::mem::offset_of!(Function, w_typeparams);
/// Field offset of `w_doc` within `Function` — the
/// `function.py:375 w_doc` docstring cache slot.
pub const FUNCTION_W_DOC_OFFSET: usize = std::mem::offset_of!(Function, w_doc);
/// Field offset of `w_qualname` within `Function` — the
/// `function.py:54 qualname` slot.
pub const FUNCTION_W_QUALNAME_OFFSET: usize = std::mem::offset_of!(Function, w_qualname);
/// Field offset of `w_objclass` within `Function` — the
/// `function.py:498-504 w_objclass` slot.
pub const FUNCTION_W_OBJCLASS_OFFSET: usize = std::mem::offset_of!(Function, w_objclass);
/// Field offset of `w_text_signature` within `Function` — the
/// `function.py:487-496 w_text_signature` slot.
pub const FUNCTION_W_TEXT_SIGNATURE_OFFSET: usize =
    std::mem::offset_of!(Function, w_text_signature);
/// Field offset of `w_new_self` within `Function` — the builtin
/// `__new__` descriptor's `__self__` (defining type).
pub const FUNCTION_W_NEW_SELF_OFFSET: usize = std::mem::offset_of!(Function, w_new_self);
/// Field offset of PyPy `BuiltinFunction.w_moduleobj`.
pub const FUNCTION_W_MODULEOBJ_OFFSET: usize = std::mem::offset_of!(Function, w_moduleobj);

/// GC type id assigned to `Function` at JitDriver init time. Held as
/// a constant alongside the struct (rather than runtime-queried) so
/// the allocation hook can reach it without a back-channel, mirroring
/// `W_INT_GC_TYPE_ID` / `W_FLOAT_GC_TYPE_ID` / `BUILTIN_CODE_GC_TYPE_ID`.
/// `pyre/pyre-jit/src/eval.rs` asserts the same id is returned by
/// `gc.register_type(...)` so any drift panics on startup.
///
/// `BuiltinFunction` (gateway.py module-level builtins) and
/// `FunctionWithFixedCode` are Rust type aliases of `Function`, so
/// instances of those PyTypes share this tid via the
/// `register_vtable_for_type` mapping.
pub const FUNCTION_GC_TYPE_ID: u32 = 14;

/// Fixed payload size used by `gct_fv_gc_malloc`'s `c_size`
/// (`framework.py:811`).
pub const FUNCTION_OBJECT_SIZE: usize = std::mem::size_of::<Function>();

/// Byte offsets of the inline `PyObjectRef`-shaped fields the GC must
/// trace during minor collection. `code` is included because
/// `BuiltinCode` is now allocated through `malloc_typed`
/// (`gateway.rs:298`) and therefore lives in the GC heap; the PyCode
/// path remains raw/immortal and the walker's `is_in_nursery` check
/// (`majit-gc/src/collector.rs:764`) leaves those entries alone.
/// `function.py:47 _immutable_fields_ = ['code?', ...]` matches PyPy's
/// Function.code? — an immutable GC reference traced as part of the
/// closure / defs_w / w_kw_defs / w_module set.
///
/// The remaining fields are non-GC: `can_change_code` is a `bool` and
/// `name` is a manually-managed `*const String`.
///
/// `ob.w_class` is intentionally absent, mirroring how W_IntObject /
/// W_FloatObject leave the typeptr-shaped header field out of their
/// `gc_ptr_offsets`. W_TypeObject instances are static-region and
/// not subject to nursery relocation.
pub const FUNCTION_GC_PTR_OFFSETS: [usize; 16] = [
    FUNCTION_CODE_OFFSET,
    FUNCTION_CLOSURE_OFFSET,
    FUNCTION_DEFS_W_OFFSET,
    FUNCTION_W_KW_DEFS_OFFSET,
    FUNCTION_W_MODULE_OFFSET,
    // `function.py:57 w_func_globals` — the module namespace dict object,
    // the function's sole globals carrier; traced for the lifetime of the
    // function so its `__globals__` identity survives minor collection.
    FUNCTION_W_FUNC_GLOBALS_OBJ_OFFSET,
    // CPython 3.14 `func_builtins` — frozen at Function construction.
    FUNCTION_W_BUILTINS_OFFSET,
    // `function.py:50 w_ann` — annotations dict, allocated lazily on
    // first read by the getter or stamped at MAKE_FUNCTION time.
    FUNCTION_W_ANN_OFFSET,
    // PEP 649 `__annotate__` callable stamped by MAKE_FUNCTION's
    // Annotate flag; live until the first `__annotations__` read
    // materialises `w_ann`.
    FUNCTION_W_ANNOTATE_OFFSET,
    // `function.py:68 w_func_dict` — lazily allocated instance dict.
    FUNCTION_W_FUNC_DICT_OFFSET,
    // CPython 3.14 `func_typeparams` — tuple or null for the empty default.
    FUNCTION_W_TYPEPARAMS_OFFSET,
    // `function.py:375 w_doc` — docstring slot cached on first read.
    FUNCTION_W_DOC_OFFSET,
    // `function.py:54 qualname` — qualified name slot stamped at
    // construction or via `f.__qualname__ = ...`.
    FUNCTION_W_QUALNAME_OFFSET,
    // `function.py:498-504 w_objclass` — bound class for descriptor
    // introspection (`inspect.getfullargspec` etc.).
    FUNCTION_W_OBJCLASS_OFFSET,
    // `function.py:487-496 w_text_signature` — PEP 437 text signature
    // line stripped from the docstring.
    FUNCTION_W_TEXT_SIGNATURE_OFFSET,
    // `w_new_self` is intentionally absent: it holds the defining type
    // of a builtin `__new__` carrier, a static-region W_TypeObject that
    // is never nursery-relocated (same reasoning as `ob.w_class`).
    // PyPy `BuiltinFunction.w_moduleobj` is an ordinary GC module reference.
    FUNCTION_W_MODULEOBJ_OFFSET,
];

impl pyre_object::lltype::GcType for Function {
    fn type_id() -> u32 {
        FUNCTION_GC_TYPE_ID
    }
    const SIZE: usize = FUNCTION_OBJECT_SIZE;
}

/// Free the off-GC name string owned by a `Function`.
///
/// # Safety
/// `obj` must point at a valid `Function` whose `name` Box is not aliased by
/// another owner.
pub unsafe fn function_dealloc_name(obj: PyObjectRef) {
    let raw = unsafe { &mut *(obj as *mut Function) };
    if !raw.name.is_null() {
        unsafe { drop(Box::from_raw(raw.name as *mut String)) };
        raw.name = std::ptr::null();
    }
}

/// Allocate a new `Function`.
///
/// `code` is a pointer to a Code object (PyCode) cast to `*const ()`.
/// `name` is the function name string stored in an off-GC Box.
/// `w_func_globals_obj` is the defining module's namespace dict object
/// (shared), or `PY_NULL` for a globals-less carrier.
pub fn function_new(code: *const (), name: String, w_func_globals_obj: PyObjectRef) -> PyObjectRef {
    function_new_with_closure(code, name, w_func_globals_obj, PY_NULL)
}

/// Allocate a new `Function` with a closure.
///
/// `pypy/interpreter/function.py:54-57 Function.__init__` —
/// `self.w_func_globals = w_globals` stores the user-visible
/// `__globals__` dict object directly, so the function's `__globals__`
/// identity IS the supplied `PyObjectRef`.  `closure` is a tuple of cell
/// objects, or PY_NULL if no closure.
pub fn function_new_with_closure(
    code: *const (),
    name: String,
    w_func_globals_obj: PyObjectRef,
    closure: PyObjectRef,
) -> PyObjectRef {
    function_new_impl(
        &FUNCTION_TYPE,
        code,
        name,
        w_func_globals_obj,
        closure,
        true,
    )
}

/// Allocate a `Function` object, GC-managed for user code and immortal for
/// builtin code.
///
/// Reads `FUNCTION_OBJECT_SIZE`/`FUNCTION_GC_TYPE_ID` and calls
/// `lltype::malloc_typed` (`NewWithVtable`) the tracer cannot model; the JIT
/// residualises the call instead of tracing into it
/// (`@dont_look_inside`, `rlib/jit.py:139`), the `box_str_constant` /
/// `try_gc_add_root` twin.
#[majit_macros::dont_look_inside]
pub(crate) fn function_new_impl(
    ob_type: &'static PyType,
    code: *const (),
    name: String,
    w_func_globals_obj: PyObjectRef,
    closure: PyObjectRef,
    can_change_code: bool,
) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`) for
    // the `lltype::malloc_typed` call below. `closure`, `code`, and
    // `w_func_globals_obj` are PyObjectRef-shaped GC roots across the
    // alloc — `BuiltinCode` lives in the GC heap (`gateway.rs:298
    // malloc_typed`) and `PyCode` is currently raw/immortal; the
    // walker's `is_in_nursery` filter (`majit-gc/src/collector.rs:764`)
    // is what makes the heterogeneous case safe. `name_ptr` is allocated
    // below via `malloc_raw` (non-GC) and stored into the struct as
    // part of the same `malloc_typed` call, so it never spans a
    // collection point.
    let _roots = pyre_object::gc_roots::push_roots();
    let closure_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(closure);
    let code_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(code as PyObjectRef);
    // `function.py:57 self.w_func_globals = w_globals` stores the dict
    // object directly as the function's sole globals carrier.
    let globals_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(w_func_globals_obj);

    // CPython 3.14 `_PyEval_BuiltinsFromGlobals` at function construction:
    // retain the selected mapping identity even if the globals entry changes
    // later.  Builtin/gateway carriers have no globals and keep a null slot.
    let w_builtins = if w_func_globals_obj.is_null() {
        PY_NULL
    } else {
        let selected = crate::baseobjspace::pick_builtin_obj(
            w_func_globals_obj,
            crate::call::take_last_exec_ctx(),
        );
        if !selected.is_null() && unsafe { pyre_object::is_module(selected) } {
            unsafe { pyre_object::w_module_get_w_dict(selected) }
        } else {
            selected
        }
    };
    let builtins_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(w_builtins);

    // `pick_builtin_obj` and later allocations may collect.  Reload every
    // pinned input before embedding it in the new Function; the original raw
    // locals are not rewritten when the shadow-stack slots are forwarded.
    let closure = pyre_object::gc_roots::shadow_stack_get(closure_slot);
    let code = pyre_object::gc_roots::shadow_stack_get(code_slot) as *const ();
    let w_func_globals_obj = pyre_object::gc_roots::shadow_stack_get(globals_slot);
    let w_builtins = pyre_object::gc_roots::shadow_stack_get(builtins_slot);

    let name_ptr = pyre_object::lltype::malloc_raw(name) as *const String;
    let function = Function {
        ob: PyObject {
            ob_type: ob_type as *const PyType,
            w_class: pyre_object::pyobject::get_instantiate(ob_type),
        },
        code,
        can_change_code,
        name: name_ptr,
        closure,
        defs_w: PY_NULL,
        w_kw_defs: PY_NULL,
        w_module: PY_NULL,
        w_func_globals_obj,
        w_builtins,
        w_ann: PY_NULL,
        w_annotate: PY_NULL,
        w_func_dict: PY_NULL,
        w_typeparams: PY_NULL,
        w_doc: PY_NULL,
        w_qualname: PY_NULL,
        w_objclass: PY_NULL,
        w_text_signature: PY_NULL,
        w_new_self: PY_NULL,
        w_moduleobj: PY_NULL,
    };

    // A `BuiltinCode`-backed function is a permanent type / module slot (the
    // interp2app analogue of a translation-time prebuilt object): its code is
    // immortal (`gateway.rs builtin_code_new_full` malloc_typed) and its only
    // other fields are null for a freshly-made builtin. Allocate it immortal so
    // a full mark-sweep can never reclaim it out of an off-GC builtin type dict
    // — the collector assumes no immortal object holds heap pointers and so does
    // not trace such dicts (collector.rs:1803), which would otherwise free the
    // method functions of a builtin type built lazily at runtime (weakref, …)
    // after the GC hook is wired. Startup builtin functions are already immortal
    // (no hook installed yet); this extends that to runtime-created ones. User
    // functions (`PyCode`) stay GC-managed.
    let is_builtin =
        !code.is_null() && unsafe { crate::gateway::is_builtin_code(code as PyObjectRef) };
    if !is_builtin {
        let raw = pyre_object::gc_hook::try_gc_alloc_stable_raw(
            FUNCTION_GC_TYPE_ID,
            FUNCTION_OBJECT_SIZE,
        );
        if !raw.is_null() {
            unsafe {
                std::ptr::write(raw as *mut Function, function);
            }
            return raw as PyObjectRef;
        }
    }

    pyre_object::lltype::malloc_typed(function) as PyObjectRef
}

/// function.py:703 — `class FunctionWithFixedCode(Function): can_change_code = False`
/// Allocate a function whose code pointer the JIT can treat as immutable.
pub fn function_new_with_fixed_code(
    code: *const (),
    name: String,
    w_func_globals_obj: PyObjectRef,
) -> PyObjectRef {
    function_new_impl(
        &FUNCTION_TYPE,
        code,
        name,
        w_func_globals_obj,
        PY_NULL,
        false,
    )
}

/// function.py:706 — `class BuiltinFunction(Function): can_change_code = False`
/// Allocate a module builtin whose typedef intentionally omits `__get__`.
pub fn function_new_builtin(
    code: *const (),
    name: String,
    w_func_globals_obj: PyObjectRef,
) -> PyObjectRef {
    function_new_impl(
        &BUILTIN_FUNCTION_TYPE,
        code,
        name,
        w_func_globals_obj,
        PY_NULL,
        false,
    )
}

/// function.py:385-388 — `_check_code_mutable(attr)`:
///
/// ```python
/// def _check_code_mutable(self, attr):
///     if not self.can_change_code:
///         raise oefmt(self.space.w_AttributeError,
///                     "Cannot change %s attribute of builtin functions", attr)
/// ```
pub unsafe fn _check_code_mutable(func: PyObjectRef, attr: &str) -> Result<(), crate::PyError> {
    unsafe {
        if (*(func as *const Function)).can_change_code {
            Ok(())
        } else {
            Err(crate::PyError::attribute_error(format!(
                "Cannot change {} attribute of builtin functions",
                attr
            )))
        }
    }
}

/// function.py:23 — `@jit.elidable_promote()`
/// Only valid when `can_change_code == false`.
#[majit_macros::elidable_promote]
#[inline]
pub unsafe fn _get_immutable_code(func: PyObjectRef) -> *const () {
    // function.py:25
    debug_assert!(
        !unsafe { (*(func as *const Function)).can_change_code },
        "_get_immutable_code called on function with can_change_code=true"
    );
    unsafe { (*(func as *const Function)).code }
}

/// Check if an object is a user-defined function.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_function(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &FUNCTION_TYPE) || py_type_check(obj, &BUILTIN_FUNCTION_TYPE) }
}

/// Stamp a module-level builtin function's `__module__` with its
/// containing module at install time — `MixedModule` registration sets
/// `func.w_module = w(modulename)` for each interp-level definition.
/// Touches `BUILTIN_FUNCTION_TYPE` objects (the `py_module!`
/// `inline_functions` / `functions` / `module_functions` carriers) and any
/// globals-less `FUNCTION_TYPE` object whose module is still unset.
/// App-level functions carry globals and derive `__module__` lazily from
/// `globals['__name__']`, so a function with globals is left alone.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
pub unsafe fn builtin_function_set_module(obj: PyObjectRef, w_module: PyObjectRef) {
    unsafe {
        let stampable = py_type_check(obj, &BUILTIN_FUNCTION_TYPE)
            || (py_type_check(obj, &FUNCTION_TYPE)
                && (*(obj as *const Function)).w_func_globals_obj.is_null());
        if stampable {
            let func = obj as *mut Function;
            if (*func).w_module.is_null() {
                function_write_barrier(obj);
                (*func).w_module = w_module;
            }
        }
    }
}

/// CPython 3.14 `PyCFunctionObject.m_module` member assignment. Unlike
/// PyPy's shared Function getset, builtin `__module__` is a writable direct
/// member and accepts any object; deletion stores `None`.
pub unsafe fn builtin_function_set_module_attr(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        function_write_barrier(obj);
        (*(obj as *mut Function)).w_module = value;
    }
}

/// Stamp the `__self__` of a builtin `__new__` carrier — the defining
/// type whose `tp_new` it wraps (`typeobject.c add_tp_new_wrapper`).
/// Only touches functions whose `w_new_self` is still unset, so an
/// inherited `__new__` keeps the ancestor that defined it.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `Function`.
pub unsafe fn function_set_new_self(obj: PyObjectRef, w_type: PyObjectRef) {
    unsafe {
        let func = obj as *mut Function;
        if (*func).w_new_self.is_null() {
            function_write_barrier(obj);
            (*func).w_new_self = w_type;
        }
    }
}

/// `__self__` getter for the builtin-function type.  Returns the
/// stamped `w_new_self` (the defining type) for a builtin `__new__`
/// carrier, else `None` — `typedef.py:816 GetSetProperty(always_none,
/// cls=BuiltinFunction)` returns `None` for an ordinary builtin.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `Function`.
pub unsafe fn function_get_self_or_none(obj: PyObjectRef) -> PyObjectRef {
    let func = unsafe { &*(obj as *const Function) };
    let w_self = if !func.w_moduleobj.is_null() {
        func.w_moduleobj
    } else {
        func.w_new_self
    };
    if w_self.is_null() {
        pyre_object::w_none()
    } else {
        w_self
    }
}

/// CPython 3.14 `meth_reduce`: type-bound builtins reconstruct through
/// `getattr(__self__, __name__)`; module-level builtins reduce by qualname.
pub unsafe fn descr_builtin_function_reduce(obj: PyObjectRef) -> crate::PyResult {
    let w_self = unsafe { function_get_self_or_none(obj) };
    if !w_self.is_null()
        && !unsafe { pyre_object::is_none(w_self) }
        && !unsafe { pyre_object::is_module(w_self) }
    {
        let name = pyre_object::w_str_new(unsafe { crate::function_get_name(obj) });
        return Ok(pyre_object::w_tuple_new(vec![
            crate::baseobjspace::builtin_callable("getattr"),
            pyre_object::w_tuple_new(vec![w_self, name]),
        ]));
    }
    Ok(pyre_object::w_str_new(&unsafe {
        function_get_qualname(obj)
    }))
}

/// Stamp PyPy `BuiltinFunction.w_moduleobj` after the defining module object
/// has been created. The module name is installed first through
/// `builtin_function_set_module`; this second field preserves the distinct
/// `__module__` (string) / `__self__` (module object) identities.
///
/// # Safety
/// `obj` must be a valid PyObjectRef. Non-builtin functions are ignored.
pub unsafe fn builtin_function_set_module_obj(obj: PyObjectRef, w_module: PyObjectRef) {
    unsafe {
        if py_type_check(obj, &BUILTIN_FUNCTION_TYPE) {
            let func = obj as *mut Function;
            function_write_barrier(obj);
            (*func).w_moduleobj = w_module;
        }
    }
}

/// `isinstance(obj, FunctionWithFixedCode)` parity.
///
/// function.py:783 — `class FunctionWithFixedCode(Function):
///     can_change_code = False`.  Pyre encodes the `Function`
/// vs `FunctionWithFixedCode` distinction through the
/// `can_change_code` flag (true for user `def`s built via
/// `function_new_with_closure`, false for gateway-built builtins
/// via `function_new_with_fixed_code`).  `BuiltinFunction` has its
/// own `BUILTIN_FUNCTION_TYPE`, so a strict `FUNCTION_TYPE` check
/// excludes it (function.py:786 makes BuiltinFunction a sibling
/// subclass of Function, not a subclass of FunctionWithFixedCode).
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_function_with_fixed_code(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &FUNCTION_TYPE) && !(*(obj as *const Function)).can_change_code }
}

/// function.py:78-83 — `getcode(self)`: three-way dispatch.
///   - JIT + immutable code → _get_immutable_code (elidable_promote)
///   - JIT + mutable code  → promote(self.code)
///   - interpreter         → self.code
#[inline]
pub unsafe fn getcode(obj: PyObjectRef) -> *const () {
    unsafe {
        let func = obj as *const Function;
        if majit_metainterp::jit::we_are_jitted() {
            if !(*func).can_change_code {
                // function.py:80-81
                return _get_immutable_code(obj);
            }
            // function.py:82
            return majit_metainterp::jit::promote((*func).code as usize) as *const ();
        }
        // function.py:83
        (*func).code
    }
}

/// Get the Code object pointer from a function object.
///
/// Returns a pointer to the Code-level object (PyCode or BuiltinCode).
///
/// # Safety
/// `obj` must point to a valid `Function`.
/// NOTE: NOT elidable — code field can change (can_change_code).
/// Use _get_immutable_code() for the elidable path.
#[inline]
pub unsafe fn function_get_code(obj: PyObjectRef) -> *const () {
    unsafe { (*(obj as *const Function)).code }
}

/// Extract the raw bytecode CodeObject pointer from a user function.
///
/// Equivalent to accessing `self.getcode().code_ptr` in PyPy terms:
/// `getcode()` returns the Code wrapper (PyCode), and this
/// dereferences through it to the underlying CodeObject.
///
/// # Safety
/// `obj` must point to a valid `Function` whose `code` field is a `PyCode`
/// (i.e., NOT a BuiltinCode). Only call on user-defined functions.
#[inline]
pub unsafe fn get_pycode(obj: PyObjectRef) -> *const () {
    unsafe {
        let code = getcode(obj);
        debug_assert!(
            !crate::is_builtin_code(code as PyObjectRef),
            "get_pycode called on a builtin function"
        );
        crate::w_code_get_ptr(code as PyObjectRef)
    }
}

/// Get the function name.
///
/// # Safety
/// `obj` must point to a valid `Function`.
#[inline]
pub unsafe fn function_get_name(obj: PyObjectRef) -> &'static str {
    unsafe { &*(*(obj as *const Function)).name }
}

/// `function.py:476-485 fset_func_qualname` parity:
///
/// ```python
/// def fset_func_qualname(self, space, w_name):
///     self._check_code_mutable("__qualname__")
///     try:
///         qualname = space.realutf8_w(w_name)
///     except OperationError as e:
///         if e.match(space, space.w_TypeError):
///             raise oefmt(space.w_TypeError,
///                         "__qualname__ must be set to a string object")
///         raise
///     self.set_qualname(qualname)
/// ```
///
/// `space.realutf8_w` accepts any `isinstance_w(w_name, w_text)` —
/// `str` and its subclasses.  `isinstance_str_w` mirrors that.
///
/// # Safety
/// `obj` must point to a valid `Function`.
pub unsafe fn fset_func_qualname(
    obj: PyObjectRef,
    value: PyObjectRef,
) -> Result<(), crate::PyError> {
    unsafe {
        _check_code_mutable(obj, "__qualname__")?; // function.py:477
        if !crate::baseobjspace::isinstance_str_w(value) {
            return Err(crate::PyError::type_error(
                "__qualname__ must be set to a string object",
            ));
        }
        function_write_barrier(obj);
        (*(obj as *mut Function)).w_qualname = value;
        Ok(())
    }
}

/// `function.py:470-471 fget_func_qualname` parity:
///
/// ```python
/// def fget_func_qualname(self, space):
///     return space.newtext(self.qualname)
/// ```
///
/// `self.qualname` is initialised from `qualname or self.name` at
/// `function.py:54 __init__`.  MAKE_FUNCTION
/// (`runtime_ops::make_function_from_code_obj_with_globals_obj`) stamps `w_qualname`
/// from `codeobj.co_qualname` at construction, so subsequent
/// `__code__ = new_code` assignments do NOT alter `__qualname__`
/// (matching `pypy/interpreter/pyopcode.py:1457` + `function.py:54`).
/// When `w_qualname` is still `PY_NULL` (legacy callers that never
/// stamped it — builtins built via `gateway.rs` / unit tests), fall
/// back to the bare `name` string per `qualname or self.name`.
///
/// # Safety
/// `obj` must point to a valid `Function`.
pub unsafe fn function_get_qualname(obj: PyObjectRef) -> String {
    unsafe {
        let cached = (*(obj as *const Function)).w_qualname;
        if !cached.is_null() && pyre_object::is_str(cached) {
            return pyre_object::w_str_get_value(cached).to_string();
        }
        function_get_name(obj).to_string()
    }
}

/// `function.py:54 self.qualname = qualname or self.name` setter —
/// used by MAKE_FUNCTION (`runtime_ops::make_function_from_code_obj_with_globals_obj`)
/// to freeze the qualified name immediately after `function_new`.
/// Bypasses `_check_code_mutable` because this is the construction
/// path; user code goes through `fset_func_qualname` instead.
///
/// # Safety
/// `obj` must point to a valid `Function`; `value` must be a string
/// `PyObjectRef`.
#[inline]
pub unsafe fn function_set_qualname(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        function_write_barrier(obj);
        (*(obj as *mut Function)).w_qualname = value;
    }
}

#[inline]
pub unsafe fn _eq(_obj: PyObjectRef, other: PyObjectRef) -> bool {
    _obj == other
}

/// PyPy-compatible descriptor accessor for function name.
#[inline]
pub unsafe fn fget_func_name(obj: PyObjectRef) -> PyObjectRef {
    unsafe { pyre_object::w_str_new(function_get_name(obj)) }
}

/// Return the canonical W_DictObject stored as `function.w_func_globals`.
///
/// `function.py:57 self.w_func_globals = w_globals` stores the dict
/// object directly; this is a plain field load.  Returns `PY_NULL` for
/// a globals-less carrier (gateway builtins).
///
/// # Safety
/// `obj` must point to a valid `Function`.
#[inline]
pub unsafe fn function_get_globals_obj(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const Function)).w_func_globals_obj }
}

/// CPython 3.14 `PyFunctionObject.func_builtins`, resolved once during
/// construction. Returns `PY_NULL` for globals-less builtin carriers.
#[inline]
pub unsafe fn function_get_builtins(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const Function)).w_builtins }
}

/// Get the closure tuple from a function object.
/// Returns PY_NULL if the function has no closure.
///
/// # Safety
/// `obj` must point to a valid `Function`.
#[inline]
pub unsafe fn function_get_closure(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const Function)).closure }
}

/// Set the closure on a function object.
///
/// # Safety
/// `obj` must point to a valid `Function`.
#[inline]
pub unsafe fn function_set_closure(obj: PyObjectRef, closure: PyObjectRef) {
    unsafe {
        function_write_barrier(obj);
        (*(obj as *mut Function)).closure = closure;
    }
}

/// Get defaults tuple.
#[inline]
pub unsafe fn function_get_defaults(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const Function)).defs_w }
}

/// Set defaults tuple.
#[inline]
pub unsafe fn function_set_defaults(obj: PyObjectRef, defaults: PyObjectRef) {
    unsafe {
        function_write_barrier(obj);
        (*(obj as *mut Function)).defs_w = defaults;
    }
}

/// Get kwdefaults dict.
#[inline]
pub unsafe fn function_get_kwdefaults(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const Function)).w_kw_defs }
}

/// Set kwdefaults dict.
#[inline]
pub unsafe fn function_set_kwdefaults(obj: PyObjectRef, kwdefaults: PyObjectRef) {
    unsafe {
        function_write_barrier(obj);
        (*(obj as *mut Function)).w_kw_defs = kwdefaults;
    }
}

/// PyPy-compatible `__dict__` storage field alias.
#[inline]
pub unsafe fn function_getdict(obj: PyObjectRef) -> PyObjectRef {
    unsafe {
        let func = obj as *mut Function;
        if (*func).w_func_dict.is_null() {
            function_write_barrier(obj);
            (*func).w_func_dict = pyre_object::w_dict_new();
        }
        (*func).w_func_dict
    }
}

/// `function.py:238 Function.setdict` — replace the function's instance
/// dict, raising `TypeError` when `value` is not a dict.  Routes through
/// `setdict` (the wholesale dict replacement), not `setattr_str(obj,
/// "__dict__", ..)` which would store a literal `"__dict__"` dict entry.
#[inline]
pub unsafe fn function_setdict(obj: PyObjectRef, value: PyObjectRef) -> Result<(), crate::PyError> {
    let w_dict_type = crate::typedef::gettypeobject(&pyre_object::pyobject::DICT_TYPE);
    if !unsafe { crate::baseobjspace::isinstance_w(value, w_dict_type) } {
        return Err(crate::PyError::type_error(
            "setting function's dictionary to a non-dict",
        ));
    }
    unsafe {
        function_write_barrier(obj);
        (*(obj as *mut Function)).w_func_dict = value;
    }
    Ok(())
}

/// CPython 3.14 `function.__annotate__` getter.
pub unsafe fn function_get_annotate(obj: PyObjectRef) -> PyObjectRef {
    let value = unsafe { (*(obj as *const Function)).w_annotate };
    if value.is_null() {
        pyre_object::w_none()
    } else {
        value
    }
}

/// CPython 3.14 `function.__annotate__` setter.
pub unsafe fn function_set_annotate(
    obj: PyObjectRef,
    value: PyObjectRef,
) -> Result<(), crate::PyError> {
    if value.is_null() {
        return Err(crate::PyError::type_error("__annotate__ cannot be deleted"));
    }
    if !pyre_object::is_none(value) && !crate::baseobjspace::callable_w(value) {
        return Err(crate::PyError::type_error(
            "__annotate__ must be callable or None",
        ));
    }
    unsafe {
        function_write_barrier(obj);
        let func = obj as *mut Function;
        (*func).w_annotate = value;
        if !pyre_object::is_none(value) {
            (*func).w_ann = PY_NULL;
        }
    }
    Ok(())
}

/// CPython 3.14 `function.__type_params__` getter.
pub unsafe fn function_get_typeparams(obj: PyObjectRef) -> PyObjectRef {
    let value = unsafe { (*(obj as *const Function)).w_typeparams };
    if value.is_null() {
        pyre_object::w_tuple_new(vec![])
    } else {
        value
    }
}

/// CPython 3.14 `function.__type_params__` setter and
/// `_Py_set_function_type_params` opcode helper.
pub unsafe fn function_set_typeparams(
    obj: PyObjectRef,
    value: PyObjectRef,
) -> Result<(), crate::PyError> {
    if value.is_null() || !pyre_object::is_tuple(value) {
        return Err(crate::PyError::type_error(
            "__type_params__ must be set to a tuple",
        ));
    }
    unsafe {
        function_write_barrier(obj);
        (*(obj as *mut Function)).w_typeparams = value;
    }
    Ok(())
}

/// PyPy-compatible `getdict()` descriptor helper.
#[inline]
pub unsafe fn getdict(obj: PyObjectRef) -> PyObjectRef {
    unsafe { function_getdict(obj) }
}

/// PyPy-compatible `setdict()` helper.
#[inline]
pub unsafe fn setdict(obj: PyObjectRef, value: PyObjectRef) -> Result<(), crate::PyError> {
    unsafe { function_setdict(obj, value) }
}

/// `function.py:446-449 fget_func_doc` parity:
///
/// ```python
/// def fget_func_doc(self, space):
///     if self.w_doc is None:
///         self.w_doc = self.code.getdocstring(space)
///     return self.w_doc
/// ```
///
/// `(*func).w_doc == PY_NULL` mirrors PyPy's `self.w_doc is None`
/// (the unset / not-yet-cached state).  First reader resolves
/// `code.getdocstring(space)` and stamps the result back so subsequent
/// `f.__doc__ is f.__doc__` holds; `function_del_doc` writes
/// `w_none()` to make the deleted state sticky against the lazy
/// fallback (`function.py:455-457 fdel_func_doc`).
///
/// `code.getdocstring(space)` has two shapes in pyre:
///   - `BuiltinCode`: stores `docstring` directly (gateway.rs:581).
///   - `PyCode`: docstring is the first const when `code.flags`
///     has `HAS_DOCSTRING` set, mirroring `pycode.py:230
///     PyCode.getdocstring`.
pub fn function_get_doc(obj: PyObjectRef) -> PyObjectRef {
    if obj.is_null() {
        return pyre_object::w_none();
    }
    let func = obj as *mut Function;
    let cached = unsafe { (*func).w_doc };
    if !cached.is_null() {
        return cached;
    }
    // Lazy fallback: `code.getdocstring(space)` (function.py:448).
    let resolved = code_getdocstring(obj);
    unsafe {
        function_write_barrier(obj);
        (*func).w_doc = resolved;
    }
    resolved
}

/// `pycode.py:230 PyCode.getdocstring(space)` parity for the two code
/// shapes pyre carries — extracted out of `function_get_doc` so
/// `fset_func_code`'s pre-assignment cache step
/// (`function.py:538 self.fget_func_doc(space)`) can reach the same
/// path without going through the cache write.
fn code_getdocstring(obj: PyObjectRef) -> PyObjectRef {
    let code = unsafe { function_get_code(obj) } as PyObjectRef;
    if code.is_null() {
        return pyre_object::w_none();
    }
    if unsafe { crate::gateway::is_builtin_code(code) } {
        return unsafe { crate::gateway::builtin_code_get_docstring(code) };
    }
    if unsafe { crate::pycode::is_code(code) } {
        let raw = unsafe { crate::pycode::w_code_get_ptr(code) } as *const crate::CodeObject;
        if !raw.is_null() {
            let code_ref = unsafe { &*raw };
            if code_ref.flags.contains(crate::CodeFlags::HAS_DOCSTRING)
                && !code_ref.constants.is_empty()
            {
                let first = crate::pyframe::load_const_from_code(code_ref, 0);
                if !first.is_null() && unsafe { pyre_object::is_str(first) } {
                    return first;
                }
            }
        }
    }
    pyre_object::w_none()
}

/// `function.py:451-453 fset_func_doc` parity:
///
/// ```python
/// def fset_func_doc(self, space, w_doc):
///     self._check_code_mutable("__doc__")
///     self.w_doc = w_doc
/// ```
pub unsafe fn function_set_doc(obj: PyObjectRef, value: PyObjectRef) -> Result<(), crate::PyError> {
    unsafe {
        _check_code_mutable(obj, "__doc__")?; // function.py:452
        if obj.is_null() {
            return Ok(());
        }
        function_write_barrier(obj);
        (*(obj as *mut Function)).w_doc = value;
        Ok(())
    }
}

/// `function.py:455-457 fdel_func_doc` parity:
///
/// ```python
/// def fdel_func_doc(self, space):
///     self._check_code_mutable("__doc__")
///     self.w_doc = space.w_None
/// ```
///
/// Stamps `w_none()` so the next `function_get_doc` short-circuits
/// the lazy `code.getdocstring(space)` fallback — `w_doc` is no
/// longer null, so the cache hit returns `w_none()` directly.
pub unsafe fn function_del_doc(obj: PyObjectRef) -> Result<(), crate::PyError> {
    unsafe {
        _check_code_mutable(obj, "__doc__")?; // function.py:456
        if obj.is_null() {
            return Ok(());
        }
        function_write_barrier(obj);
        (*(obj as *mut Function)).w_doc = pyre_object::w_none();
        Ok(())
    }
}

/// `function.py:548-551 fget_func_annotations` parity — returns the
/// stored annotations dict, lazily allocating an empty dict on the
/// first read when none was set.  PyPy mutates `self.w_ann = space
/// .newdict()` in place so subsequent reads return the same dict
/// (`f.__annotations__ is f.__annotations__`); pyre stamps the slot
/// the same way through `function_set_annotations`.
///
/// PEP 649 lazy annotations (no PyPy counterpart — upstream targets
/// 3.11): when `w_ann` is unset but a `w_annotate` callable was
/// stamped at MAKE_FUNCTION time, evaluate it with `format=1` and
/// stamp the resulting dict, mirroring CPython 3.14
/// `func_get_annotations`.
///
/// # Safety
/// `obj` must point to a valid `Function`.
#[inline]
pub unsafe fn function_get_annotations(obj: PyObjectRef) -> PyObjectRef {
    unsafe {
        if obj.is_null() {
            return pyre_object::w_dict_new();
        }
        let func = obj as *mut Function;
        let cached = (*func).w_ann;
        if !cached.is_null() {
            return cached;
        }
        let annotate_fn = (*func).w_annotate;
        if !annotate_fn.is_null() && !pyre_object::is_none(annotate_fn) {
            let dict = crate::call_function(annotate_fn, &[pyre_object::w_int_new(1)]);
            if !dict.is_null() {
                function_write_barrier(obj);
                (*func).w_ann = dict;
                return dict;
            }
        }
        let fresh = pyre_object::w_dict_new();
        function_write_barrier(obj);
        (*func).w_ann = fresh;
        fresh
    }
}

/// MAKE_FUNCTION ANNOTATIONS opcode helper — stamps the
/// compile-time annotations dict directly into `Function.w_ann`
/// without running `function.py:553-559 fset_func_annotations`'s
/// type validation.  The compiler always emits a real dict here, so
/// the `isinstance(w_new, dict)` check would never fail; bypassing
/// it keeps the opcode hot path free of error-handling overhead.
///
/// User-level `f.__annotations__ = X` writes go through
/// `fset_func_annotations` (typedef getset descriptor) instead.
///
/// # Safety
/// `obj` must point to a valid `Function`.
#[inline]
pub unsafe fn function_set_annotations(obj: PyObjectRef, w_ann: PyObjectRef) {
    unsafe {
        if obj.is_null() {
            return;
        }
        function_write_barrier(obj);
        let func = obj as *mut Function;
        (*func).w_ann = w_ann;
        (*func).w_annotate = PY_NULL;
    }
}

/// `function.py:553-559 fset_func_annotations` parity:
///
/// ```python
/// def fset_func_annotations(self, space, w_new):
///     self._check_code_mutable("__annotations__")
///     if space.is_w(w_new, space.w_None):
///         w_new = None
///     elif not space.isinstance_w(w_new, space.w_dict):
///         raise oefmt(space.w_TypeError, "__annotations__ must be a dict")
///     self.w_ann = w_new
/// ```
///
/// # Safety
/// `obj` must point to a valid `Function`.
pub unsafe fn fset_func_annotations(
    obj: PyObjectRef,
    value: PyObjectRef,
) -> Result<(), crate::PyError> {
    unsafe {
        _check_code_mutable(obj, "__annotations__")?; // function.py:554
        if obj.is_null() {
            return Ok(());
        }
        // function.py:555-558 — None clears the slot, anything else
        // must be a dict.
        let stored = if value.is_null() || pyre_object::is_none(value) {
            PY_NULL
        } else if pyre_object::is_dict(value) {
            value
        } else {
            return Err(crate::PyError::type_error("__annotations__ must be a dict"));
        };
        function_write_barrier(obj);
        let func = obj as *mut Function;
        (*func).w_ann = stored;
        // CPython 3.14 function___annotations___set_impl clears the lazy
        // annotation callable whenever eager annotations are assigned.
        (*func).w_annotate = PY_NULL;
        Ok(())
    }
}

/// `function.py:561-563 fdel_func_annotations` parity:
///
/// ```python
/// def fdel_func_annotations(self, space):
///     self._check_code_mutable("__annotations__")
///     self.w_ann = None
/// ```
///
/// # Safety
/// `obj` must point to a valid `Function`.
pub unsafe fn fdel_func_annotations(obj: PyObjectRef) -> Result<(), crate::PyError> {
    unsafe {
        _check_code_mutable(obj, "__annotations__")?; // function.py:562
        if obj.is_null() {
            return Ok(());
        }
        function_write_barrier(obj);
        let func = obj as *mut Function;
        (*func).w_ann = PY_NULL;
        (*func).w_annotate = PY_NULL;
        Ok(())
    }
}

/// PyPy `fget_func_defaults` accessor.
#[inline]
pub unsafe fn fget_func_defaults(obj: PyObjectRef) -> PyObjectRef {
    unsafe {
        let value = function_get_defaults(obj);
        if value.is_null() {
            pyre_object::w_none()
        } else {
            value
        }
    }
}

/// `function.py:408-416 fset_func_defaults` parity:
///
/// ```python
/// def fset_func_defaults(self, space, w_defaults):
///     self._check_code_mutable("__defaults__")
///     if space.is_w(w_defaults, space.w_None):
///         self.defs_w = []
///         return
///     if not space.isinstance_w(w_defaults, space.w_tuple):
///         raise oefmt(space.w_TypeError,
///                     "__defaults__ must be set to a tuple object or None")
///     self.defs_w = space.fixedview(w_defaults)
/// ```
#[inline]
pub unsafe fn fset_func_defaults(
    obj: PyObjectRef,
    value: PyObjectRef,
) -> Result<(), crate::PyError> {
    unsafe {
        _check_code_mutable(obj, "__defaults__")?; // function.py:409
        if value.is_null() || pyre_object::is_none(value) {
            function_set_defaults(obj, pyre_object::PY_NULL);
            return Ok(());
        }
        if !pyre_object::is_tuple(value) {
            return Err(crate::PyError::type_error(
                "__defaults__ must be set to a tuple object or None",
            ));
        }
        function_set_defaults(obj, value);
        Ok(())
    }
}

/// `function.py:418-420 fdel_func_defaults` parity:
///
/// ```python
/// def fdel_func_defaults(self, space):
///     self._check_code_mutable("__defaults__")
///     self.defs_w = []
/// ```
#[inline]
pub unsafe fn fdel_func_defaults(obj: PyObjectRef) -> Result<(), crate::PyError> {
    unsafe {
        _check_code_mutable(obj, "__defaults__")?; // function.py:419
        function_set_defaults(obj, pyre_object::PY_NULL);
        Ok(())
    }
}

/// PyPy `fget_func_kwdefaults` accessor.
#[inline]
pub unsafe fn fget_func_kwdefaults(obj: PyObjectRef) -> PyObjectRef {
    unsafe {
        let value = function_get_kwdefaults(obj);
        if value.is_null() {
            pyre_object::w_none()
        } else {
            value
        }
    }
}

/// `function.py:427-433 fset_func_kwdefaults` parity:
///
/// ```python
/// def fset_func_kwdefaults(self, space, w_new):
///     if space.is_w(w_new, space.w_None):
///         self.w_kw_defs = None
///     else:
///         if not space.isinstance_w(w_new, space.w_dict):
///             raise oefmt(space.w_TypeError, "__kwdefaults__ must be a dict")
///         self.w_kw_defs = w_new
/// ```
///
/// PyPy intentionally omits `_check_code_mutable` here — `__kwdefaults__`
/// is settable on builtins too.
#[inline]
pub unsafe fn fset_func_kwdefaults(
    obj: PyObjectRef,
    value: PyObjectRef,
) -> Result<(), crate::PyError> {
    unsafe {
        if value.is_null() || pyre_object::is_none(value) {
            function_set_kwdefaults(obj, pyre_object::PY_NULL);
            return Ok(());
        }
        if !pyre_object::is_dict(value) {
            return Err(crate::PyError::type_error("__kwdefaults__ must be a dict"));
        }
        function_set_kwdefaults(obj, value);
        Ok(())
    }
}

/// `function.py:435-436 fdel_func_kwdefaults` parity:
///
/// ```python
/// def fdel_func_kwdefaults(self, space):
///     self.w_kw_defs = None
/// ```
///
/// PyPy intentionally omits `_check_code_mutable` here — symmetric
/// with `fset_func_kwdefaults` at function.py:427-433.
#[inline]
pub unsafe fn fdel_func_kwdefaults(obj: PyObjectRef) -> Result<(), crate::PyError> {
    unsafe {
        function_set_kwdefaults(obj, pyre_object::PY_NULL);
        Ok(())
    }
}

/// function.py:435-436 — `fget_func_code(self, space): return self.getcode()`
/// Uses getcode() for JIT elidable_promote / promote path.
#[inline]
pub unsafe fn function_get_func_code(obj: PyObjectRef) -> *const () {
    unsafe { getcode(obj) }
}

/// PyPy-compatible `__code__` setter.
#[inline]
pub unsafe fn function_set_func_code(obj: PyObjectRef, code: *const ()) {
    unsafe {
        function_write_barrier(obj);
        (*(obj as *mut Function)).code = code;
    }
}

/// PyPy-compatible `__name__` getter alias.
#[inline]
pub unsafe fn function_get_func_name(obj: PyObjectRef) -> &'static str {
    unsafe { function_get_name(obj) }
}

/// PyPy-compatible `__name__` setter.
#[inline]
pub unsafe fn function_set_func_name(obj: PyObjectRef, name: PyObjectRef) {
    unsafe {
        if !pyre_object::is_str(name) {
            return;
        }
        let name = pyre_object::w_str_get_value(name);
        let name = pyre_object::lltype::malloc_raw(name.to_string()) as *const String;
        let old = (*(obj as *mut Function)).name;
        if !old.is_null() {
            drop(Box::from_raw(old as *mut String));
        }
        (*(obj as *mut Function)).name = name;
    }
}

/// `function.py:498-501 fget_func_objclass` parity:
///
/// ```python
/// def fget_func_objclass(self, space):
///     if self.w_objclass is None:
///         raise oefmt(space.w_AttributeError, "__objclass__")
///     return self.w_objclass
/// ```
///
/// # Safety
/// `obj` must point to a valid `Function`.
#[inline]
pub unsafe fn fget_func_objclass(obj: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    unsafe {
        let value = (*(obj as *const Function)).w_objclass;
        if value.is_null() {
            return Err(crate::PyError::attribute_error("__objclass__"));
        }
        Ok(value)
    }
}

/// `function.py:503-504 set_objclass(w_type)` parity — direct field
/// write used by descriptor-bind helpers.
///
/// # Safety
/// `obj` must point to a valid `Function`.
#[inline]
pub unsafe fn function_set_objclass(obj: PyObjectRef, w_type: PyObjectRef) {
    unsafe {
        function_write_barrier(obj);
        (*(obj as *mut Function)).w_objclass = w_type;
    }
}

/// `function.py:487-490 fget_func_text_signature` parity:
///
/// ```python
/// def fget_func_text_signature(self, space):
///     if self.w_text_signature is None:
///         raise oefmt(space.w_AttributeError, "__text_signature__")
///     return self.w_text_signature
/// ```
///
/// PyPy distinguishes the RPython-level `None` (unset, raises
/// `AttributeError`) from `space.w_None` (explicitly stored). Pyre
/// uses `PY_NULL` for the former; `space.w_None` survives as a
/// real PyObjectRef.
///
/// # Safety
/// `obj` must point to a valid `Function`.
#[inline]
pub unsafe fn fget_func_text_signature(obj: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    unsafe {
        let value = (*(obj as *const Function)).w_text_signature;
        if value.is_null() {
            return Err(crate::PyError::attribute_error("__text_signature__"));
        }
        Ok(value)
    }
}

/// `function.py:492-493 fset_func_text_signature` parity:
///
/// ```python
/// def fset_func_text_signature(self, space, w_value):
///     self.w_text_signature = w_value
/// ```
///
/// Direct field write — even `space.w_None` is preserved as a real
/// value (only the RPython-level `None`, i.e. `PY_NULL`, is treated
/// as "unset" by `fget_func_text_signature`).
///
/// # Safety
/// `obj` must point to a valid `Function`.
#[inline]
pub unsafe fn fset_func_text_signature(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        function_write_barrier(obj);
        (*(obj as *mut Function)).w_text_signature = value;
    }
}

/// `function.py:462-468 fset_func_name` parity:
///
/// ```python
/// def fset_func_name(self, space, w_name):
///     self._check_code_mutable("__name__")
///     if space.isinstance_w(w_name, space.w_text):
///         self.name = space.text_w(w_name)
///     else:
///         raise oefmt(space.w_TypeError,
///                     "__name__ must be set to a string object")
/// ```
///
/// `isinstance_str_w` mirrors `space.isinstance_w(w_name, space.w_text)`
/// — accepts `str` and any `str` subclass.
///
/// # Safety
/// `obj` must point to a valid `Function`.
#[inline]
pub unsafe fn fset_func_name(obj: PyObjectRef, name: PyObjectRef) -> Result<(), crate::PyError> {
    unsafe {
        _check_code_mutable(obj, "__name__")?; // function.py:463
        if !crate::baseobjspace::isinstance_str_w(name) {
            return Err(crate::PyError::type_error(
                "__name__ must be set to a string object",
            ));
        }
        function_set_func_name(obj, name);
        Ok(())
    }
}

// _check_code_mutable is defined above (function.py:367-370 parity).

/// `function.py:525-553 fset_func_code` parity:
///
/// ```python
/// def fset_func_code(self, space, w_code):
///     from pypy.interpreter.pycode import PyCode
///     if not self.can_change_code:
///         raise oefmt(space.w_AttributeError,
///                     "Cannot change code attribute of builtin functions")
///     code = space.interp_w(Code, w_code)
///     closure_len = 0
///     if self.closure:
///         closure_len = len(self.closure)
///     if isinstance(code, PyCode) and closure_len != len(code.co_freevars):
///         raise oefmt(space.w_ValueError,
///                     "%N() requires a code object with %d free vars, not "
///                     "%d", self, closure_len, len(code.co_freevars))
///     self.code = code
/// ```
///
/// `w_code` is the user-provided value as a `PyObjectRef`; this
/// helper enforces the `is_code(...)` (interp_w PyCode) check and
/// the closure / freevar arity invariant before reaching for the
/// inner code pointer.  Without these checks the previous setter
/// would happily reinterpret arbitrary objects as `*const CodeObject`
/// and corrupt the function's internal pointer (worse than main's
/// pre-fix shadowing-via-instance-dict regression).
///
/// # Safety
/// `obj` must point to a valid `Function`.
pub unsafe fn fset_func_code(obj: PyObjectRef, w_code: PyObjectRef) -> Result<(), crate::PyError> {
    unsafe {
        // function.py:527-529 — `can_change_code = False` on
        // BuiltinFunction / FunctionWithFixedCode.  pyre exposes
        // this via `_check_code_mutable` which raises the same
        // AttributeError.
        _check_code_mutable(obj, "__code__")?;
        // function.py:530 — `space.interp_w(Code, w_code)` raises
        // TypeError when `w_code` is not a `PyCode` instance.
        if w_code.is_null() || !crate::pycode::is_code(w_code) {
            return Err(crate::PyError::type_error(
                "__code__ must be set to a code object",
            ));
        }
        // function.py:531-537 — closure-vs-freevars arity check.
        // PyPy raises ValueError if the function has N closure cells
        // and the new code object declares M co_freevars with N != M;
        // mismatch leaves dangling cells / unbound freevars at runtime.
        let closure = function_get_closure(obj);
        let closure_len = if closure.is_null() || pyre_object::is_none(closure) {
            0
        } else {
            pyre_object::w_tuple_len(closure)
        };
        let raw_code = crate::pycode::w_code_get_ptr(w_code) as *const crate::CodeObject;
        let freevars_len = if raw_code.is_null() {
            0
        } else {
            (&(*raw_code).freevars).len()
        };
        if closure_len != freevars_len {
            let name = function_get_name(obj);
            return Err(crate::PyError::value_error(format!(
                "{name}() requires a code object with {closure_len} free vars, not {freevars_len}"
            )));
        }
        // function.py:538 self.fget_func_doc(space) — see test_issue1293.
        // Resolves the OLD code's docstring into `w_doc` *before* the
        // pointer flip so the cached value reflects the function's
        // original docstring, not the new code's first const.
        let _ = function_get_doc(obj);
        function_set_func_code(obj, w_code as *const ());
        Ok(())
    }
}

/// PyPy-compatible `__closure__` getter alias.
#[inline]
pub unsafe fn function_get_func_closure(obj: PyObjectRef) -> PyObjectRef {
    unsafe { function_get_closure(obj) }
}

/// PyPy-compatible `fget_func_closure`.
#[inline]
pub unsafe fn fget_func_closure(obj: PyObjectRef) -> PyObjectRef {
    unsafe {
        let value = function_get_func_closure(obj);
        if value.is_null() {
            pyre_object::w_none()
        } else {
            value
        }
    }
}

/// PyPy-compatible `__closure__` setter alias.
#[inline]
pub unsafe fn fset_func_closure(obj: PyObjectRef, closure: PyObjectRef) {
    unsafe {
        function_set_closure(obj, closure);
    }
}

/// `function.py:503-509 fget___module__`:
///
/// ```python
/// def fget___module__(self, space):
///     if self.w_module is None:
///         if self.w_func_globals is not None and not space.is_w(
///                 self.w_func_globals, space.w_None):
///             self.w_module = space.call_method(
///                 self.w_func_globals, "get", space.newtext("__name__"))
///         else:
///             self.w_module = space.w_None
///     return self.w_module
/// ```
///
/// Caches on first read: if w_module is PY_NULL (RPython-level None,
/// unset), computes from globals["__name__"] and stores into
/// self.w_module.  PY_NULL = RPython `None`, w_none() = space.w_None.
/// After fdel___module__ writes w_none(), subsequent reads return
/// None without re-computing from globals.
///
/// Routes through `w_func_globals_obj` (the canonical PyObjectRef the
/// user sees via `__globals__`) and dispatches `dict.get` so dict
/// subclasses that override `get` are observed.
#[inline]
pub unsafe fn fget___module__(obj: PyObjectRef) -> PyObjectRef {
    unsafe {
        let func = obj as *mut Function;
        // function.py:504: if self.w_module is None
        if (*func).w_module.is_null() {
            // function.py:505-506: space.call_method(self.w_func_globals,
            // "get", space.newtext("__name__"))
            let w_globals = (*func).w_func_globals_obj;
            if !w_globals.is_null() && !pyre_object::is_none(w_globals) {
                // Dispatch through `dict.get` so dict subclasses that
                // override `get` are observed.  When the lookup yields
                // PY_NULL we fall back to `space.w_None` per the
                // upstream attribute-not-found branch.
                let name_key = pyre_object::w_str_new("__name__");
                let result = crate::baseobjspace::call_method(w_globals, "get", &[name_key]);
                function_write_barrier(obj);
                (*func).w_module = if result.is_null() {
                    pyre_object::w_none()
                } else {
                    result
                };
            } else {
                // function.py:508: self.w_module = space.w_None
                function_write_barrier(obj);
                (*func).w_module = pyre_object::w_none();
            }
        }
        // function.py:509: return self.w_module
        (*func).w_module
    }
}

/// PyPy-compatible `descr_function__new__` helper.
#[inline]
pub unsafe fn descr_function__new__(
    code: *const (),
    w_globals: PyObjectRef,
    w_name: PyObjectRef,
    _argdefs: PyObjectRef,
    w_closure: PyObjectRef,
) -> PyObjectRef {
    unsafe {
        let _ = _argdefs;
        let name = if !w_name.is_null() && !pyre_object::is_none(w_name) {
            pyre_object::w_str_get_value(w_name).to_string()
        } else {
            String::new()
        };
        let closure = if w_closure.is_null() || pyre_object::is_none(w_closure) {
            pyre_object::PY_NULL
        } else {
            w_closure
        };
        function_new_with_closure(code, name, w_globals, closure)
    }
}

/// `function.py:511-513 fset___module__` parity:
///
/// ```python
/// def fset___module__(self, space, w_module):
///     self._check_code_mutable("__module__")
///     self.w_module = w_module
/// ```
#[inline]
pub unsafe fn fset___module__(obj: PyObjectRef, value: PyObjectRef) -> Result<(), crate::PyError> {
    unsafe {
        _check_code_mutable(obj, "__module__")?; // function.py:512
        function_write_barrier(obj);
        (*(obj as *mut Function)).w_module = value;
        Ok(())
    }
}

/// `function.py:515-517 fdel___module__` parity:
///
/// ```python
/// def fdel___module__(self, space):
///     self._check_code_mutable("__module__")
///     self.w_module = space.w_None
/// ```
#[inline]
pub unsafe fn fdel___module__(obj: PyObjectRef) -> Result<(), crate::PyError> {
    unsafe {
        _check_code_mutable(obj, "__module__")?; // function.py:516
        // function.py:517: self.w_module = space.w_None
        function_write_barrier(obj);
        (*(obj as *mut Function)).w_module = pyre_object::w_none();
        Ok(())
    }
}

/// PyPy-compatible `descr_function__new__` overload.
#[inline]
pub unsafe fn _cleanup_(_obj: PyObjectRef) -> bool {
    true
}

#[inline]
pub unsafe fn descr_builtinfunction__new__(
    code: *const (),
    w_globals: PyObjectRef,
    w_name: PyObjectRef,
    _argdefs: PyObjectRef,
    w_closure: PyObjectRef,
) -> PyObjectRef {
    unsafe { descr_function__new__(code, w_globals, w_name, _argdefs, w_closure) }
}

/// `funcobject.c func_new_impl` — `FunctionType(code, globals,
/// name=None, argdefs=None, closure=None, kwdefaults=None)`.
///
/// `args[0]` is the class (the `function` type); the remaining
/// positional args plus the trailing `__pyre_kw__` dict
/// (`split_builtin_kwargs`) supply the constructor parameters, so both
/// `FunctionType(code, g)` and `FunctionType(code, g, closure=c,
/// kwdefaults=k)` reach the same slot resolution.
pub fn descr_function_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    crate::builtins::kwarg_reject_unknown(
        kwargs,
        &[
            "code",
            "globals",
            "name",
            "argdefs",
            "closure",
            "kwdefaults",
        ],
        "function",
    )?;
    // `positional[0]` is `cls`; the constructor parameters start at index 1.
    let pos = |i: usize| positional.get(i).copied().unwrap_or(PY_NULL);
    let resolve = |i: usize, name: &str| {
        let p = pos(i);
        if p.is_null() {
            crate::builtins::kwarg_get(kwargs, name).unwrap_or(PY_NULL)
        } else {
            p
        }
    };
    let w_code = resolve(1, "code");
    let w_globals = resolve(2, "globals");
    let w_name = resolve(3, "name");
    let w_argdefs = resolve(4, "argdefs");
    let w_closure = resolve(5, "closure");
    let w_kwdefaults = crate::builtins::kwarg_get(kwargs, "kwdefaults").unwrap_or(PY_NULL);

    if w_code.is_null() || !unsafe { crate::pycode::is_code(w_code) } {
        return Err(crate::PyError::type_error(
            "function() argument 'code' must be code, not ...",
        ));
    }
    // `PyDict_Check` accepts dict subclasses (annotationlib hands a
    // `_StringifierDict`); resolve the backing storage rather than
    // demanding an exact `dict`.
    let w_globals_backing = crate::type_methods::resolve_dict_backing(w_globals);
    if w_globals_backing.is_null() {
        return Err(crate::PyError::type_error(
            "function() argument 'globals' must be dict, not ...",
        ));
    }
    let code_ptr = unsafe { crate::w_code_get_ptr(w_code) } as *const crate::CodeObject;
    let name = if w_name.is_null() || unsafe { pyre_object::is_none(w_name) } {
        unsafe { (*code_ptr).obj_name.to_string() }
    } else if unsafe { pyre_object::is_str(w_name) } {
        unsafe { pyre_object::w_str_get_value(w_name).to_string() }
    } else {
        return Err(crate::PyError::type_error(
            "arg 3 (name) must be None or string",
        ));
    };
    let closure = if w_closure.is_null() || unsafe { pyre_object::is_none(w_closure) } {
        PY_NULL
    } else {
        w_closure
    };
    // Normalise a dict-subclass globals to its backing `W_DictObject`
    // (the call-path frame builder reads the storage proxy off the
    // `__globals__` object directly; a subclass instance carries no proxy
    // and would fault — same normalisation the exec/eval `createframe_obj`
    // path applies).  `__missing__`-based forward references are not
    // surfaced through the backing, but defined-name annotations resolve.
    // `function.py:57 self.w_func_globals = w_globals` stores the dict
    // object as the function's sole globals carrier.
    let func = function_new_with_closure(w_code as *const (), name, w_globals_backing, closure);
    let qualname = pyre_object::w_str_new(unsafe { (*code_ptr).qualname.as_ref() });
    unsafe { function_set_qualname(func, qualname) };
    if !w_argdefs.is_null() && !unsafe { pyre_object::is_none(w_argdefs) } {
        if !unsafe { pyre_object::is_tuple(w_argdefs) } {
            return Err(crate::PyError::type_error(
                "arg 4 (defaults) must be None or tuple",
            ));
        }
        unsafe { function_set_defaults(func, w_argdefs) };
    }
    if !w_kwdefaults.is_null() && !unsafe { pyre_object::is_none(w_kwdefaults) } {
        if !unsafe { pyre_object::is_dict(w_kwdefaults) } {
            return Err(crate::PyError::type_error("kwdefaults must be a dict"));
        }
        unsafe { function_set_kwdefaults(func, w_kwdefaults) };
    }
    Ok(func)
}

/// PyPy-compatible static registry hook.
#[inline]
pub fn add_to_table() {}

/// PyPy-compatible `__doc__` getter.
#[inline]
pub unsafe fn fget_func_doc(obj: PyObjectRef) -> PyObjectRef {
    function_get_doc(obj)
}

/// function.py:400 — `fset_func_doc` descriptor.
#[inline]
pub unsafe fn fset_func_doc(obj: PyObjectRef, value: PyObjectRef) -> Result<(), crate::PyError> {
    unsafe { function_set_doc(obj, value) }
}

/// function.py:404 — `fdel_func_doc` descriptor.
#[inline]
pub unsafe fn fdel_func_doc(obj: PyObjectRef) -> Result<(), crate::PyError> {
    unsafe { function_del_doc(obj) }
}

/// `pypy/objspace/std/util.py:6-13` — `id()` of a plain `int` / `float`
/// / `complex` is its value tagged `(value << IDTAG_SHIFT) | IDTAG_*`;
/// the unique-ified immutables (empty/short `bytes`/`str`, empty
/// `tuple`/`frozenset`) use `IDTAG_SPECIAL`.
const IDTAG_SHIFT: i64 = 4;
const IDTAG_INT: i64 = 1;
const IDTAG_FLOAT: i64 = 5;
const IDTAG_COMPLEX: i64 = 7;
const IDTAG_SPECIAL: i64 = 11;

#[inline]
pub fn immutable_unique_id(obj: PyObjectRef) -> Option<PyObjectRef> {
    // `W_AbstractIntObject.immutable_unique_id` (intobject.py:55-60): a
    // plain `int` — `W_IntObject` or the BigInt-backed `W_LongObject` —
    // has a value-derived id `(bigint_w << IDTAG_SHIFT) | IDTAG_INT`,
    // wrapped as an `int` (a `long` when it overflows i64).
    // `W_FloatObject.immutable_unique_id` (floatobject.py:206-215) does
    // the same with the float bit pattern (`float2longlong`) and
    // `IDTAG_FLOAT`. `bool` (`W_BoolObject.immutable_unique_id` returns
    // None, boolobject.py:28) and `int`/`float` subclasses
    // (`user_overridden_class`) return `None`, so `space.id` falls back
    // to the address-based uid.
    unsafe {
        if is_exact_type(obj, &INT_TYPE) {
            // `b.lshift(IDTAG_SHIFT).int_or_(IDTAG_INT)`; the shifted
            // value is even, so `| IDTAG_INT` equals `+ IDTAG_INT`.
            let b = (pyre_object::functional::range_obj_to_bigint(obj) << IDTAG_SHIFT as usize)
                + malachite_bigint::BigInt::from(IDTAG_INT);
            return Some(pyre_object::functional::range_bigint_to_obj(b));
        }
        if is_exact_type(obj, &FLOAT_TYPE) {
            // `float2longlong(float_w(self))` reinterprets the f64 bits as
            // a signed i64; the same `| IDTAG_FLOAT` == `+ IDTAG_FLOAT`.
            let bits = pyre_object::floatobject::w_float_get_value(obj).to_bits() as i64;
            let b = (malachite_bigint::BigInt::from(bits) << IDTAG_SHIFT as usize)
                + malachite_bigint::BigInt::from(IDTAG_FLOAT);
            return Some(pyre_object::functional::range_bigint_to_obj(b));
        }
        if is_exact_type(obj, &COMPLEX_TYPE) {
            // `(real_b << 64 | imag_b) << IDTAG_SHIFT | IDTAG_COMPLEX`
            // (complexobject.py:303-314): the real bits are signed
            // (`float2longlong`), the imag bits unsigned (`r_ulonglong`);
            // the high/low 64-bit halves don't overlap, so each `|` is a
            // `+`.
            let real_bits = pyre_object::complexobject::w_complex_get_real(obj).to_bits() as i64;
            let imag_bits = pyre_object::complexobject::w_complex_get_imag(obj).to_bits();
            let combined = (malachite_bigint::BigInt::from(real_bits) << 64usize)
                + malachite_bigint::BigInt::from(imag_bits);
            let b =
                (combined << IDTAG_SHIFT as usize) + malachite_bigint::BigInt::from(IDTAG_COMPLEX);
            return Some(pyre_object::functional::range_bigint_to_obj(b));
        }
        if is_exact_type(obj, &TUPLE_TYPE) {
            // `W_AbstractTupleObject.immutable_unique_id`
            // (tupleobject.py:57-62): only the empty tuple is unique-ified
            // — a non-empty tuple (`length() > 0`) returns `None` and
            // `space.id` falls back to the address-based uid. `tuple`
            // subclasses (`user_overridden_class`) also return `None`; the
            // exact-type gate excludes them. The empty tuple has base value
            // 258: `(258 << IDTAG_SHIFT) | IDTAG_SPECIAL`; the shifted value
            // has low 4 bits clear, so `| IDTAG_SPECIAL` == `+`. The uid
            // fits i64, so it is wrapped with `w_int_new` (`space.newint`).
            if pyre_object::tupleobject::w_tuple_len(obj) > 0 {
                return None;
            }
            let uid = (258i64 << IDTAG_SHIFT) + IDTAG_SPECIAL;
            return Some(pyre_object::intobject::w_int_new(uid));
        }
        if is_exact_type(obj, &pyre_object::bytesobject::BYTES_TYPE) {
            // `W_AbstractBytesObject.immutable_unique_id`
            // (bytesobject.py:40-52): `len(s) > 1` is address-based
            // (`compute_unique_id(s)`) so returning `None` falls back to the
            // object address (invariant-preserving — distinct `bytes` never
            // share storage). `len(s) <= 1` is unique-ified:
            // `base = ord(s[0])` (0..255) for one byte, `base = 256` for the
            // empty bytes, `uid = (base << IDTAG_SHIFT) | IDTAG_SPECIAL`.
            let len = pyre_object::bytesobject::w_bytes_len(obj);
            if len > 1 {
                return None;
            }
            let base: i64 = if len == 1 {
                pyre_object::bytesobject::w_bytes_getitem(obj, 0) as i64
            } else {
                256
            };
            let uid = (base << IDTAG_SHIFT) + IDTAG_SPECIAL;
            return Some(pyre_object::intobject::w_int_new(uid));
        }
        if is_exact_type(obj, &STR_TYPE) {
            // `W_UnicodeObject.immutable_unique_id` (unicodeobject.py:115-131).
            // `l` is the codepoint count (`_len()`), not the byte length.
            // `l > 1` is address-based (upstream `compute_unique_id(_utf8) +
            // IDTAG_ALT_UID`); returning `None` falls back to the object
            // address, invariant-preserving with `is_w` returning `false`
            // for distinct len>1 strings. `l <= 1` is unique-ified: for a
            // single codepoint `base = ~codepoint_at_pos(_utf8, 0)`
            // (negative), and `base = 257` for the empty string.
            let l = pyre_object::unicodeobject::w_str_len(obj);
            if l > 1 {
                return None;
            }
            let base: i64 = if l == 1 {
                // `code_points()` yields the codepoint regardless of
                // surrogates, matching `rutf8.codepoint_at_pos`.
                let cp = pyre_object::unicodeobject::w_str_get_wtf8(obj)
                    .code_points()
                    .next()
                    .expect("len==1 str has a code point")
                    .to_u32();
                // `(neg << IDTAG_SHIFT) | IDTAG_SPECIAL` == `+` (low 4 bits 0).
                !(cp as i64)
            } else {
                257
            };
            let uid = (base << IDTAG_SHIFT) + IDTAG_SPECIAL;
            return Some(pyre_object::intobject::w_int_new(uid));
        }
        if is_exact_type(obj, &pyre_object::setobject::FROZENSET_TYPE) {
            // `W_FrozensetObject.immutable_unique_id` (setobject.py:602-607):
            // a non-empty frozenset (`length() > 0`) and `frozenset`
            // subclasses (excluded by the exact-type gate) return `None`.
            // The empty frozenset is unique-ified with base value 259. The
            // mutable `set` has its own type and does not override this, so
            // the `FROZENSET_TYPE` gate does not match it.
            if pyre_object::setobject::w_set_len(obj) > 0 {
                return None;
            }
            let uid = (259i64 << IDTAG_SHIFT) + IDTAG_SPECIAL;
            return Some(pyre_object::intobject::w_int_new(uid));
        }
    }
    None
}

/// PyPy-compatible `find` helper.
#[inline]
pub fn find(_identifier: &str) -> PyObjectRef {
    let _ = _identifier;
    pyre_object::PY_NULL
}

#[inline]
fn is_builtin_code(obj: PyObjectRef) -> bool {
    unsafe { crate::gateway::is_builtin_code(obj) }
}

#[inline]
pub fn descr_init() {}

#[inline]
pub unsafe fn descr_classmethod__new__(
    _subtype: PyObjectRef,
    w_function: PyObjectRef,
) -> PyObjectRef {
    let _ = _subtype;
    if w_function.is_null() {
        pyre_object::w_none()
    } else {
        pyre_object::function::w_classmethod_new(w_function)
    }
}

#[inline]
pub unsafe fn descr_classmethod_get(
    w_obj: PyObjectRef,
    obj: PyObjectRef,
    w_cls: PyObjectRef,
) -> PyObjectRef {
    unsafe {
        let _ = w_cls;
        if obj.is_null() || pyre_object::is_none(obj) {
            w_obj
        } else {
            let func = pyre_object::w_classmethod_get_func(w_obj);
            let cls = if w_obj.is_null() {
                pyre_object::w_none()
            } else {
                obj
            };
            pyre_object::w_method_new(func, cls, cls)
        }
    }
}

#[inline]
pub unsafe fn descr_staticmethod__new__(
    _subtype: PyObjectRef,
    w_function: PyObjectRef,
) -> PyObjectRef {
    let _ = _subtype;
    if w_function.is_null() {
        pyre_object::w_none()
    } else {
        pyre_object::function::w_staticmethod_new(w_function)
    }
}

#[inline]
pub unsafe fn descr_staticmethod_get(
    obj: PyObjectRef,
    _obj: PyObjectRef,
    _cls: PyObjectRef,
) -> PyObjectRef {
    unsafe {
        let _ = (_obj, _cls);
        if obj.is_null() {
            pyre_object::w_none()
        } else {
            pyre_object::w_staticmethod_get_func(obj)
        }
    }
}

#[inline]
pub unsafe fn descr_method__new__(
    _subtype: PyObjectRef,
    w_function: PyObjectRef,
    w_instance: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    let _ = _subtype;
    if w_function.is_null() || !crate::baseobjspace::callable_w(w_function) {
        return Err(crate::PyError::type_error(
            "first argument must be callable",
        ));
    }
    if w_instance.is_null() || unsafe { pyre_object::is_none(w_instance) } {
        return Err(crate::PyError::type_error("instance must not be None"));
    }
    let w_class = crate::typedef::r#type(w_instance).unwrap_or(pyre_object::PY_NULL);
    Ok(pyre_object::w_method_new(w_function, w_instance, w_class))
}

/// `interp2app` unwraps the receiver as `Method` before entering PyPy's
/// method bodies.  Reproduce that gateway contract before reading the Rust
/// payload, including for direct descriptor calls with a foreign receiver.
#[inline]
pub fn require_method(method: PyObjectRef, name: &str) -> Result<PyObjectRef, crate::PyError> {
    if method.is_null() || !unsafe { pyre_object::function::is_method(method) } {
        let received = crate::typedef::r#type(method)
            .map(|tp| unsafe { pyre_object::w_type_get_name(tp) })
            .unwrap_or("object");
        return Err(crate::PyError::type_error(format!(
            "descriptor '{name}' requires a 'method' object but received a '{received}'"
        )));
    }
    Ok(method)
}

#[inline]
pub unsafe fn descr_method_get(
    method: PyObjectRef,
    obj: PyObjectRef,
    cls: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    let method = require_method(method, "__get__")?;
    let obj_is_none = obj.is_null() || unsafe { pyre_object::is_none(obj) };
    let cls_is_none = cls.is_null() || unsafe { pyre_object::is_none(cls) };
    if obj_is_none && cls_is_none {
        return Err(crate::PyError::type_error("__get__(None, None) is invalid"));
    }
    Ok(method)
}

#[inline]
pub fn descr_method_call(args: &[PyObjectRef]) -> crate::PyResult {
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    let method = require_method(
        positional.first().copied().unwrap_or(pyre_object::PY_NULL),
        "__call__",
    )?;
    let call_args = positional.get(1..).unwrap_or(&[]);
    if !crate::builtins::has_real_kwargs(kwargs) {
        return crate::call::call_function_impl_result(method, call_args);
    }
    let keyword_args: Vec<(rustpython_wtf8::Wtf8Buf, PyObjectRef)> = unsafe {
        pyre_object::w_dict_str_entries(kwargs.unwrap())
            .into_iter()
            .filter(|(name, _)| name != "__pyre_kw__")
            .map(|(name, value)| (rustpython_wtf8::Wtf8Buf::from_string(name), value))
            .collect()
    };
    crate::eval::CURRENT_FRAME.with(|current| {
        let frame = current.get();
        if frame.is_null() {
            return Err(crate::PyError::runtime_error(
                "method call has no current frame",
            ));
        }
        crate::call::call_with_kwargs(unsafe { &mut *frame }, method, call_args, &keyword_args)
    })
}

#[inline]
pub unsafe fn descr_method_eq(
    this: PyObjectRef,
    other: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    let this = require_method(this, "__eq__")?;
    if !unsafe { pyre_object::is_method(other) } {
        return Ok(pyre_object::special::w_not_implemented());
    }
    let funcs_equal =
        crate::baseobjspace::eq_w(unsafe { pyre_object::w_method_get_func(this) }, unsafe {
            pyre_object::w_method_get_func(other)
        })?;
    let selves_identical =
        unsafe { pyre_object::w_method_get_self(this) == pyre_object::w_method_get_self(other) };
    Ok(pyre_object::w_bool_from(funcs_equal && selves_identical))
}

#[inline]
pub unsafe fn descr_method_ne(
    this: PyObjectRef,
    other: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    require_method(this, "__ne__")?;
    let equal = unsafe { descr_method_eq(this, other)? };
    if unsafe { pyre_object::is_not_implemented(equal) } {
        Ok(equal)
    } else {
        Ok(pyre_object::w_bool_from(!unsafe {
            pyre_object::w_bool_get_value(equal)
        }))
    }
}

#[inline]
pub unsafe fn descr_method_repr(obj: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    let obj = require_method(obj, "__repr__")?;
    let function = unsafe { pyre_object::w_method_get_func(obj) };
    let instance = unsafe { pyre_object::w_method_get_self(obj) };
    let w_name = match crate::baseobjspace::getattr_str(function, "__qualname__") {
        Ok(value) => Some(value),
        Err(err) if err.kind == crate::PyErrorKind::AttributeError => {
            match crate::baseobjspace::getattr_str(function, "__name__") {
                Ok(value) => Some(value),
                Err(err) if err.kind == crate::PyErrorKind::AttributeError => None,
                Err(err) => return Err(err),
            }
        }
        Err(err) => return Err(err),
    };
    let name = w_name
        .filter(|&value| unsafe { pyre_object::is_str(value) })
        .and_then(|value| unsafe { pyre_object::w_str_get_value_opt(value) })
        .unwrap_or("?");
    let instance_repr = unsafe { crate::display::py_repr(instance)? };
    Ok(pyre_object::w_str_new(&format!(
        "<bound method {name} of {instance_repr}>"
    )))
}

#[inline]
pub unsafe fn descr_method_getattribute(
    obj: PyObjectRef,
    name: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    let obj = require_method(obj, "__getattribute__")?;
    if !unsafe { pyre_object::is_str(name) } {
        return Err(crate::PyError::type_error("attribute name must be string"));
    }
    let Some(name) = (unsafe { pyre_object::w_str_get_value_opt(name) }) else {
        return Err(crate::PyError::type_error("attribute name must be string"));
    };
    // function.py:604-614 — method attributes win, except `__doc__`;
    // an AttributeError falls back to the wrapped function.
    if name != "__doc__" {
        match crate::baseobjspace::object_getattribute(obj, name) {
            Ok(value) => return Ok(value),
            Err(err) if err.kind == crate::PyErrorKind::AttributeError => {}
            Err(err) => return Err(err),
        }
    }
    let function = unsafe { pyre_object::w_method_get_func(obj) };
    crate::baseobjspace::getattr_str(function, name)
}

#[inline]
pub unsafe fn descr_method_hash(obj: PyObjectRef) -> Result<i64, crate::PyError> {
    let obj = require_method(obj, "__hash__")?;
    let function = unsafe { pyre_object::w_method_get_func(obj) };
    let instance = unsafe { pyre_object::w_method_get_self(obj) };
    let x = pyre_object::gc_hook::gc_identity_hash(instance as usize) as i64;
    let y = crate::baseobjspace::hash_w_strict(function)?;
    let value = x ^ y;
    Ok(if value == -1 { -2 } else { value })
}

#[inline]
pub unsafe fn descr_method__reduce__(obj: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    let obj = require_method(obj, "__reduce__")?;
    let function = unsafe { pyre_object::w_method_get_func(obj) };
    let instance = unsafe { pyre_object::w_method_get_self(obj) };
    let name = crate::baseobjspace::getattr_str(function, "__name__")?;
    Ok(pyre_object::w_tuple_new(vec![
        crate::baseobjspace::builtin_callable("getattr"),
        pyre_object::w_tuple_new(vec![instance, name]),
    ]))
}

#[inline]
pub unsafe fn is_w(_obj: PyObjectRef, _other: PyObjectRef) -> bool {
    _obj == _other
}

/// PyPy-compatible `descr_function_call` helper.
#[inline]
pub fn descr_function_call(args: &[PyObjectRef]) -> PyObjectRef {
    if args.is_empty() {
        pyre_object::PY_NULL
    } else {
        call_args(args[0], &args[1..])
    }
}

/// PyPy-compatible `descr_function_get` helper.
#[inline]
pub unsafe fn descr_function_get(
    _func: PyObjectRef,
    obj: PyObjectRef,
    cls: PyObjectRef,
) -> PyObjectRef {
    let _ = cls;
    if obj.is_null() || unsafe { pyre_object::is_none(obj) } {
        _func
    } else {
        pyre_object::w_method_new(_func, obj, cls)
    }
}

/// PyPy-compatible `descr_function_repr` helper.
#[inline]
pub unsafe fn descr_function_repr(obj: PyObjectRef) -> PyObjectRef {
    unsafe {
        let name = function_get_name(obj);
        pyre_object::w_str_new(&format!("function {name}"))
    }
}

/// PyPy-compatible `__code__` getter for direct descriptors.
#[inline]
pub unsafe fn fget_func_code(obj: PyObjectRef) -> *const () {
    unsafe { function_get_code(obj) }
}

/// PyPy-compatible `descr__reduce__` helper.
#[inline]
pub fn descr_function__reduce__(_obj: PyObjectRef) -> PyObjectRef {
    pyre_object::w_tuple_new(vec![
        pyre_object::w_tuple_new(vec![]),
        pyre_object::w_tuple_new(vec![]),
    ])
}

/// PyPy-compatible `descr__setstate__` helper.
#[inline]
pub fn descr_function__setstate__(_obj: PyObjectRef, _state: PyObjectRef) {
    let _ = _state;
}

#[inline]
pub fn __init__() {}

#[inline]
pub fn __repr__() -> String {
    "function".to_string()
}

/// PyPy-compatible `__call__` alias helper.
#[inline]
pub fn call(frame: PyObjectRef, args: &[PyObjectRef]) -> PyObjectRef {
    call_args(frame, args)
}

/// PyPy-compatible call fast-path hooks.
#[inline]
pub fn call_args(func: PyObjectRef, args: &[PyObjectRef]) -> PyObjectRef {
    crate::call_function(func, args)
}

/// PyPy-compatible `call_obj_args` helper.
#[inline]
pub fn call_obj_args(func: PyObjectRef, obj: PyObjectRef, args: &[PyObjectRef]) -> PyObjectRef {
    crate::baseobjspace::call_obj_args(func, obj, args)
}

/// PyPy-compatible `call_args` instance method.
#[inline]
pub fn function_call_args(func: PyObjectRef, args: &[PyObjectRef]) -> PyObjectRef {
    call_args(func, args)
}

/// PyPy-compatible `call_obj_args` instance method.
#[inline]
pub(crate) fn function_call_obj_args(
    func: PyObjectRef,
    obj: PyObjectRef,
    args: &[PyObjectRef],
) -> PyObjectRef {
    call_obj_args(func, obj, args)
}

/// PyPy-compatible `funccall` helper.
#[inline]
pub fn funccall(func: PyObjectRef, args: &[PyObjectRef]) -> PyObjectRef {
    call_args(func, args)
}

/// baseobjspace.py: `space._code_of_sys_exc_info` — the BuiltinCode object
/// backing `sys.exc_info`, captured at sys-module init time so the JIT
/// fast-path in `funccall_valuestack` can recognize this specific call and
/// inline `exc_info_direct` without going through the regular dispatch.
///
/// PyPy stores this on the space; pyre is single-space-per-thread, so a
/// thread-local cell suffices. The paired `direct_fn` returns the same
/// `(type, value, traceback)` tuple as the regular closure but skips the
/// builtin-call setup.
type ExcInfoDirectFn = fn() -> PyObjectRef;
thread_local! {
    static SYS_EXC_INFO_CODE: std::cell::Cell<*const ()> = const { std::cell::Cell::new(std::ptr::null()) };
    static SYS_EXC_INFO_DIRECT_FN: std::cell::Cell<Option<ExcInfoDirectFn>> = const { std::cell::Cell::new(None) };
}

/// Register the BuiltinCode pointer + direct helper for `sys.exc_info`.
///
/// Called once during sys module init (after the builtin is created). The
/// `code` pointer is the BuiltinCode object underlying the `exc_info`
/// builtin function; `direct_fn` is the JIT-direct equivalent of the
/// closure body. `funccall_valuestack` consults both to take the fast path.
pub fn register_sys_exc_info_path(code: *const (), direct_fn: ExcInfoDirectFn) {
    SYS_EXC_INFO_CODE.with(|cell| cell.set(code));
    SYS_EXC_INFO_DIRECT_FN.with(|cell| cell.set(Some(direct_fn)));
}

#[inline]
fn sys_exc_info_code() -> *const () {
    SYS_EXC_INFO_CODE.with(|cell| cell.get())
}

#[inline]
fn sys_exc_info_direct_fn() -> Option<ExcInfoDirectFn> {
    SYS_EXC_INFO_DIRECT_FN.with(|cell| cell.get())
}

/// function.py:139-203 `funccall_valuestack` — fast-path call dispatcher.
///
/// Dispatches based on `code.fast_natural_arity`:
/// - nargs == arity (0-4): direct builtin fastcall from stack (no Vec alloc)
/// - (nargs | FLATPYCALL) == arity: _flat_pycall (user code, exact arity)
/// - FLATPYCALL + defaults: _flat_pycall_defaults
/// - Fallback: allocate Vec via peekvalues + generic call path
pub fn funccall_valuestack(
    func: PyObjectRef,
    nargs: usize,
    frame: &mut crate::pyframe::PyFrame,
    dropvalues: usize,
    methodcall: bool,
) -> PyObjectRef {
    // rpython/rlib/rstack.py:42 stack_check(): every interpreter call
    // boundary checks the native stack synchronously, so deep recursion
    // raises RecursionError instead of letting the OS abort on a
    // guard-page hit. funccall_valuestack is the bytecode CALL fast
    // path that bypasses dispatch_callable, so it carries its own
    // probe. Also drain any JIT-prologue pending overflow.
    if let Err(e) = crate::stack_check::drain_jit_pending_exception()
        .and_then(|_| crate::stack_check::stack_check())
    {
        crate::call::set_call_error(e);
        return pyre_object::PY_NULL;
    }

    let code = unsafe { crate::getcode(func) };

    // function.py:146-150 — JIT direct path for `sys.exc_info()` with no
    // arguments: skip the builtin call entirely and inline the tuple
    // construction. PyPy uses `space._code_of_sys_exc_info`; pyre uses the
    // thread-local cache populated during sys module init.
    if nargs == 0
        && majit_metainterp::jit::we_are_jitted()
        && std::ptr::eq(code, sys_exc_info_code())
    {
        if let Some(direct_fn) = sys_exc_info_direct_fn() {
            frame.dropvalues(dropvalues);
            return direct_fn();
        }
    }

    let fast_natural_arity =
        unsafe { crate::pycode::code_get_fast_natural_arity(code as PyObjectRef) } as usize;

    // function.py:153-184 — nargs == fast_natural_arity: builtin fast path
    // baseobjspace.py:1243 — skip when profiling (c_call/c_return events)
    if nargs == fast_natural_arity && nargs <= 4 && !frame.get_is_being_profiled() {
        debug_assert!(
            (fast_natural_arity & crate::FLATPYCALL as usize) == 0,
            "FLATPYCALL bit set on arity {fast_natural_arity} — not a builtin code"
        );
        let builtin_fn = unsafe { crate::builtin_code_get(code as PyObjectRef) };
        // function.py:154-184 — BuiltinCodeN.fastcall_N dispatch.
        // Pyre builtins share a single fn(&[PyObjectRef]) signature, so we
        // build a fixed-size stack array instead of heap-allocating a Vec.
        let result = match nargs {
            0 => {
                frame.dropvalues(dropvalues);
                builtin_fn(&[])
            }
            1 => {
                let a0 = frame.peekvalue(0);
                frame.dropvalues(dropvalues);
                builtin_fn(&[a0])
            }
            2 => {
                // function.py:168 — peekvalue order: 0=top, 1=below top
                let a0 = frame.peekvalue(1);
                let a1 = frame.peekvalue(0);
                frame.dropvalues(dropvalues);
                builtin_fn(&[a0, a1])
            }
            3 => {
                let a0 = frame.peekvalue(2);
                let a1 = frame.peekvalue(1);
                let a2 = frame.peekvalue(0);
                frame.dropvalues(dropvalues);
                builtin_fn(&[a0, a1, a2])
            }
            4 => {
                let a0 = frame.peekvalue(3);
                let a1 = frame.peekvalue(2);
                let a2 = frame.peekvalue(1);
                let a3 = frame.peekvalue(0);
                frame.dropvalues(dropvalues);
                builtin_fn(&[a0, a1, a2, a3])
            }
            _ => unreachable!(),
        };
        return match result {
            Ok(v) => v,
            Err(e) => {
                crate::call::set_call_error(e);
                pyre_object::PY_NULL
            }
        };
    }

    // function.py:185-187 — (nargs | FLATPYCALL) == fast_natural_arity
    if (nargs | crate::FLATPYCALL as usize) == fast_natural_arity {
        return _flat_pycall(func, code, nargs, frame, dropvalues);
    }

    // function.py:188-193 — FLATPYCALL bit set + nargs within defaults range
    if (fast_natural_arity & crate::FLATPYCALL as usize) != 0 {
        let natural_arity = fast_natural_arity & 0xff;
        if nargs < natural_arity {
            let raw_defs = unsafe { crate::function_get_defaults(func) };
            let defs = raw_defs;
            let defs_len = if defs.is_null() || !unsafe { pyre_object::is_tuple(defs) } {
                0
            } else {
                unsafe { pyre_object::w_tuple_len(defs) }
            };
            if nargs >= natural_arity.saturating_sub(defs_len) {
                return _flat_pycall_defaults(
                    func,
                    code,
                    nargs,
                    frame,
                    defs,
                    natural_arity - nargs,
                    dropvalues,
                );
            }
        }
    }

    // function.py:194-199 — PASSTHROUGHARGS1 dispatch.
    // PyPy's BuiltinCodePassThroughArguments1.funcrun_obj receives w_obj
    // separately from an Arguments rest, then concatenates them as
    // `args_w = [w_obj] + _args_w` before calling the unwrapped fn. Pyre's
    // single BuiltinCodeFn signature already takes a flat slice, so the
    // peek/Arguments split is structural — the final closure invocation
    // sees `[w_obj, ...rest]` exactly as PyPy's post-merge args_w.
    if fast_natural_arity == crate::PASSTHROUGHARGS1 as usize && nargs >= 1 {
        let builtin_fn = unsafe { crate::builtin_code_get(code as PyObjectRef) };
        let w_obj = frame.peekvalue(nargs - 1);
        let rest = frame.make_arguments(nargs - 1, false, func);
        let mut args_w = Vec::with_capacity(nargs);
        args_w.push(w_obj);
        args_w.extend_from_slice(&rest);
        frame.dropvalues(dropvalues);
        return match builtin_fn(&args_w) {
            Ok(v) => v,
            Err(e) => {
                crate::call::set_call_error(e);
                pyre_object::PY_NULL
            }
        };
    }

    // function.py:201-203 — fallback: build Arguments via make_arguments
    // (carries methodcall + w_function for diagnostics) and dispatch through
    // call_args.
    let args = frame.make_arguments(nargs, methodcall, func);
    frame.dropvalues(dropvalues);
    funccall(func, &args)
}

/// function.py:206-214 `_flat_pycall` — create frame directly from stack.
///
/// For user functions with exact arity match (no defaults needed).
/// Copies args from caller's value stack into the new frame's locals
/// without intermediate Vec allocation.
fn _flat_pycall(
    func: PyObjectRef,
    code: *const (),
    nargs: usize,
    frame: &mut crate::pyframe::PyFrame,
    dropvalues: usize,
) -> PyObjectRef {
    // call.rs:423-424 parity — increment call depth for JIT depth tracking.
    let _depth_guard = crate::call::increment_call_depth();
    let w_globals = unsafe { function_get_globals_obj(func) };
    let closure = unsafe { function_get_closure(func) };

    // function.py:208-209 — createframe(code, w_func_globals, self)
    // FrameBox: the callee runs through the JIT, so it must be a
    // header-bearing heap frame (write barrier reads a valid header at
    // frame - GC_HEADER_SIZE) rather than a bare interpreter-stack frame.
    let mut new_frame = crate::pyframe::FrameBox::new(
        match crate::pyframe::PyFrame::try_new_for_call_with_closure_and_globals_obj(
            code,
            &[], // locals filled below directly from stack
            w_globals,
            frame.execution_context,
            closure,
            crate::pyframe::FrameLocalsArrayAllocation::OldGenGc,
        ) {
            Ok(f) => f,
            Err(e) => {
                crate::call::set_call_error(e);
                return pyre_object::PY_NULL;
            }
        },
    );

    // function.py:210-211 — copy from stack into locals directly
    // peekvalue(nargs-1-i) gives bottom-to-top order (matching local slot order)
    for i in 0..nargs {
        new_frame.locals_w_mut()[i] = frame.peekvalue(nargs - 1 - i);
    }
    frame.dropvalues(dropvalues);
    new_frame.fix_array_ptrs();

    // function.py:214 — return new_frame.run(self.name, self.qualname)
    // Generator/coroutine: wrap the frame in a generator object and hand it
    // ownership (no execution). Normal functions execute through the JIT-aware
    // eval, which needs the locals roots registered for the duration.
    if new_frame._is_generator_or_coroutine() {
        match new_frame.into_generator() {
            Ok(v) => v,
            Err(e) => {
                crate::call::set_call_error(e);
                pyre_object::PY_NULL
            }
        }
    } else {
        let _caller_locals_root = FrameLocalsRoot::new(frame);
        let _callee_locals_root = FrameLocalsRoot::new(&mut new_frame);
        let eval_fn = crate::call::get_eval_fn();
        match eval_fn(&mut new_frame) {
            Ok(v) => v,
            Err(e) => {
                crate::call::set_call_error(e);
                pyre_object::PY_NULL
            }
        }
    }
}

/// function.py:217-231 `_flat_pycall_defaults` — flat call with defaults.
///
/// Same as `_flat_pycall` but also fills missing positional args from
/// `self.defs_w[ndefs - defs_to_load ..]`.
/// `defs` is the pre-unwrapped defaults tuple (already null-checked and
/// verified as a tuple by the caller in `funccall_valuestack`).
fn _flat_pycall_defaults(
    func: PyObjectRef,
    code: *const (),
    nargs: usize,
    frame: &mut crate::pyframe::PyFrame,
    defs: PyObjectRef,
    defs_to_load: usize,
    dropvalues: usize,
) -> PyObjectRef {
    let _depth_guard = crate::call::increment_call_depth();
    let w_globals = unsafe { function_get_globals_obj(func) };
    let closure = unsafe { function_get_closure(func) };

    // FrameBox: header-bearing heap frame for the JIT write barrier.
    let mut new_frame = crate::pyframe::FrameBox::new(
        match crate::pyframe::PyFrame::try_new_for_call_with_closure_and_globals_obj(
            code,
            &[], // locals filled below
            w_globals,
            frame.execution_context,
            closure,
            crate::pyframe::FrameLocalsArrayAllocation::OldGenGc,
        ) {
            Ok(f) => f,
            Err(e) => {
                crate::call::set_call_error(e);
                return pyre_object::PY_NULL;
            }
        },
    );

    // function.py:221-222 — copy positional args from stack
    for i in 0..nargs {
        new_frame.locals_w_mut()[i] = frame.peekvalue(nargs - 1 - i);
    }

    // function.py:224-229 — fill remaining from defs_w
    if !defs.is_null() {
        let ndefs = unsafe { pyre_object::w_tuple_len(defs) };
        let start = ndefs - defs_to_load;
        let mut i = nargs;
        for j in start..ndefs {
            if let Some(val) = unsafe { pyre_object::w_tuple_getitem(defs, j as i64) } {
                new_frame.locals_w_mut()[i] = val;
            }
            i += 1;
        }
    }

    frame.dropvalues(dropvalues);
    new_frame.fix_array_ptrs();

    // function.py:231 — return new_frame.run(self.name, self.qualname)
    if new_frame._is_generator_or_coroutine() {
        match new_frame.into_generator() {
            Ok(v) => v,
            Err(e) => {
                crate::call::set_call_error(e);
                pyre_object::PY_NULL
            }
        }
    } else {
        let _caller_locals_root = FrameLocalsRoot::new(frame);
        let _callee_locals_root = FrameLocalsRoot::new(&mut new_frame);
        let eval_fn = crate::call::get_eval_fn();
        match eval_fn(&mut new_frame) {
            Ok(v) => v,
            Err(e) => {
                crate::call::set_call_error(e);
                pyre_object::PY_NULL
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_function_create() {
        crate::test_hooks::install_hash_hook();
        // Function.code now stores a Code-level wrapper (PyCode).
        let raw_code = 0xDEAD_BEEF as *const ();
        let w_code = crate::w_code_new(raw_code);
        let w_globals = pyre_object::w_module_dict_new();
        let obj = function_new(w_code as *const (), "myfunc".to_string(), w_globals);
        unsafe {
            assert!(is_function(obj));
            assert!(!is_int(obj));
            assert_eq!(function_get_code(obj), w_code as *const ());
            assert_eq!(function_get_name(obj), "myfunc");
            assert_eq!(function_get_globals_obj(obj), w_globals);
            assert!(function_get_closure(obj).is_null());
        }
    }

    #[test]
    fn test_function_field_offsets() {
        assert_eq!(FUNCTION_CODE_OFFSET, 16); // after PyObject { ob_type(8) + w_class(8) }
        assert_eq!(FUNCTION_NAME_OFFSET, 32); // after code(8) + can_change_code(1) + padding(7)
        assert_eq!(FUNCTION_CLOSURE_OFFSET, 40); // after name
    }

    /// Guard against drift between the constant colocated with
    /// `Function` and the id that `pyre-jit/src/eval.rs` asserts at
    /// JitDriver init. Mirror of the W_INT/W_FLOAT/BuiltinCode
    /// trip-wire tests.
    #[test]
    fn function_gc_type_id_matches_descr() {
        assert_eq!(FUNCTION_GC_TYPE_ID, 14);
        assert_eq!(
            <Function as pyre_object::lltype::GcType>::type_id(),
            FUNCTION_GC_TYPE_ID
        );
        assert_eq!(
            <Function as pyre_object::lltype::GcType>::SIZE,
            std::mem::size_of::<Function>()
        );
    }

    /// `FUNCTION_GC_PTR_OFFSETS` must list every inline
    /// `PyObjectRef`-shaped field the GC traces (`code` is `*const ()`
    /// but points at a `[ob: PyObject, ...]`-prefixed Code object so
    /// the walker can interpret it as a typed reference). If a new GC
    /// field is added to `Function` (or one of these fields is removed)
    /// the array has to follow — this test makes the change a
    /// compile-time failure rather than a silent traversal gap.
    #[test]
    fn function_gc_ptr_offsets_cover_inline_pyobjectref_fields() {
        assert_eq!(
            FUNCTION_GC_PTR_OFFSETS,
            [
                std::mem::offset_of!(Function, code),
                std::mem::offset_of!(Function, closure),
                std::mem::offset_of!(Function, defs_w),
                std::mem::offset_of!(Function, w_kw_defs),
                std::mem::offset_of!(Function, w_module),
                std::mem::offset_of!(Function, w_func_globals_obj),
                std::mem::offset_of!(Function, w_builtins),
                std::mem::offset_of!(Function, w_ann),
                std::mem::offset_of!(Function, w_annotate),
                std::mem::offset_of!(Function, w_func_dict),
                std::mem::offset_of!(Function, w_typeparams),
                std::mem::offset_of!(Function, w_doc),
                std::mem::offset_of!(Function, w_qualname),
                std::mem::offset_of!(Function, w_objclass),
                std::mem::offset_of!(Function, w_text_signature),
                std::mem::offset_of!(Function, w_moduleobj),
            ]
        );
    }
}
