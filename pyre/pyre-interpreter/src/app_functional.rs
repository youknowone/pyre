//! `sorted` builtin — PyPy: pypy/module/__builtin__/app_functional.py
//!
//! `pypy/module/__builtin__/moduledef.py` installs `sorted` as an
//! appleveldef (`'sorted': 'app_functional.sorted'`).  The Python body
//! is `list(iterable)` then `list.sort`, which the JIT traces as
//! bytecode and `CALL_ASSEMBLER`s into `list__do_extend_from_iterable`.
//! The interp-level `builtin_sorted` body cannot be descended: its
//! generated wrapper reaches hundreds of un-lowered helpers after an
//! effect (`PYRE_FBW_INLINE_DIAG` `un-lowered helper call in body`).
//!
//! `mixedmodule.py MixedModule._load_lazily` then wraps that Function as
//! `BuiltinFunction` so the published name has no `__get__`.  `type(sorted)`
//! is `builtin_function_or_method` here (PyPy: `builtin_function`); both
//! are the mixed-module builtin type, not `function`.  The wrapper still
//! carries the app-level PyCode (`sorted.__code__` exists), which
//! `resolve_inlinable_callee` inlines.
//!
//! The builtins dict is filled inside `ExecutionContext::new` before the
//! context is registered, so `appleveldef_install` cannot run then
//! (same constraint `async_operation` documents for `aiter`/`anext`).
//! A trampoline occupies the name until the first EC registration, which
//! publishes the BuiltinFunction MixedModule would have stored.  That
//! publish deletes the trampoline slot first: a second `write_cell` over
//! the trampoline wraps it in `ObjectMutableCell`, and LOAD_GLOBAL then
//! keeps a live getfield so the walker refuses to inline the app-level
//! body (`callable_guard_op.is_constant()`).  MixedModule._load_lazily
//! writes the BuiltinFunction once (`write_cell` `StoreBare`).
//! `HANDLE` is the off-GC stand-in for that module-dict slot until then
//! (`async_operation` `HANDLES`); MixedModule's cache is the dict itself.

use pyre_object::PyObjectRef;

use crate::PyResult;

const APP_FUNCTIONAL_SRC: &str = include_str!("module/__builtin__/app_functional.py");

static HANDLE: parking_lot::Mutex<usize> = parking_lot::Mutex::new(0);

/// `mixedmodule.py MixedModule._load_lazily`: an applevel Function published
/// in a mixed-module dict becomes `BuiltinFunction`.  `function.py
/// BuiltinFunction.__init__` allocates a new object and copies `code`,
/// `w_func_globals`, `defs_w`, `closure`, `name`, `w_doc`, `w_func_dict`,
/// `w_module`, `w_kw_defs`, `w_text_signature`.  `MixedModule._cleanup_`
/// forces every lazy loader at translation time, so the wrapper is a
/// prebuilt constant and `rgc.can_move` is false.  Allocate the sibling
/// with `lltype.malloc_typed_stable` (old-gen, non-moving) so LOAD_GLOBAL
/// of `sorted` folds to ConstPtr the same way.
fn wrap_as_builtin_function(func: PyObjectRef) -> PyObjectRef {
    unsafe {
        if !pyre_object::py_type_check(func, &crate::FUNCTION_TYPE) {
            return func;
        }
        let _roots = pyre_object::gc_roots::push_roots();
        let src_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(func);
        let src = pyre_object::gc_roots::shadow_stack_get(src_slot) as *const crate::Function;
        let mut payload = std::ptr::read(src);
        payload.ob.ob_type = &crate::BUILTIN_FUNCTION_TYPE as *const _;
        payload.ob.w_class = pyre_object::pyobject::get_instantiate(&crate::BUILTIN_FUNCTION_TYPE);
        payload.can_change_code = false;
        payload.mutate_slots = std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());
        pyre_object::lltype::malloc_typed_stable(payload) as PyObjectRef
    }
}

fn app_sorted() -> PyResult {
    let cached = *HANDLE.lock();
    if cached != 0 {
        return Ok(cached as PyObjectRef);
    }

    let ctx = crate::call::getexecutioncontext();
    if ctx.is_null() {
        panic!("app_functional: no execution context");
    }
    let _roots = pyre_object::gc_roots::push_roots();
    let save_point = pyre_object::gc_roots::shadow_stack_len();
    let w_app_globals = pyre_object::dictmultiobject::w_module_dict_new();
    let _ = pyre_object::gc_roots::pin_root(w_app_globals);
    crate::importing::appleveldef_install(
        pyre_object::gc_roots::shadow_stack_get(save_point),
        APP_FUNCTIONAL_SRC,
        "app_functional.py",
        "builtins",
        &["sorted"],
    )?;
    let w_app_globals = pyre_object::gc_roots::shadow_stack_get(save_point);
    let func = wrap_as_builtin_function(
        unsafe { pyre_object::w_dict_getitem_str(w_app_globals, "sorted") }
            .unwrap_or_else(|| panic!("app_functional: `sorted` not bound")),
    );
    let mut handle = HANDLE.lock();
    if *handle == 0 {
        *handle = func as usize;
    }
    Ok(*handle as PyObjectRef)
}

/// Publish the MixedModule `BuiltinFunction` into the live builtins dict so
/// subsequent `LOAD_GLOBAL` sees it.  Called from the first EC registration,
/// after `getexecutioncontext` is live.
pub(crate) fn install_applevel_builtins() {
    let Ok(func) = app_sorted() else {
        return;
    };
    let ctx = crate::call::getexecutioncontext();
    if ctx.is_null() {
        return;
    }
    let builtins = unsafe { (*ctx).get_builtin_dict() };
    if builtins.is_null() {
        return;
    }
    // `typeobject.py write_cell`: overwriting the trampoline wraps the
    // slot in `ObjectMutableCell`.  Delete first so the insert is
    // `StoreBare` — MixedModule._load_lazily writes the applevel
    // BuiltinFunction once, and LOAD_GLOBAL then folds to ConstPtr.
    let _roots = pyre_object::gc_roots::push_roots();
    let func_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(func);
    crate::module_ns_delete(builtins, "sorted");
    let func = pyre_object::gc_roots::shadow_stack_get(func_slot);
    crate::module_ns_store(builtins, "sorted", func);
    // `func` is the old-gen BuiltinFunction; `w_str_new` may allocate in
    // the nursery.  Reload `func` after that alloc (it cannot move) and
    // let `fset_func_text_signature` run the old-to-young write barrier.
    let sig = pyre_object::w_str_new("($module, iterable, /, *, key=None, reverse=False)");
    let func = pyre_object::gc_roots::shadow_stack_get(func_slot);
    unsafe {
        crate::function::fset_func_text_signature(func, sig);
    }
}

pub(crate) fn walk_handle_roots(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    let mut handle = HANDLE.lock();
    if *handle != 0 {
        unsafe { visitor(&mut *(&mut *handle as *mut usize as *mut majit_ir::GcRef)) };
    }
}

/// Trampoline occupying `builtins.sorted` until
/// [`install_applevel_builtins`] replaces it.  First call also publishes
/// the app-level function, then delegates.
pub fn builtin_sorted(args: &[PyObjectRef]) -> PyResult {
    install_applevel_builtins();
    crate::call::call_function_impl_result(app_sorted()?, args)
}

#[cfg(not(target_arch = "wasm32"))]
#[linkme::distributed_slice(crate::gateway::BUILTIN_WRAPPER_DESCRIPTORS)]
#[allow(non_upper_case_globals)]
static __majit_builtin_wrapper_target_app_sorted: crate::gateway::BuiltinWrapperDescriptor =
    crate::gateway::BuiltinWrapperDescriptor {
        path: concat!(module_path!(), "::", stringify!(builtin_sorted)),
        func: builtin_sorted,
    };
