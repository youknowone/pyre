//! `aiter` / `anext` builtins — PyPy: pypy/module/__builtin__/app_operation.py
//!
//! The two names are pure app-level functions in PyPy's MixedModule.  The
//! builtins module dict is populated inside `ExecutionContext::new`, before
//! the execution context is registered, so `appleveldef_install` cannot run
//! at builtins-setup time (there is no execution context yet).  Instead the
//! two names are registered as interp-level builtin functions that resolve
//! the app-level implementations lazily on first call, when an execution
//! context exists — the same lazy-resolution shape `reduce_protocol` uses.

use pyre_object::PyObjectRef;

use crate::PyResult;

/// `aiter` / `anext` ported verbatim from app_operation.py:36-81.  The
/// `_NOT_PROVIDED` sentinel stays in the intermediate namespace that both
/// functions retain as their `__globals__`, so only the two public names
/// need to be surfaced.
const ASYNC_OP_SRC: &str = r#"
_NOT_PROVIDED = object()

def aiter(obj):
    """aiter(async_iterable) -> async_iterator
    aiter(async_callable, sentinel) -> async_iterator
    Like the iter() builtin but for async iterables and callables.
    """
    typ = type(obj)
    try:
        meth = typ.__aiter__
    except AttributeError:
        raise TypeError(f"'{type(obj).__name__}' object is not an async iterable")
    ait = meth(obj)
    if not hasattr(ait, '__anext__'):
        raise TypeError(f"aiter() returned not an async iterator of type '{type(ait).__name__}'")
    return ait

def anext(iterator, default=_NOT_PROVIDED):
    """anext(async_iterator[, default])
    Return the next item from the async iterator.
    If default is given and the iterator is exhausted,
    it is returned instead of raising StopAsyncIteration.
    """
    typ = type(iterator)

    try:
        __anext__ = typ.__anext__
    except AttributeError:
        raise TypeError(f"'{type(iterator).__name__}' object is not an async iterator")

    if default is _NOT_PROVIDED:
        return __anext__(iterator)

    async def anext_impl():
        try:
            return await __anext__(iterator)
        except StopAsyncIteration:
            return default

    return anext_impl()
"#;

const AITER: usize = 0;
const ANEXT: usize = 1;

thread_local! {
    static HANDLES: std::cell::OnceCell<[PyObjectRef; 2]> = const { std::cell::OnceCell::new() };
}

/// Resolve (and cache) the two app-level handles.  Executes `ASYNC_OP_SRC`
/// into its own fresh module globals and stores the `aiter` / `anext`
/// function objects; both retain that namespace as their `__globals__`,
/// keeping the `_NOT_PROVIDED` sentinel reachable.
fn handle(which: usize) -> PyObjectRef {
    HANDLES.with(|cell| {
        cell.get_or_init(|| {
            let ctx = crate::call::getexecutioncontext();
            if ctx.is_null() {
                panic!("async_operation: no execution context");
            }
            let _roots = pyre_object::gc_roots::push_roots();
            let save_point = pyre_object::gc_roots::shadow_stack_len();
            let w_app_globals = pyre_object::dictmultiobject::w_module_dict_new();
            pyre_object::gc_roots::pin_root(w_app_globals);
            crate::importing::appleveldef_install(
                pyre_object::gc_roots::shadow_stack_get(save_point),
                ASYNC_OP_SRC,
                "app_operation.py",
                &["aiter", "anext"],
            );
            let w_app_globals = pyre_object::gc_roots::shadow_stack_get(save_point);
            let get = |name: &str| {
                unsafe { pyre_object::w_dict_getitem_str(w_app_globals, name) }
                    .unwrap_or_else(|| panic!("async_operation: `{name}` not bound"))
            };
            [get("aiter"), get("anext")]
        })[which]
    })
}

/// `aiter(obj)` — delegates to the app-level `aiter`, whose `def aiter(obj)`
/// signature enforces the single-argument arity.
pub fn builtin_aiter(args: &[PyObjectRef]) -> PyResult {
    crate::call::call_function_impl_result(handle(AITER), args)
}

/// `anext(iterator[, default])` — delegates to the app-level `anext`.
pub fn builtin_anext(args: &[PyObjectRef]) -> PyResult {
    crate::call::call_function_impl_result(handle(ANEXT), args)
}
