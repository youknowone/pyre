//! `__pypy__` module — PyPy: pypy/module/__pypy__/
//!
//! Pyre exposes the small slice of the `__pypy__` surface that the
//! PyPy-flavored stdlib needs.  `pickle.py` imports `identity_dict`
//! (an identity-keyed memo dict) and `builders.BytesBuilder` in one
//! shared `try` block; both must resolve for the optimized path to
//! activate, so both are provided here as app-level classes.
//!
//! `collections.OrderedDict` (`_collections/app_odict.py`) imports
//! `reversed_dict`, `move_to_end`, and `objects_in_repr` — the dict-order
//! primitives PyPy keeps interp-level in `interp_dict.py` / `interp_magic.py`.
//! Pyre matches that placement (the `functions:` block below), so they are
//! non-binding `BuiltinFunction`s; only `identity_dict` stays an app-level
//! class.
//!
//! `PickleBuffer` (`interp_buffer.py W_PickleBuffer`) is exposed here as
//! an interp-level class; `pickle.py` re-exports it and the `_pickle`
//! accelerator serializes it in-band or out-of-band under protocol 5.

pub mod interp_buffer;

pub use interp_buffer::W_PickleBuffer;

/// `interp_magic.get_contextvar_context` — read the value from the live
/// PyPy-style ExecutionContext, returning None before the first Context is
/// installed.
fn get_contextvar_context(_: &[pyre_object::PyObjectRef]) -> crate::PyResult {
    let ec = crate::call::getexecutioncontext();
    if ec.is_null() {
        return Ok(pyre_object::w_none());
    }
    let context = unsafe { (*ec).contextvar_context };
    if context.is_null() {
        Ok(pyre_object::w_none())
    } else {
        Ok(context)
    }
}

/// `interp_magic.set_contextvar_context` — replace the current
/// ExecutionContext's PEP 567 Context slot.
fn set_contextvar_context(args: &[pyre_object::PyObjectRef]) -> crate::PyResult {
    let Some(&context) = args.first() else {
        return Err(crate::PyError::type_error(
            "set_contextvar_context() missing required argument",
        ));
    };
    let ec = crate::call::getexecutioncontext() as *mut crate::PyExecutionContext;
    if ec.is_null() {
        return Err(crate::PyError::runtime_error(
            "no current execution context",
        ));
    }
    unsafe { (*ec).contextvar_context = context };
    Ok(pyre_object::w_none())
}

/// `interp_dict.py:43-45 / 79-81` `isinstance(w_obj, W_DictMultiObject)`:
/// resolve the backing of an exact dict, module dict, or dict subclass,
/// rejecting a read-only `mappingproxy` (a `W_Root`, not a `W_DictMultiObject`)
/// and any non-dict.  `what` names the primitive for the `TypeError`.
fn dict_backing_or_type_error(
    obj: pyre_object::PyObjectRef,
    what: &str,
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    let is_proxy = unsafe { pyre_object::is_dict_proxy(obj) };
    if !is_proxy {
        let backing = crate::type_methods::resolve_dict_backing(obj);
        if !backing.is_null() {
            return Ok(backing);
        }
    }
    Err(crate::PyError::type_error(format!(
        "{what}() argument must be a dict"
    )))
}

/// `interp_dict.py:34 reversed_dict` — `W_DictMultiObject.descr_reversed`
/// (`dictmultiobject.py:207`): a reverse iterator over the dict keys, resolved
/// through the backing dict so a subclass `__reversed__` is bypassed.
fn reversed_dict(args: &[pyre_object::PyObjectRef]) -> crate::PyResult {
    let backing = dict_backing_or_type_error(args[0], "reversed_dict")?;
    Ok(
        pyre_object::dictmultiobject::w_dict_view_reverse_iterator_new(
            backing,
            pyre_object::dictmultiobject::DictViewKind::Keys,
        ),
    )
}

/// `interp_dict.py:69 move_to_end` — `W_DictMultiObject.nondescr_move_to_end`
/// (`dictmultiobject.py:221`): move an existing key to the back (`last`, the
/// default) or front of the insertion order.  `@unwrap_spec(last=bool)`: `last`
/// may be supplied positionally or by keyword and is coerced by truthiness.
fn move_to_end(args: &[pyre_object::PyObjectRef]) -> crate::PyResult {
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    crate::builtins::kwarg_reject_unknown(kwargs, &["last"], "move_to_end")?;
    let (d, key) = match positional {
        [d, key] | [d, key, _] => (*d, *key),
        _ => {
            return Err(crate::PyError::type_error(
                "move_to_end() takes 2 or 3 positional arguments",
            ));
        }
    };
    let last =
        match crate::builtins::bind_pos_or_kw(positional, kwargs, 2, "last", "move_to_end", 3)? {
            Some(w) => crate::baseobjspace::is_true(w)?,
            None => true,
        };
    let backing = dict_backing_or_type_error(d, "move_to_end")?;
    if crate::baseobjspace::dict_move_to_end(backing, key, last)? {
        Ok(pyre_object::w_none())
    } else {
        Err(crate::PyError::key_error_with_key(key))
    }
}

/// `interp_magic.py:19 objects_in_repr` — `space.get_objects_in_repr()`
/// (`objspace.py:134`): the execution-context-owned identity dict of objects
/// currently being `repr()`'d, lazily built and cached on the EC.  `identity_dict`
/// stays an app-level class, so the instance is constructed by calling that type.
///
/// `get_objects_in_repr` builds `W_IdentityDict(self)` directly (`objspace.py:136`),
/// so the recursion guard cannot be swapped through module monkeypatching.  pyre
/// mirrors that by constructing from the canonical type captured at module init
/// (`CANONICAL_IDENTITY_DICT_KEY`), not the publicly reassignable
/// `__pypy__.identity_dict` attribute.
fn objects_in_repr(_: &[pyre_object::PyObjectRef]) -> crate::PyResult {
    let ec = crate::call::getexecutioncontext() as *mut crate::PyExecutionContext;
    if ec.is_null() {
        return Err(crate::PyError::runtime_error(
            "no current execution context",
        ));
    }
    let cached = unsafe { (*ec).py_repr };
    if !cached.is_null() {
        return Ok(cached);
    }
    let module = crate::importing::check_sys_modules("__pypy__")
        .ok_or_else(|| crate::PyError::runtime_error("__pypy__ module not loaded"))?;
    let ty =
        crate::baseobjspace::getattr_str(module, CANONICAL_IDENTITY_DICT_KEY).map_err(|_| {
            crate::PyError::runtime_error("__pypy__: canonical identity_dict unavailable")
        })?;
    let inst = crate::call::call_function_impl_result(ty, &[])?;
    // Stored immediately (no intervening allocation); thereafter forwarded by
    // `ExecutionContext::walk_builtin_roots`.
    unsafe { (*ec).py_repr = inst };
    Ok(inst)
}

/// Module-dict key holding the `identity_dict` type captured at module init.
/// Not a valid Python identifier, so it cannot be reached — or rewritten —
/// through `__pypy__.identity_dict` attribute access; the module dict is
/// Box-immortal, so the entry roots the type for the lifetime of the process.
const CANONICAL_IDENTITY_DICT_KEY: &str = "@objects_in_repr_identity_dict";

crate::py_module! {
    "__pypy__",
    // `PickleBuffer` wraps a bytes-like object for proto-5 out-of-band
    // buffers; `identity_dict` keys a memo by object identity (id(key))
    // so the Pickler can memoize unhashable containers.
    interpleveldefs: {
        "PickleBuffer" => interp_buffer::picklebuffer_type_object(),
    },
    appleveldefs: {
        "identity_dict_app.py" => ["identity_dict"],
    },
    functions: {
        "get_contextvar_context" / 0 = get_contextvar_context,
        "set_contextvar_context" / 1 = set_contextvar_context,
        "reversed_dict" / 1 = reversed_dict,
        "move_to_end" / * = move_to_end,
        "objects_in_repr" / 0 = objects_in_repr,
    },
    extra_init: |ns| {
        // Mark as a package so `from __pypy__.builders import ...`
        // treats `__pypy__` as a package with submodules.
        crate::module_ns_store(ns, "__path__", pyre_object::w_list_new(vec![]));
        // Snapshot the canonical `identity_dict` type before any app code can
        // reassign `__pypy__.identity_dict`, keyed so attribute access cannot
        // reach it.  `objects_in_repr` builds its recursion guard from this
        // snapshot, matching PyPy's direct `W_IdentityDict` construction.
        if let Some(ty) = crate::module_ns_get(ns, "identity_dict") {
            crate::module_ns_store(ns, CANONICAL_IDENTITY_DICT_KEY, ty);
        }
    }
}

/// `__pypy__.builders` submodule — exposes the string/bytes builders.
pub mod builders {
    crate::py_module! {
        "__pypy__.builders",
        // BytesBuilder is the append-only byte buffer pickle.py writes
        // frames into; StringBuilder is its text analogue.
        appleveldefs: {
            "builders_app.py" => ["BytesBuilder", "StringBuilder"],
        }
    }
}
