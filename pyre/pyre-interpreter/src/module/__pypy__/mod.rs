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
//! primitives PyPy keeps in `interp_dict.py` / `interp_magic.py`.  Pyre
//! provides them app-level alongside `identity_dict`.
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

crate::py_module! {
    "__pypy__",
    // `PickleBuffer` wraps a bytes-like object for proto-5 out-of-band
    // buffers; `identity_dict` keys a memo by object identity (id(key))
    // so the Pickler can memoize unhashable containers.
    interpleveldefs: {
        "PickleBuffer" => interp_buffer::picklebuffer_type_object(),
    },
    appleveldefs: {
        "identity_dict_app.py" =>
            ["identity_dict", "reversed_dict", "move_to_end", "objects_in_repr"],
    },
    functions: {
        "get_contextvar_context" / 0 = get_contextvar_context,
        "set_contextvar_context" / 1 = set_contextvar_context,
    },
    extra_init: |ns| {
        // Mark as a package so `from __pypy__.builders import ...`
        // treats `__pypy__` as a package with submodules.
        crate::module_ns_store(ns, "__path__", pyre_object::w_list_new(vec![]));
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
