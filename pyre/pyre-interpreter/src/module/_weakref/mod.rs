//! _weakref module — PyPy: pypy/module/_weakref/moduledef.py.
//!
//! ```python
//! interpleveldefs = {
//!     'ref': 'interp__weakref.W_Weakref',
//!     'getweakrefcount': 'interp__weakref.getweakrefcount',
//!     'getweakrefs': 'interp__weakref.getweakrefs',
//!     'ReferenceType': 'interp__weakref.W_Weakref',
//!     'ProxyType': 'interp__weakref.W_Proxy',
//!     'CallableProxyType': 'interp__weakref.W_CallableProxy',
//!     'proxy': 'interp__weakref.proxy',
//! }
//! ```
//!
//! `_remove_dead_weakref` is CPython-only and is stubbed as a no-op for
//! cleanup-driven users like weakref.py's WeakValueDictionary.

pub mod interp_weakref;

crate::py_module! {
    "_weakref",
    interpleveldefs: {
        "ref" => interp_weakref::weakref_type(),
        "ReferenceType" => interp_weakref::weakref_type(),
        "ProxyType" => interp_weakref::proxy_type(),
        "CallableProxyType" => interp_weakref::callable_proxy_type(),
        "proxy" => crate::make_builtin_function("proxy", interp_weakref::proxy),
        "getweakrefcount" => crate::make_module_builtin_function_with_arity(
            "getweakrefcount", interp_weakref::getweakrefcount, 1),
        "getweakrefs" => crate::make_module_builtin_function_with_arity(
            "getweakrefs", interp_weakref::getweakrefs, 1),
        "_remove_dead_weakref" => crate::make_module_builtin_function_with_arity(
            "_remove_dead_weakref", |_| Ok(pyre_object::w_none()), 2),
    }
}
