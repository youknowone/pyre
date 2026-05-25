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
        "ref"               => interp_weakref::weakref_type(),
        "ReferenceType"     => interp_weakref::weakref_type(),
        "ProxyType"         => interp_weakref::proxy_type(),
        "CallableProxyType" => interp_weakref::callable_proxy_type(),
    },
    functions: {
        "proxy" / * = interp_weakref::proxy,
    },
    module_functions: {
        "getweakrefcount"      / 1 = interp_weakref::getweakrefcount,
        "getweakrefs"          / 1 = interp_weakref::getweakrefs,
        "_remove_dead_weakref" / 2 = |_| Ok(pyre_object::w_none()),
    },
}
