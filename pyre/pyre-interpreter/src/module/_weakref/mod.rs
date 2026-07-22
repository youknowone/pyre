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
//! CPython exposes `_remove_dead_weakref`; PyPy implements the same operation
//! in `app_weakref.py` through its atomic `delitem_if_value_is` helper.

pub mod interp__weakref;

crate::py_module! {
    "_weakref",
    interpleveldefs: {
        "ref"               => interp__weakref::weakref_type(),
        "ReferenceType"     => interp__weakref::weakref_type(),
        "ProxyType"         => interp__weakref::proxy_type(),
        "CallableProxyType" => interp__weakref::callable_proxy_type(),
    },
    functions: {
        "proxy" / * = interp__weakref::proxy,
    },
    module_functions: {
        "getweakrefcount"      / 1 = interp__weakref::getweakrefcount,
        "getweakrefs"          / 1 = interp__weakref::getweakrefs,
        "_remove_dead_weakref" / 2 = interp__weakref::remove_dead_weakref,
    },
}
