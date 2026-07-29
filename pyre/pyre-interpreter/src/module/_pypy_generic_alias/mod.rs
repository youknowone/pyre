//! `_pypy_generic_alias` — PyPy's app-level GenericAlias/UnionType module.
//!
//! The line-by-line port lives in `crate::_pypy_generic_alias`; expose its
//! canonical type objects under the private module names that PyPy app-level
//! modules import.

crate::py_module! {
    "_pypy_generic_alias",
    interpleveldefs: {
        "GenericAlias" => crate::typedef::gettypeobject(
            &pyre_object::GENERIC_ALIAS_TYPE,
        ),
        "UnionType" => crate::typedef::gettypeobject(
            &pyre_object::UNION_TYPE,
        ),
    },
}
