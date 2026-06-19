//! _typing module — the type-parameter runtime objects (TypeVar, ParamSpec,
//! TypeVarTuple, ParamSpecArgs/Kwargs, TypeAliasType, Generic, NoDefault,
//! _idfunc) that `typing.py` imports.  CPython implements these in C
//! (Objects/typevarobject.c); here they are app-level Python that delegates
//! the substitution logic back to `typing.py`, while `Union` is bound to the
//! builtin `types.UnionType`.

crate::py_module! {
    "_typing",
    appleveldefs: {
        "_typing_app.py" => [
            "_idfunc", "TypeVar", "ParamSpec", "TypeVarTuple",
            "ParamSpecArgs", "ParamSpecKwargs", "TypeAliasType",
            "Generic", "Union", "NoDefault",
            "_intrinsic_typevar", "_intrinsic_paramspec",
            "_intrinsic_typevartuple", "_intrinsic_typevar_with_bound",
            "_intrinsic_typevar_with_constraints",
            "_intrinsic_set_typeparam_default",
            "_intrinsic_subscript_generic", "_intrinsic_typealias",
        ],
    },
}
