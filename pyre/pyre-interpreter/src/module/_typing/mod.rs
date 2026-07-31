//! _typing module — the type-parameter runtime objects (TypeVar, ParamSpec,
//! TypeVarTuple, ParamSpecArgs/Kwargs, TypeAliasType, Generic, NoDefault)
//! that `typing.py` imports.  These are app-level Python (mirroring
//! Objects/typevarobject.c) that delegates the substitution logic back to
//! `typing.py`, while `Union` is bound to the builtin `types.UnionType`.
//! `_idfunc` is interp-level: the non-binding identity used as `NewType.__call__`.

crate::py_module! {
    "_typing",
    appleveldefs: {
        "_typing_app.py" => [
            "TypeVar", "ParamSpec", "TypeVarTuple",
            "ParamSpecArgs", "ParamSpecKwargs", "TypeAliasType",
            "Generic", "Union", "NoDefault",
            "_intrinsic_typevar", "_intrinsic_paramspec",
            "_intrinsic_typevartuple", "_intrinsic_typevar_with_bound",
            "_intrinsic_typevar_with_constraints",
            "_intrinsic_set_typeparam_default",
            "_intrinsic_subscript_generic", "_intrinsic_typealias",
        ],
    },
    functions: {
        // `NewType.__call__` (typing.py) — a non-binding identity; a binding
        // function would receive the NewType instance as an implicit self and
        // return it instead of the argument.
        "_idfunc" / 1 = |args| Ok(args[0]),
    },
}
