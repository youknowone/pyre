//! _typing module — the type-parameter runtime objects (TypeVar, ParamSpec,
//! TypeVarTuple, ParamSpecArgs/Kwargs, TypeAliasType, Generic, NoDefault)
//! that `typing.py` imports.  These are app-level Python (mirroring
//! Objects/typevarobject.c) that delegates the substitution logic back to
//! `typing.py`, while `Union` is bound to the builtin `types.UnionType`.
//! `_idfunc` is interp-level: the identity used as `NewType.__call__`.

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
        // PyPy `lib_pypy/_typing.py:19` spells this `def _idfunc(_, x):
        // return x`: `DescrOperation.get_and_call_args` passes the NewType
        // receiver explicitly for every Function subclass, including a module
        // builtin stored as `NewType.__call__`.  CPython 3.14 exposes the same
        // helper as a non-descriptor METH_O builtin, so retain its direct
        // `_typing._idfunc(x)` surface as the one-argument case.
        "_idfunc" / * = |args| match args {
            [value] | [_, value] => Ok(*value),
            _ => Err(crate::PyError::type_error(
                "_typing._idfunc() takes exactly one argument",
            )),
        },
    },
}
