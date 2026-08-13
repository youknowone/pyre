"""App-level _typing — the type-parameter runtime objects typing.py imports.

These mirror the objects in Objects/typevarobject.c: TypeVar, ParamSpec,
TypeVarTuple, ParamSpecArgs/Kwargs, TypeAliasType, Generic, plus the
NoDefault sentinel.  The heavy substitution and
class-getitem logic lives in typing.py; these objects delegate to its
module-level helpers (_typevar_subst, _paramspec_subst, _generic_class_getitem,
...), exactly as the C objects call back into the typing module.
"""

# CPython exposes these runtime classes as members of `typing`, and their
# repr/pickle/substitution semantics depend on that public owner. Mixed-module
# applevel code executes in a private globals dict, so declare it explicitly
# instead of inheriting that dict's default `builtins` module name.
__name__ = "typing"

import sys
from types import GenericAlias, UnionType as Union


def _caller_module():
    # Equivalent to sys._getframe(2).f_globals['__name__']: frame 0 is this
    # helper, frame 1 the constructor, frame 2 the user that called it.
    try:
        return sys._getframe(2).f_globals.get('__name__')
    except (AttributeError, ValueError):
        return None


def _evaluate_typeparam(thunk):
    # Call a PEP 695 bound / constraints thunk emitted by the compiler. A 3.14
    # thunk takes an annotation `format` argument (annotationlib.Format.VALUE
    # == 1); accept a zero-argument thunk as well.
    code = getattr(thunk, '__code__', None)
    if code is not None and code.co_argcount >= 1:
        return thunk(1)
    return thunk()


class _NoDefaultMeta(type):
    def __new__(mcls, name, bases, namespace):
        if any(isinstance(base, _NoDefaultMeta) for base in bases):
            raise TypeError("type 'typing.NoDefault' is not an acceptable base type")
        return super().__new__(mcls, name, bases, namespace)

    def __setattr__(cls, name, value):
        raise TypeError("cannot set attributes of immutable type 'typing.NoDefault'")


_no_default_instance = None


class _NoDefaultType(metaclass=_NoDefaultMeta):
    """Type of the typing.NoDefault sentinel."""

    __slots__ = ()

    def __new__(cls):
        global _no_default_instance
        if _no_default_instance is None:
            _no_default_instance = super().__new__(cls)
        return _no_default_instance

    def __repr__(self):
        return "typing.NoDefault"

    def __reduce__(self):
        return "NoDefault"


NoDefault = _NoDefaultType()


_MISSING = object()


def _typing_type_repr(value):
    """CPython 3.14 `_Py_typing_type_repr`, for constant evaluators."""
    if value is Ellipsis:
        return "..."
    if value is type(None):
        return "None"
    if hasattr(value, "__origin__") and hasattr(value, "__args__"):
        return repr(value)
    qualname = getattr(value, "__qualname__", None)
    module = getattr(value, "__module__", None)
    if qualname is None or module is None:
        return repr(value)
    if module == "builtins":
        return qualname
    return f"{module}.{qualname}"


def _index(value) -> int:
    # `operator.index`, written out because this module loads before the
    # pure-Python `operator` is importable.  An evaluator's format argument is
    # a `Format` member, so it reaches here through the integer index
    # protocol: a float or a string is a TypeError rather than a format that
    # silently compares unequal to every member.
    if isinstance(value, int):
        result = value
    else:
        try:
            index = type(value).__index__
        except AttributeError:
            raise TypeError(
                f"{type(value).__name__!r} object cannot be interpreted as an integer"
            ) from None
        result = index(value)
        if not isinstance(result, int):
            raise TypeError(f"__index__ returned non-int (type {type(result).__name__})")
    # `int(result)` would route a subclass through its own `__int__`, which may
    # answer something other than the value the subclass stores. Narrowing to
    # an exact `int` is what `PyNumber_Index` does with `_PyLong_Copy`.
    return int.__index__(result)


def _immutable_const_evaluator_error(name):
    return TypeError(
        f"cannot set '{name}' attribute of immutable type "
        "'_typing._ConstEvaluator'"
    )


class _ConstEvaluatorMeta(type):
    def __setattr__(cls, name, value):
        raise _immutable_const_evaluator_error(name)

    def __delattr__(cls, name):
        # `type_setattro` handles both writes, so an immutable type refuses a
        # deletion with the wording it uses for an assignment. Without this,
        # `del type(evaluator).__call__` removes call support from every
        # constant evaluator in the process.
        raise _immutable_const_evaluator_error(name)


class _ConstEvaluator(metaclass=_ConstEvaluatorMeta):
    """CPython 3.14 `constevaluatorobject`."""

    __slots__ = ("_value",)

    def __new__(cls):
        raise TypeError("cannot create '_typing._ConstEvaluator' instances")

    def __call__(self, format, /):
        # The parameter keeps CPython's name; the converted value gets its own
        # local so the builtin stays reachable in this scope.
        format_value = _index(format)
        if format_value == 4:  # annotationlib.Format.STRING
            if isinstance(self._value, tuple):
                return "(" + ", ".join(map(_typing_type_repr, self._value)) + ")"
            return _typing_type_repr(self._value)
        return self._value

    def __repr__(self):
        return f"<constevaluator {self._value!r}>"


def _const_evaluator(value):
    evaluator = object.__new__(_ConstEvaluator)
    evaluator._value = value
    return evaluator


def _variance_prefix(infer_variance, covariant, contravariant):
    if infer_variance:
        return ''
    if covariant:
        return '+'
    if contravariant:
        return '-'
    return '~'


class TypeVar:
    """Type variable — PEP 484 / PEP 695."""

    def __init__(self, name, *constraints, bound=None, default=NoDefault,
                 covariant=False, contravariant=False, infer_variance=False):
        self.__name__ = name
        if covariant and contravariant:
            raise ValueError("Bivariant types are not supported.")
        if infer_variance and (covariant or contravariant):
            raise ValueError("Variance cannot be specified with infer_variance.")
        self.__covariant__ = bool(covariant)
        self.__contravariant__ = bool(contravariant)
        self.__infer_variance__ = bool(infer_variance)
        self._default_value = default
        self._evaluate_default = None
        if constraints and bound is not None:
            raise TypeError("Constraints cannot be combined with bound=...")
        if len(constraints) == 1:
            raise TypeError("A single constraint is not allowed")
        import typing
        self._constraints = tuple(
            typing._type_check(
                constraint,
                f"TypeVar(name, constraint, ...). Constraints must be types. Got {constraint!r}.",
            )
            for constraint in constraints
        )
        self._evaluate_constraints = None
        self._bound = (
            None
            if bound is None
            else typing._type_check(bound, "Bound must be a type.")
        )
        self._evaluate_bound = None
        self.__module__ = _caller_module()

    @classmethod
    def _make(cls, name, *, evaluate_bound=None, evaluate_constraints=None):
        # Lazy construction for the TYPEVAR_WITH_BOUND and
        # TYPEVAR_WITH_CONSTRAINTS intrinsics. The bound / constraints arrive as
        # thunks the compiler defers so they may reference names bound later in
        # the enclosing scope; they are evaluated on first `__bound__` /
        # `__constraints__` access and cached (Objects/typevarobject.c).
        self = cls.__new__(cls)
        self.__name__ = name
        self.__covariant__ = False
        self.__contravariant__ = False
        # CPython 3.14 `_Py_make_typevar`: type parameters created by the
        # compiler infer variance, unlike an ordinary `TypeVar(...)` call.
        self.__infer_variance__ = True
        self._default_value = NoDefault
        self._evaluate_default = None
        self._constraints = _MISSING if evaluate_constraints is not None else ()
        self._evaluate_constraints = evaluate_constraints
        self._bound = _MISSING if evaluate_bound is not None else None
        self._evaluate_bound = evaluate_bound
        self.__module__ = _caller_module()
        return self

    @property
    def __bound__(self):
        if self._bound is _MISSING:
            self._bound = _evaluate_typeparam(self._evaluate_bound)
        return self._bound

    @property
    def __constraints__(self):
        if self._constraints is _MISSING:
            self._constraints = tuple(_evaluate_typeparam(self._evaluate_constraints))
        return self._constraints

    @property
    def __default__(self):
        if self._default_value is _MISSING:
            self._default_value = _evaluate_typeparam(self._evaluate_default)
        return self._default_value

    @property
    def evaluate_bound(self):
        if self._evaluate_bound is not None:
            return self._evaluate_bound
        if self._bound is not None:
            return _const_evaluator(self._bound)
        return None

    @property
    def evaluate_constraints(self):
        if self._evaluate_constraints is not None:
            return self._evaluate_constraints
        if self._constraints:
            return _const_evaluator(self._constraints)
        return None

    @property
    def evaluate_default(self):
        if self._evaluate_default is not None:
            return self._evaluate_default
        return _const_evaluator(self._default_value)

    def __typing_subst__(self, arg):
        import typing
        return typing._typevar_subst(self, arg)

    def __typing_prepare_subst__(self, alias, args):
        params = alias.__parameters__
        try:
            index = list(params).index(self)
        except ValueError:
            return args
        if len(args) == index and self.has_default():
            args = list(args)
            args.append(self.__default__)
            args = tuple(args)
        return args

    def has_default(self):
        return self._evaluate_default is not None or self._default_value is not NoDefault

    def __reduce__(self):
        return self.__name__

    def __mro_entries__(self, bases):
        raise TypeError("Cannot subclass an instance of TypeVar")

    def __or__(self, other):
        import typing
        return Union[self, typing._type_convert(other)]

    def __ror__(self, other):
        import typing
        return Union[typing._type_convert(other), self]

    def __repr__(self):
        return _variance_prefix(self.__infer_variance__, self.__covariant__,
                                self.__contravariant__) + self.__name__

    def __init_subclass__(cls, **kwargs):
        raise TypeError("type 'typing.TypeVar' is not an acceptable base type")


class ParamSpec:
    """Parameter specification variable — PEP 612."""

    def __init__(self, name, *, bound=None, default=NoDefault,
                 covariant=False, contravariant=False, infer_variance=False):
        self.__name__ = name
        if covariant and contravariant:
            raise ValueError("Bivariant types are not supported.")
        if infer_variance and (covariant or contravariant):
            raise ValueError("Variance cannot be specified with infer_variance.")
        self.__covariant__ = bool(covariant)
        self.__contravariant__ = bool(contravariant)
        self.__infer_variance__ = bool(infer_variance)
        self._default_value = default
        self._evaluate_default = None
        self.__bound__ = bound
        self.__module__ = _caller_module()

    @property
    def args(self):
        return ParamSpecArgs(self)

    @property
    def kwargs(self):
        return ParamSpecKwargs(self)

    def __typing_subst__(self, arg):
        import typing
        return typing._paramspec_subst(self, arg)

    def __typing_prepare_subst__(self, alias, args):
        import typing
        return typing._paramspec_prepare_subst(self, alias, args)

    def has_default(self):
        return self._evaluate_default is not None or self._default_value is not NoDefault

    @property
    def __default__(self):
        if self._default_value is _MISSING:
            self._default_value = _evaluate_typeparam(self._evaluate_default)
        return self._default_value

    @property
    def evaluate_default(self):
        if self._evaluate_default is not None:
            return self._evaluate_default
        return _const_evaluator(self._default_value)

    def __reduce__(self):
        return self.__name__

    def __mro_entries__(self, bases):
        raise TypeError("Cannot subclass an instance of ParamSpec")

    def __or__(self, other):
        return Union[self, other]

    def __ror__(self, other):
        return Union[other, self]

    def __init_subclass__(cls, **kwargs):
        raise TypeError("type 'typing.ParamSpec' is not an acceptable base type")

    def __repr__(self):
        return _variance_prefix(self.__infer_variance__, self.__covariant__,
                                self.__contravariant__) + self.__name__


class ParamSpecArgs:
    """The args of a ParamSpec, e.g. P.args."""

    def __init__(self, origin):
        self.__origin__ = origin

    def __repr__(self):
        return f"{self.__origin__.__name__}.args"

    def __eq__(self, other):
        if not isinstance(other, ParamSpecArgs):
            return NotImplemented
        return self.__origin__ == other.__origin__

    def __hash__(self):
        return hash((self.__origin__, "args"))

    def __mro_entries__(self, bases):
        raise TypeError("Cannot subclass an instance of ParamSpecArgs")

    def __init_subclass__(cls, **kwargs):
        raise TypeError("type 'typing.ParamSpecArgs' is not an acceptable base type")


class ParamSpecKwargs:
    """The kwargs of a ParamSpec, e.g. P.kwargs."""

    def __init__(self, origin):
        self.__origin__ = origin

    def __repr__(self):
        return f"{self.__origin__.__name__}.kwargs"

    def __eq__(self, other):
        if not isinstance(other, ParamSpecKwargs):
            return NotImplemented
        return self.__origin__ == other.__origin__

    def __hash__(self):
        return hash((self.__origin__, "kwargs"))

    def __mro_entries__(self, bases):
        raise TypeError("Cannot subclass an instance of ParamSpecKwargs")

    def __init_subclass__(cls, **kwargs):
        raise TypeError("type 'typing.ParamSpecKwargs' is not an acceptable base type")


class TypeVarTuple:
    """Type variable tuple — PEP 646."""

    def __init__(self, name, *, default=NoDefault):
        self.__name__ = name
        self._default_value = default
        self._evaluate_default = None
        self.__module__ = _caller_module()

    def __iter__(self):
        import typing
        yield typing.Unpack[self]

    def __typing_subst__(self, arg):
        raise TypeError("Substitution of bare TypeVarTuple is not supported")

    def __typing_prepare_subst__(self, alias, args):
        import typing
        return typing._typevartuple_prepare_subst(self, alias, args)

    def has_default(self):
        return self._evaluate_default is not None or self._default_value is not NoDefault

    @property
    def __default__(self):
        if self._default_value is _MISSING:
            self._default_value = _evaluate_typeparam(self._evaluate_default)
        return self._default_value

    @property
    def evaluate_default(self):
        if self._evaluate_default is not None:
            return self._evaluate_default
        return _const_evaluator(self._default_value)

    def __reduce__(self):
        return self.__name__

    def __mro_entries__(self, bases):
        raise TypeError("Cannot subclass an instance of TypeVarTuple")

    def __repr__(self):
        return self.__name__

    def __init_subclass__(cls, **kwargs):
        raise TypeError("type 'typing.TypeVarTuple' is not an acceptable base type")


class TypeAliasType:
    """A PEP 695 ``type X = ...`` alias."""

    def __init__(self, name, value, *, type_params=(), _evaluate_value=None):
        self.__name__ = name
        self._value = value
        self._evaluate_value = _evaluate_value
        self.__type_params__ = tuple(type_params)
        self.__module__ = _caller_module()

    @property
    def __value__(self):
        if self._value is _MISSING:
            self._value = _evaluate_typeparam(self._evaluate_value)
        return self._value

    @property
    def evaluate_value(self):
        if self._evaluate_value is not None:
            return self._evaluate_value
        return _const_evaluator(self._value)

    @property
    def __parameters__(self):
        return self.__type_params__

    def __getitem__(self, args):
        if not self.__type_params__:
            raise TypeError("Only generic type aliases are subscriptable")
        if not isinstance(args, tuple):
            args = (args,)
        return GenericAlias(self, args)

    def __or__(self, other):
        return Union[self, other]

    def __ror__(self, other):
        return Union[other, self]

    def __repr__(self):
        return self.__name__


class Generic:
    """Abstract base class for generic types — PEP 484."""

    __slots__ = ()

    def __class_getitem__(cls, params):
        import typing
        return typing._generic_class_getitem(cls, params)

    def __init_subclass__(cls, *args, **kwargs):
        import typing
        return typing._generic_init_subclass(cls, *args, **kwargs)


# ── PEP 695 intrinsic helpers ────────────────────────────────────────────
# Called by the bytecode intrinsics (INTRINSIC_TYPEVAR, INTRINSIC_PARAMSPEC,
# INTRINSIC_SUBSCRIPT_GENERIC, ...) emitted for `class C[T]:`, `def f[T]()`,
# and `type X = ...`.  Keeping the construction here lets the interpreter side
# call a single positional helper per intrinsic.

def _intrinsic_typevar(name):
    # CPython 3.14 `_Py_make_typevar` passes `infer_variance=true`.
    return TypeVar(name, infer_variance=True)


def _intrinsic_paramspec(name):
    # CPython 3.14 `_Py_make_paramspec` passes `infer_variance=true`.
    return ParamSpec(name, infer_variance=True)


def _intrinsic_typevartuple(name):
    return TypeVarTuple(name)


def _intrinsic_typevar_with_bound(name, evaluate_bound):
    return TypeVar._make(name, evaluate_bound=evaluate_bound)


def _intrinsic_typevar_with_constraints(name, evaluate_constraints):
    return TypeVar._make(name, evaluate_constraints=evaluate_constraints)


def _intrinsic_set_typeparam_default(typeparam, default):
    # CPython 3.14 `_Py_set_typeparam_default` stores the evaluator, not its
    # result.  `__default__` evaluates and caches it on first access.
    typeparam._default_value = _MISSING
    typeparam._evaluate_default = default
    return typeparam


def _intrinsic_subscript_generic(params):
    import typing
    if not isinstance(params, tuple):
        params = (params,)
    return typing._GenericAlias(typing.Generic, params)


def _intrinsic_typealias(args):
    # args is the (name, type_params, value) tuple the TYPEALIAS intrinsic
    # builds; `value` is the lazy evaluator (or the value itself).
    name, type_params, value = args
    if type_params is None:
        type_params = ()
    alias = TypeAliasType(
        name, _MISSING, type_params=type_params, _evaluate_value=value
    )
    # The constructor is called through this intrinsic helper, so its ordinary
    # two-frame caller lookup sees the helper's private ``typing`` globals.
    # CPython records the namespace executing the TYPEALIAS intrinsic instead.
    try:
        alias.__module__ = sys._getframe(1).f_globals.get('__name__')
    except (AttributeError, ValueError):
        alias.__module__ = None
    return alias
