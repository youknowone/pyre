"""App-level _typing — the type-parameter runtime objects typing.py imports.

These mirror the objects in Objects/typevarobject.c: TypeVar, ParamSpec,
TypeVarTuple, ParamSpecArgs/Kwargs, TypeAliasType, Generic, plus the
NoDefault sentinel and the _idfunc helper.  The heavy substitution and
class-getitem logic lives in typing.py; these objects delegate to its
module-level helpers (_typevar_subst, _paramspec_subst, _generic_class_getitem,
...), exactly as the C objects call back into the typing module.
"""

import sys
from types import GenericAlias, UnionType as Union


def _idfunc(*args, **kwargs):
    return args[0]


def _caller_module():
    # Equivalent to sys._getframe(2).f_globals['__name__']: frame 0 is this
    # helper, frame 1 the constructor, frame 2 the user that called it.
    try:
        return sys._getframe(2).f_globals.get('__name__', '__main__')
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


class _NoDefaultType:
    """Type of the typing.NoDefault sentinel."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "typing.NoDefault"

    def __reduce__(self):
        return "NoDefault"


NoDefault = _NoDefaultType()


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
        self.__default__ = default
        if constraints and bound is not None:
            raise TypeError("Constraints cannot be combined with bound=...")
        if len(constraints) == 1:
            raise TypeError("A single constraint is not allowed")
        self._constraints = tuple(constraints)
        self._evaluate_constraints = None
        self._bound = bound
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
        self.__infer_variance__ = False
        self.__default__ = NoDefault
        self._constraints = ()
        self._evaluate_constraints = evaluate_constraints
        self._bound = None
        self._evaluate_bound = evaluate_bound
        self.__module__ = _caller_module()
        return self

    @property
    def __bound__(self):
        if self._evaluate_bound is not None:
            self._bound = _evaluate_typeparam(self._evaluate_bound)
            self._evaluate_bound = None
        return self._bound

    @property
    def __constraints__(self):
        if self._evaluate_constraints is not None:
            self._constraints = tuple(_evaluate_typeparam(self._evaluate_constraints))
            self._evaluate_constraints = None
        return self._constraints

    def __typing_subst__(self, arg):
        import typing
        return typing._typevar_subst(self, arg)

    def __typing_prepare_subst__(self, alias, args):
        params = alias.__parameters__
        try:
            index = list(params).index(self)
        except ValueError:
            return args
        if len(args) <= index and self.has_default():
            args = list(args)
            while len(args) <= index:
                args.append(self.__default__)
            args = tuple(args)
        return args

    def has_default(self):
        return self.__default__ is not NoDefault

    def __reduce__(self):
        return self.__name__

    def __mro_entries__(self, bases):
        raise TypeError("Cannot subclass an instance of TypeVar")

    def __or__(self, other):
        return Union[self, other]

    def __ror__(self, other):
        return Union[other, self]

    def __repr__(self):
        return _variance_prefix(self.__infer_variance__, self.__covariant__,
                                self.__contravariant__) + self.__name__


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
        self.__default__ = default
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
        return self.__default__ is not NoDefault

    def __reduce__(self):
        return self.__name__

    def __mro_entries__(self, bases):
        raise TypeError("Cannot subclass an instance of ParamSpec")

    def __or__(self, other):
        return Union[self, other]

    def __ror__(self, other):
        return Union[other, self]

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


class TypeVarTuple:
    """Type variable tuple — PEP 646."""

    def __init__(self, name, *, default=NoDefault):
        self.__name__ = name
        self.__default__ = default
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
        return self.__default__ is not NoDefault

    def __reduce__(self):
        return self.__name__

    def __mro_entries__(self, bases):
        raise TypeError("Cannot subclass an instance of TypeVarTuple")

    def __repr__(self):
        return self.__name__


class TypeAliasType:
    """A PEP 695 ``type X = ...`` alias."""

    def __init__(self, name, value, *, type_params=()):
        self.__name__ = name
        self.__value__ = value
        self.__type_params__ = tuple(type_params)
        self.__module__ = _caller_module()

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
    return TypeVar(name)


def _intrinsic_paramspec(name):
    return ParamSpec(name)


def _intrinsic_typevartuple(name):
    return TypeVarTuple(name)


def _intrinsic_typevar_with_bound(name, evaluate_bound):
    return TypeVar._make(name, evaluate_bound=evaluate_bound)


def _intrinsic_typevar_with_constraints(name, evaluate_constraints):
    return TypeVar._make(name, evaluate_constraints=evaluate_constraints)


def _intrinsic_set_typeparam_default(typeparam, default):
    typeparam.__default__ = default
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
    return TypeAliasType(name, value, type_params=type_params)
