from collections.abc import Awaitable, Callable
from types import GenericAlias
from typing import ClassVar, ParamSpec, Protocol, TypeVar

T = TypeVar("T")


def abort_signal_handler(
    fn: Callable[[], Awaitable[T]], on_abort: Callable[[], None] | None = None
) -> T:
    pass


# Ensure PEP 604 unions work with typing.Callable aliases.
TracebackFilter = bool | Callable[[int], int]


# Test that Union/Optional in function parameter annotations work correctly.
# This tests that annotation scopes can access global implicit symbols (like Union)
# that are imported at module level but not explicitly bound in the function scope.
# Regression test for: rich
from typing import Optional, Union


def function_with_union_param(x: Optional[Union[int, str]] = None) -> None:
    pass


class ClassWithUnionParams:
    def __init__(
        self,
        color: Optional[Union[str, int]] = None,
        bold: Optional[bool] = None,
    ) -> None:
        pass

    def method(self, value: Union[int, float]) -> Union[str, bytes]:
        return str(value)


# PEP 649 class annotation scopes use LOAD_FROM_DICT_OR_GLOBALS.  A miss in
# the class namespace must continue through module globals and builtins.
class AnnotatedCoordinate:
    x: int


if hasattr(AnnotatedCoordinate.__annotate__, "__jit__"):
    # Force the annotation thunk through the LOAD_FROM_DICT_OR_GLOBALS JIT
    # residual before materializing its cache.
    AnnotatedCoordinate.__annotate__.__jit__()
assert AnnotatedCoordinate.__annotations__ == {"x": int}


class CoordinateProtocol(Protocol):
    x: int


assert CoordinateProtocol.__protocol_attrs__ == {"x"}


# types.FunctionType must retain a dict-subclass globals object.  CPython's
# annotationlib clones PEP 649 annotation thunks this way and relies on
# __missing__ to synthesize ForwardRef values for unresolved names.
import types


class StringifyingGlobals(dict):
    def __missing__(self, key):
        return f"missing:{key}"


def resolve_from_globals():
    return unresolved_annotation_name


stringifying_globals = StringifyingGlobals(resolve_from_globals.__globals__)
cloned_resolver = types.FunctionType(resolve_from_globals.__code__, stringifying_globals)
assert cloned_resolver.__globals__ is stringifying_globals
assert cloned_resolver() == "missing:unresolved_annotation_name"


# Generated dataclass methods copy lazy class annotations.  A VALUE lookup
# may raise NameError; the function getter must leave that error intact so
# annotationlib can retry in FORWARDREF format.
import annotationlib
from dataclasses import dataclass


@dataclass
class DeferredDataclass:
    value: DeferredValue


deferred_init_annotations = annotationlib.get_annotations(
    DeferredDataclass.__init__, format=annotationlib.Format.FORWARDREF
)
assert isinstance(deferred_init_annotations["value"], annotationlib.ForwardRef)
assert deferred_init_annotations["value"].__forward_arg__ == "DeferredValue"
assert DeferredDataclass.__doc__ == "DeferredDataclass(value: DeferredValue)"


# A compiler-generated class annotation thunk closes over the live class
# namespace.  Names assigned before or after the annotation function is made
# must remain visible through that one dictionary.
class ClassLocalAnnotation:
    LocalAlias = ClassVar[int]
    value: LocalAlias


assert ClassLocalAnnotation.__annotations__["value"] == ClassVar[int]


class LazyAnnotatedBase:
    value: int


class LazyAnnotatedChild(LazyAnnotatedBase):
    pass


# CPython 3.14 type annotation slots are owned by the class.  A subclass does
# not inherit its base's thunk, and replacing a thunk invalidates a previously
# materialized annotations cache.
assert LazyAnnotatedChild.__annotate__ is None
assert LazyAnnotatedBase.__annotations__ == {"value": int}
LazyAnnotatedBase.__annotate__ = lambda _: {}
assert LazyAnnotatedBase.__annotations__ == {}

# Clearing __annotate__ preserves an already materialized cache, while a new
# callable replaces both the compiler-facing and public slots.
LazyAnnotatedBase.__annotate__ = lambda format: {"cached": format}
assert LazyAnnotatedBase.__annotations__ == {"cached": 1}
LazyAnnotatedBase.__annotate__ = None
assert LazyAnnotatedBase.__annotations__ == {"cached": 1}
LazyAnnotatedBase.__annotate__ = lambda format: {"new": format}
assert LazyAnnotatedBase.__annotate__(1) == {"new": 1}
assert LazyAnnotatedBase.__annotations__ == {"new": 1}


class ExplicitAnnotations:
    __annotations__ = {"old": int}


ExplicitAnnotations.__annotations__ = {"new": str}
assert ExplicitAnnotations.__annotations__ == {"new": str}
assert ExplicitAnnotations.__dict__["__annotations__"] == {"new": str}
del ExplicitAnnotations.__annotations__
assert ExplicitAnnotations.__annotations__ == {}


class ExplicitAnnotate:
    def __annotate__(format):
        return {"old": format}


old_explicit_annotate = ExplicitAnnotate.__annotate__
ExplicitAnnotate.__annotate__ = lambda format: {"new": format}
assert ExplicitAnnotate.__annotate__ is old_explicit_annotate
assert ExplicitAnnotate.__annotations__ == {"old": 1}


class NonCallableExplicitAnnotate:
    __annotate__ = 42


assert NonCallableExplicitAnnotate.__annotate__ == 42
assert NonCallableExplicitAnnotate.__annotations__ == {}


class ResetAnnotationsThenAnnotate:
    pass


ResetAnnotationsThenAnnotate.__annotations__ = {"old": int}
ResetAnnotationsThenAnnotate.__annotate__ = lambda format: {"new": format}
assert ResetAnnotationsThenAnnotate.__annotations__ == {"new": 1}
del ResetAnnotationsThenAnnotate.__annotations__
ResetAnnotationsThenAnnotate.__annotate__ = lambda format: {"again": format}
assert ResetAnnotationsThenAnnotate.__annotations__ == {"again": 1}


class AnnotationReader:
    def __set_name__(self, owner, name):
        owner.InjectedBySetName = int
        self.seen = owner.__annotations__


class SetNameAnnotations:
    value: InjectedBySetName
    reader = AnnotationReader()


assert SetNameAnnotations.reader.seen == {"value": int}


class AnnotationMeta(type):
    pass


class MetaSetNameAnnotations(metaclass=AnnotationMeta):
    value: InjectedBySetName
    reader = AnnotationReader()


assert MetaSetNameAnnotations.reader.seen == {"value": int}

try:
    LazyAnnotatedBase.__annotate__ = 42
except TypeError:
    pass
else:
    raise AssertionError("type.__annotate__ accepted a non-callable")

try:
    del LazyAnnotatedBase.__annotate__
except TypeError:
    pass
else:
    raise AssertionError("type.__annotate__ was deletable")
assert object.__type_params__ == ()
assert ClassLocalAnnotation.__type_params__ == ()


class GenericAliasSubclass(GenericAlias):
    pass


generic_alias_subclass = GenericAliasSubclass(list, int)
assert type(generic_alias_subclass) is GenericAliasSubclass
assert generic_alias_subclass.__origin__ is list
assert generic_alias_subclass.__args__ == (int,)


# A constant bound is evaluated by `_typing._ConstEvaluator`, whose format
# argument goes through the integer index protocol.
const_evaluator = TypeVar("ConstBound", bound=int).evaluate_bound
assert type(const_evaluator).__name__ == "_ConstEvaluator"
assert const_evaluator(annotationlib.Format.STRING) == "int"
assert const_evaluator(annotationlib.Format.VALUE) is int


class IndexedFormat:
    def __index__(self):
        return 4


assert const_evaluator(IndexedFormat()) == "int"


class UnequalInt(int):
    def __eq__(self, other):
        return False

    def __hash__(self):
        return 0


# The argument is normalized to an exact `int`, so a subclass whose `__eq__`
# refuses every comparison still selects the STRING branch by its value.
assert const_evaluator(UnequalInt(4)) == "int"
assert const_evaluator(UnequalInt(1)) is int


class NonIntIndex:
    def __index__(self):
        return "4"


try:
    const_evaluator(NonIntIndex())
except TypeError as error:
    assert str(error) == "__index__ returned non-int (type str)", error
else:
    raise AssertionError("_ConstEvaluator accepted a non-int __index__ result")

try:
    const_evaluator(1.0)
except TypeError as error:
    assert str(error) == "'float' object cannot be interpreted as an integer", error
else:
    raise AssertionError("_ConstEvaluator accepted a float format")

immutable_message = (
    "cannot set '__call__' attribute of immutable type '_typing._ConstEvaluator'"
)
try:
    del type(const_evaluator).__call__
except TypeError as error:
    assert str(error) == immutable_message, error
else:
    raise AssertionError("_ConstEvaluator.__call__ was deletable")

try:
    type(const_evaluator).__call__ = None
except TypeError as error:
    assert str(error) == immutable_message, error
else:
    raise AssertionError("_ConstEvaluator.__call__ was assignable")

# A refused write leaves every constant evaluator in the process callable.
assert const_evaluator(annotationlib.Format.STRING) == "int"

# `bound=None` means two different things across the two parameter kinds:
# `typevar_new_impl` drops a None bound before `type_check`, so it stays None,
# while `paramspec_new_impl` has no such shortcut and hands every bound —
# including the one its signature defaults to — to `typing._type_check`, which
# maps None to NoneType.
assert TypeVar("T").__bound__ is None
assert TypeVar("T", bound=None).__bound__ is None
assert ParamSpec("P").__bound__ is type(None)
assert ParamSpec("P", bound=None).__bound__ is type(None)

# `paramspecargs` / `paramspeckwargs` keep the origin in a `Py_READONLY`
# member, so neither the ordinary write nor the one that steps around a
# `__setattr__` guard reaches it, and the private storage stays out of `dir()`.
_P = ParamSpec("P")
for _view in (_P.args, _P.kwargs):
    assert _view.__origin__ is _P
    for _write in (
        lambda: setattr(_view, "__origin__", 1),
        lambda: object.__setattr__(_view, "__origin__", 1),
        lambda: delattr(_view, "__origin__"),
        lambda: object.__delattr__(_view, "__origin__"),
    ):
        try:
            _write()
        except AttributeError as exc:
            assert str(exc) == "readonly attribute", str(exc)
        else:
            raise AssertionError("__origin__ accepted a write")
    assert _view.__origin__ is _P
    assert "__origin__" in dir(_view)
    assert not [name for name in dir(_view) if name == "_origin"]

# A compiler-created parameter is allocated with no module of its own, so it
# reports the class attribute rather than the module that declares it.
def _identity[T](x: T) -> T:
    return x


assert _identity.__type_params__[0].__module__ == "typing"
assert TypeVar("T").__module__ == __name__
