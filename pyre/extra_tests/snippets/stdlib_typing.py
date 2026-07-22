from collections.abc import Awaitable, Callable
from types import GenericAlias
from typing import ClassVar, Protocol, TypeVar

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
