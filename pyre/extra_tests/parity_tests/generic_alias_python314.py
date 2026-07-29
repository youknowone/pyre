"""CPython 3.14 GenericAlias list-parameter and attribute parity."""

import copy
from types import GenericAlias
from typing import TypeVar


class Origin:
    __copy__ = object()
    __deepcopy__ = object()


alias = GenericAlias(Origin, int)
assert "__copy__" not in dir(list[int])
assert "__deepcopy__" not in dir(list[int])
for name in ("__bases__", "__copy__", "__deepcopy__"):
    try:
        getattr(alias, name)
    except AttributeError:
        pass
    else:
        raise AssertionError(f"GenericAlias exposed blocked attribute {name}")

copied = copy.copy(alias)
deepcopied = copy.deepcopy(alias)
assert copied == alias
assert deepcopied == alias

T = TypeVar("T")
U = TypeVar("U")
nested = GenericAlias(list, ([T, [U]],))
assert nested.__parameters__ == (T, U)
specialized = nested[str, int]
assert specialized.__args__ == ([str, [int]],)
assert specialized.__parameters__ == ()
assert repr(GenericAlias(Origin, [int, str])).endswith("Origin[[int, str]]")
