"""CPython 3.14 GenericAlias list-parameter and attribute parity."""

import copy
import pickle
import weakref
from array import array
from collections import deque
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
deque_alias = deque[T]
assert deque_alias.__origin__ is deque
assert deque_alias.__args__ == (T,)
assert deque_alias([1, 2]) == deque([1, 2])
assert copy.copy(deque_alias) == deque_alias
assert copy.deepcopy(deque_alias) == deque_alias

array_alias = array[T]
assert array_alias.__origin__ is array
assert array_alias.__args__ == (T,)
assert array_alias("i", [1, 2]) == array("i", [1, 2])
assert copy.copy(array_alias) == array_alias
assert copy.deepcopy(array_alias) == array_alias

weakref_alias = weakref.ReferenceType[T]
assert weakref_alias.__origin__ is weakref.ReferenceType
assert weakref_alias.__args__ == (T,)
assert weakref.ref[T] == weakref_alias
assert copy.copy(weakref_alias) == weakref_alias
assert copy.deepcopy(weakref_alias) == weakref_alias


async def coroutine_sample():
    pass


coroutine = coroutine_sample()
coroutine_type = type(coroutine)
coroutine.close()
coroutine_alias = coroutine_type[T]
assert coroutine_alias.__origin__ is coroutine_type
assert coroutine_alias.__args__ == (T,)
assert copy.copy(coroutine_alias) == coroutine_alias
assert copy.deepcopy(coroutine_alias) == coroutine_alias

nested = GenericAlias(list, ([T, [U]],))
assert nested.__parameters__ == (T, U)
specialized = nested[str, int]
assert specialized.__args__ == ([str, [int]],)
assert specialized.__parameters__ == ()
assert repr(GenericAlias(Origin, [int, str])).endswith("Origin[[int, str]]")

starred = (*tuple[int, str],)[0]
for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
    restored = pickle.loads(pickle.dumps(starred, protocol))
    assert restored == starred
