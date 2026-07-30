"""CPython 3.14 GenericAlias list-parameter and attribute parity."""

import copy
import pickle
import weakref
from array import array
from collections import deque
from contextvars import ContextVar, Token
from ctypes import Array as CTypesArray
from os import DirEntry
from string.templatelib import Interpolation, Template
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

dir_entry_alias = DirEntry[T]
assert dir_entry_alias.__origin__ is DirEntry
assert dir_entry_alias.__args__ == (T,)
assert copy.copy(dir_entry_alias) == dir_entry_alias
assert copy.deepcopy(dir_entry_alias) == dir_entry_alias

ctypes_array_alias = CTypesArray[T]
assert ctypes_array_alias.__origin__ is CTypesArray
assert ctypes_array_alias.__args__ == (T,)
assert copy.copy(ctypes_array_alias) == ctypes_array_alias
assert copy.deepcopy(ctypes_array_alias) == ctypes_array_alias

for template_type in (Template, Interpolation):
    template_alias = template_type[T]
    assert template_alias.__origin__ is template_type
    assert template_alias.__args__ == (T,)
    assert copy.copy(template_alias) == template_alias
    assert copy.deepcopy(template_alias) == template_alias

for context_type in (ContextVar, Token):
    context_alias = context_type[T]
    assert context_alias.__origin__ is context_type
    assert context_alias.__args__ == (T,)
    assert copy.copy(context_alias) == context_alias
    assert copy.deepcopy(context_alias) == context_alias

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

print("OK")
