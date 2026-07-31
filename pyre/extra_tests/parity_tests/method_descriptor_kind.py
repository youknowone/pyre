"""Parity test: a `tp_methods` entry is a `method_descriptor`.

`type_ready_fill_dict` hands every `PyMethodDef` in a static type's
`tp_methods` to `PyDescr_NewMethod`, so the namespace entry is a
`method_descriptor`; `descrobject.c method_get` then hands it to
`PyCMethod_New`, so an instance access yields a
`builtin_function_or_method` carrying the receiver as `m_self`.  The slot
half of the same sweep (`add_operators` -> `PyDescr_NewWrapper`) produces
`wrapper_descriptor` instead, which is why the dunders below are excluded.

Pinned here because pyre reaches the same shape from the other side: the
namespaces are built as plain function carriers and retagged once the
namespace is complete (`typedef.rs stamp_method_owners`), gated on the
`gateway::is_slot_wrapper` classification.  A name that moves between the
two halves of that table silently changes both the descriptor kind and the
receiver-error wording, and only a differential check catches it.
"""

# A representative `tp_methods` entry from each layout family.  `bool` has
# none of its own — it inherits `int`'s — while `set` and `frozenset` each
# carry their own table, so both of those appear.
ORDINARY = [
    (list, "append"),
    (list, "count"),
    (list, "index"),
    (dict, "get"),
    (dict, "setdefault"),
    (dict, "items"),
    (tuple, "index"),
    (tuple, "count"),
    (int, "bit_length"),
    (int, "to_bytes"),
    (int, "conjugate"),
    (float, "hex"),
    (float, "is_integer"),
    (complex, "conjugate"),
    (str, "upper"),
    (str, "split"),
    (str, "encode"),
    (bytes, "hex"),
    (bytes, "decode"),
    (bytearray, "append"),
    (bytearray, "capitalize"),
    (set, "add"),
    (set, "union"),
    (frozenset, "union"),
    (range, "count"),
    (object, "__sizeof__"),
    (object, "__dir__"),
    (type, "mro"),
]

for owner, name in ORDINARY:
    descr = owner.__dict__[name]
    assert type(descr).__name__ == "method_descriptor", (owner, name, type(descr))
    assert descr.__name__ == name, (owner, name, descr.__name__)
    assert descr.__qualname__ == f"{owner.__name__}.{name}", (owner, name)
    assert descr.__objclass__ is owner, (owner, name)
    assert repr(descr) == f"<method '{name}' of '{owner.__name__}' objects>"


# The slot half stays a wrapper descriptor.  `__contains__` and
# `__getitem__` are per-type: the mapping and sequence protocols fill
# different slots, so `dict.__getitem__` is a `tp_methods` entry while
# `tuple.__getitem__` is a slot.
#
# The in-place number slots and the `tp_as_async` trio are here because they
# are reachable on only a couple of types, so a table that forgets them stays
# green on the core types and misclassifies exactly those.
import types as _types
import weakref as _weakref


class _Referent:
    pass


_referent = _Referent()
_ProxyType = type(_weakref.proxy(_referent))

SLOTS = [
    (list, "__len__"),
    (tuple, "__getitem__"),
    (int, "__add__"),
    (int, "__repr__"),
    (str, "__contains__"),
    (object, "__init__"),
    (object, "__setattr__"),
    (type(x for x in ()), "__del__"),
    (_ProxyType, "__ifloordiv__"),
    (_ProxyType, "__ilshift__"),
    (_ProxyType, "__imod__"),
    (_ProxyType, "__ipow__"),
    (_ProxyType, "__irshift__"),
    (_ProxyType, "__itruediv__"),
    (_types.CoroutineType, "__await__"),
    (_types.AsyncGeneratorType, "__aiter__"),
    (_types.AsyncGeneratorType, "__anext__"),
]

for owner, name in SLOTS:
    descr = owner.__dict__[name]
    assert type(descr).__name__ != "method_descriptor", (owner, name, type(descr))

# The two protocol splits, stated as such rather than inferred.
assert type(dict.__dict__["__getitem__"]).__name__ == "method_descriptor"
assert type(list.__dict__["__getitem__"]).__name__ == "method_descriptor"
assert type(tuple.__dict__["__getitem__"]).__name__ != "method_descriptor"
assert type(dict.__dict__["__contains__"]).__name__ == "method_descriptor"
assert type(str.__dict__["__contains__"]).__name__ != "method_descriptor"


# `FrameLocalsProxy` is the third type on that split: `framelocalsproxy_methods`
# carries both names, so both are `tp_methods` entries even though the type
# also fills the mapping slots.
def _frame_locals_proxy_type():
    _unused = 1
    import sys

    return type(sys._getframe().f_locals)


_FLP = _frame_locals_proxy_type()
for _name in ("__contains__", "__getitem__", "keys", "get"):
    _d = _FLP.__dict__[_name]
    assert type(_d).__name__ == "method_descriptor", (_name, type(_d))
    assert _d.__qualname__ == f"FrameLocalsProxy.{_name}", _name
    assert _d.__objclass__ is _FLP, _name
    assert repr(_d) == f"<method '{_name}' of 'FrameLocalsProxy' objects>"


# Instance access binds to a builtin carrier, not a `method`.
BOUND = [([], "append"), ({}, "get"), ((), "index"), (1, "bit_length"),
         ("a", "upper"), (b"a", "hex"), (bytearray(b"a"), "append"),
         (set(), "add"), (1.0, "hex")]

for receiver, name in BOUND:
    bound = getattr(receiver, name)
    assert type(bound).__name__ == "builtin_function_or_method", (name, type(bound))
    assert type(bound) is type(len), name
    assert bound.__self__ is receiver, name
    assert bound.__qualname__ == f"{type(receiver).__name__}.{name}", name
    assert repr(bound).startswith(
        f"<built-in method {name} of {type(receiver).__name__} object at 0x"
    ), repr(bound)


# `meth_richcompare` / `meth_hash` compare `m_self` and the method entry, so
# two binds of the same method on the same receiver are equal and hash equal
# even though each bind mints a fresh carrier.
_l = []
assert _l.append == _l.append
assert hash(_l.append) == hash(_l.append)
assert [].append != [].append
assert _l.append != _l.count
assert _l.append != len


# A subclass instance binds through the same descriptor, and the qualified
# name follows the receiver's own type.
class _MyList(list):
    pass


_sub = _MyList()
assert type(_sub.append).__name__ == "builtin_function_or_method"
assert _sub.append.__self__ is _sub
assert _sub.append.__qualname__ == "_MyList.append"
_sub.append(3)
assert _sub == [3]


# The unbound descriptor still checks its receiver.
try:
    dict.__dict__["get"](1, "x")
except TypeError as exc:
    assert str(exc) == (
        "descriptor 'get' for 'dict' objects doesn't apply to a 'int' object"
    ), exc
else:
    raise AssertionError("a foreign receiver must be rejected")

# Explicit `__get__` binds the same way attribute access does.
_d = {"k": 1}
assert dict.__dict__["get"].__get__(_d, dict)("k") == 1
assert type(dict.__dict__["get"].__get__(_d, dict)).__name__ == (
    "builtin_function_or_method"
)
# Class access without an instance hands back the bare descriptor.
assert dict.__dict__["get"].__get__(None, dict) is dict.__dict__["get"]

# Calling through the descriptor and through the bound carrier agree.
assert dict.__dict__["get"](_d, "k") == 1
assert _d.get("k") == 1
assert list.__dict__["append"].__get__([], list).__name__ == "append"

print("OK")
