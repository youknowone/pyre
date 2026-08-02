"""Parity test: a `METH_CLASS` entry is a `classmethod_descriptor`.

`type_ready_fill_dict` hands a `PyMethodDef` carrying `METH_CLASS` to
`PyDescr_NewClassMethod`, so `dict.__dict__['fromkeys']` is a
`classmethod_descriptor` rather than the `classmethod` a decorated Python
function produces.  The two halves diverge in what they validate and what
they bind to, and `inspect`'s `_NonUserDefinedCallables` distinguishes them,
so both are pinned here.

`descrobject.c classmethod_get` requires the owner argument to be a type
*and* a subtype of the type the descriptor was found on; `funcobject.c
cm_descr_get`, the user `@classmethod` half, accepts anything.  Without the
subtype check the wrapped implementation runs against an unrelated class and
raises from inside itself instead of rejecting the call.

`typeobject.c wrap_descr_get` is the `__get__` entry every descriptor type
shares: `PyArg_UnpackTuple(args, "__get__", 1, 2)` makes the owner optional,
so a one-argument `__get__` binds.
"""

FK = dict.__dict__["fromkeys"]
PREPARE = type.__dict__["__prepare__"]
FROM_BYTES = int.__dict__["from_bytes"]

assert type(FK).__name__ == "classmethod_descriptor", type(FK)
assert type(PREPARE).__name__ == "classmethod_descriptor", type(PREPARE)
assert type(FROM_BYTES).__name__ == "classmethod_descriptor", type(FROM_BYTES)
assert type(FK) is type(PREPARE)

assert FK.__name__ == "fromkeys"
assert FK.__qualname__ == "dict.fromkeys"
assert FK.__objclass__ is dict
assert repr(FK) == "<method 'fromkeys' of 'dict' objects>"

# A decorated Python function is the other half and stays a `classmethod`.


class _Plain:
    @classmethod
    def m(cls):
        return cls


assert type(_Plain.__dict__["m"]).__name__ == "classmethod"

# Binding yields a builtin carrier holding the class, not a `method`.
bound = dict.fromkeys
assert type(bound).__name__ == "builtin_function_or_method", type(bound)
assert bound.__self__ is dict
assert bound(["a"]) == {"a": None}
assert type(_Plain.m).__name__ == "method"


class _MyDict(dict):
    pass


# The owner must be a type, and a subtype of the descriptor's own.
assert FK(dict, ["x"]) == {"x": None}
assert FK(_MyDict, ["x"]) == {"x": None}
assert type(FK(_MyDict, ["x"])) is _MyDict

for cls in (list, int):
    try:
        FK(cls, ["x"])
    except TypeError as exc:
        assert str(exc) == (
            f"descriptor 'fromkeys' requires a subtype of 'dict' "
            f"but received '{cls.__name__}'"
        ), exc
    else:
        raise AssertionError(f"an unrelated owner must be rejected: {cls}")

try:
    PREPARE(list, "X", ())
except TypeError as exc:
    assert str(exc) == (
        "descriptor '__prepare__' requires a subtype of 'type' but received 'list'"
    ), exc
else:
    raise AssertionError("an unrelated owner must be rejected")

try:
    FK(3, ["x"])
except TypeError as exc:
    assert str(exc) == (
        "descriptor 'fromkeys' for type 'dict' needs a type, not a 'int' as arg 2"
    ), exc
else:
    raise AssertionError("an instance owner must be rejected")

try:
    FK()
except TypeError as exc:
    assert str(exc) == (
        "descriptor 'fromkeys' of 'dict' object needs an argument"
    ), exc
else:
    raise AssertionError("a missing owner must be rejected")

# `__get__` validates the same way, and infers the owner from the instance.
assert FK.__get__({}).__self__ is dict
assert FK.__get__({}, None).__self__ is dict
assert FK.__get__(None, dict).__self__ is dict
assert FK.__get__(None, _MyDict).__self__ is _MyDict

for args in ((None, list), (None, 3)):
    try:
        FK.__get__(*args)
    except TypeError:
        pass
    else:
        raise AssertionError(f"__get__{args} must be rejected")

# `wrap_descr_get` arity, shared by all three descriptor kinds.
APPEND = list.__dict__["append"]
STR = object.__dict__["__str__"]

assert type(APPEND).__name__ == "method_descriptor"
assert type(STR).__name__ == "wrapper_descriptor"
assert APPEND.__get__([1]).__self__ == [1]
assert type(STR.__get__(3)).__name__ == "method-wrapper"
assert STR.__get__(3)() == "3"

for descr in (APPEND, STR, FK):
    try:
        descr.__get__()
    except TypeError as exc:
        assert str(exc) == "__get__ expected at least 1 argument, got 0", exc
    else:
        raise AssertionError(f"a zero-argument __get__ must be rejected: {descr}")

    try:
        descr.__get__(1, 2, 3)
    except TypeError as exc:
        assert str(exc) == "__get__ expected at most 2 arguments, got 3", exc
    else:
        raise AssertionError(f"a three-argument __get__ must be rejected: {descr}")

    try:
        descr.__get__(None, None)
    except TypeError as exc:
        assert str(exc) == "__get__(None, None) is invalid", exc
    else:
        raise AssertionError(f"__get__(None, None) must be rejected: {descr}")

# The getsets reject a foreign receiver with the standard descriptor wording.
for name in ("__objclass__", "__name__", "__qualname__", "__doc__",
             "__text_signature__"):
    try:
        type(FK).__dict__[name].__get__(3, int)
    except TypeError as exc:
        assert str(exc) == (
            f"descriptor '{name}' for 'classmethod_descriptor' objects "
            f"doesn't apply to a 'int' object"
        ), exc
    else:
        raise AssertionError(f"a foreign receiver must be rejected: {name}")

# `PyClassMethodDescr_Type` carries no `__reduce__`, so pickling is refused.
import pickle  # noqa: E402

try:
    pickle.dumps(FK)
except TypeError as exc:
    assert "classmethod_descriptor" in str(exc), exc
else:
    raise AssertionError("a classmethod_descriptor must not pickle")

print("OK")
