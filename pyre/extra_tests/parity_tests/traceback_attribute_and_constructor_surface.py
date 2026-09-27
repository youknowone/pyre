# pyre-check: pypy-diverges: pins 3.14's read-only tb_lasti and its
# keyword-taking constructor; pypy3 lets `tb.tb_lasti = 1` through and its
# `descr_new` takes no keyword arguments at all.
# CPython-suite gap: test_traceback covers tb_next's setter and the
# constructor's positional form, and nothing there reads the descriptor kind,
# the refusal wording for the other three fields, or a keyword call.
# parity-tests reason: the two upstreams disagree on both, so the assertions
# below are the pinned CPython behaviour and the header records what PyPy does.

"""`traceback`'s four attributes are not one kind of descriptor.

`tb_memberlist` holds `tb_frame` and `tb_lasti` as `Py_READONLY` members, and
`tb_getsetters` holds `tb_lineno` (no setter) and `tb_next` (a setter that
checks the chain).  The split is observable three ways: the descriptor's own
type, its repr, and what a refused write says -- a member answers `readonly
attribute`, a getset with a null setter answers `attribute '<name>' of
'traceback' objects is not writable`.  Registering all four alike answers one
of those for the other's fields.

`PyTraceback.typedef` makes `tb_frame` an `interp_attrproperty_w` and the other
three writable `GetSetProperty`s, so the read-only half lines up while
`tb_lasti` and `tb_lineno` accept writes there.

The constructor declares `(tb_next, tb_frame, tb_lasti, tb_lineno)` as
positional-or-keyword: keywords fill their own slots, the limit counts
positionals and keywords together, and the first slot left empty is what gets
reported -- so a keyword duplicating a filled slot is reported against the
empty one, not against itself.
"""

from types import TracebackType

try:
    raise ValueError("x")
except ValueError as exc:
    tb = exc.__traceback__
frame = tb.tb_frame


def refuses(action, message):
    try:
        action()
    except Exception as err:  # noqa: BLE001 - the type is asserted below
        got = f"{type(err).__name__}: {err}"
        assert got == message, f"expected {message!r}\n     got {got!r}"
    else:
        raise AssertionError(f"expected {message!r}, nothing raised")


# The two members: one kind, one repr, one refusal for both a write and a
# delete.
for name in ("tb_frame", "tb_lasti"):
    descr = getattr(TracebackType, name)
    assert type(descr).__name__ == "member_descriptor", (name, type(descr))
    assert repr(descr) == f"<member '{name}' of 'traceback' objects>", repr(descr)
    assert descr.__objclass__ is TracebackType, descr.__objclass__
    refuses(
        lambda n=name: setattr(tb, n, None), "AttributeError: readonly attribute"
    )
    refuses(lambda n=name: delattr(tb, n), "AttributeError: readonly attribute")

# The two getsets.  `tb_lineno` has no setter and says so in the other wording;
# `tb_next` has one, and rejects a non-traceback and a loop.
for name in ("tb_lineno", "tb_next"):
    descr = getattr(TracebackType, name)
    assert type(descr).__name__ == "getset_descriptor", (name, type(descr))
    assert repr(descr) == f"<attribute '{name}' of 'traceback' objects>", repr(descr)

refuses(
    lambda: setattr(tb, "tb_lineno", 1),
    "AttributeError: attribute 'tb_lineno' of 'traceback' objects is not writable",
)
refuses(
    lambda: delattr(tb, "tb_lineno"),
    "AttributeError: attribute 'tb_lineno' of 'traceback' objects is not writable",
)
refuses(
    lambda: setattr(tb, "tb_next", "x"),
    "TypeError: expected traceback object, got 'str'",
)
refuses(
    lambda: delattr(tb, "tb_next"),
    "TypeError: can't delete tb_next attribute",
)
refuses(
    lambda: setattr(tb, "tb_next", tb), "ValueError: traceback loop detected"
)

# The receiver check runs ahead of the refusal, for either descriptor kind.
for name in ("tb_frame", "tb_lasti", "tb_lineno", "tb_next"):
    descr = getattr(TracebackType, name)
    for call in (lambda d=descr: d.__get__(1), lambda d=descr: d.__set__(1, 0)):
        refuses(
            call,
            "TypeError: descriptor '%s' for 'traceback' objects "
            "doesn't apply to a 'int' object" % name,
        )

# Reads still answer the slots.
assert tb.tb_frame is frame
assert isinstance(tb.tb_lasti, int)
assert tb.tb_lineno == tb.tb_frame.f_lineno or isinstance(tb.tb_lineno, int)
assert tb.tb_next is None or isinstance(tb.tb_next, TracebackType)
assert [n for n in dir(TracebackType) if n.startswith("tb_")] == [
    "tb_frame",
    "tb_lasti",
    "tb_lineno",
    "tb_next",
]

# The constructor binds by name, in any order, mixed with positionals.
built = TracebackType(tb_lineno=3, tb_lasti=0, tb_frame=frame, tb_next=None)
assert (built.tb_next, built.tb_frame, built.tb_lasti, built.tb_lineno) == (
    None,
    frame,
    0,
    3,
)
mixed = TracebackType(None, frame, tb_lineno=3, tb_lasti=0)
assert (mixed.tb_lasti, mixed.tb_lineno) == (0, 3)
positional = TracebackType(None, frame, 0, 7)
assert positional.tb_lineno == 7

# Positionals and keywords count against one limit.
for call in (
    lambda: TracebackType(None, frame, 0, 0, 0),
    lambda: TracebackType(None, frame, 0, 0, tb_lineno=3),
    lambda: TracebackType(None, frame, 0, 0, spam=1),
):
    refuses(call, "TypeError: traceback() takes at most 4 arguments (5 given)")

# Below the limit, the first empty slot is what is reported -- including when
# the call also passed a keyword the constructor does not know, and when a
# keyword named a slot a positional had already filled.
refuses(
    lambda: TracebackType(),
    "TypeError: traceback() missing required argument 'tb_next' (pos 1)",
)
refuses(
    lambda: TracebackType(tb_frame=frame),
    "TypeError: traceback() missing required argument 'tb_next' (pos 1)",
)
for call in (
    lambda: TracebackType(None, frame, 0),
    lambda: TracebackType(None, frame, 0, spam=1),
    lambda: TracebackType(None, frame, 0, tb_lasti=1),
):
    refuses(
        call, "TypeError: traceback() missing required argument 'tb_lineno' (pos 4)"
    )

# The argument conversions are the clinic ones.
refuses(
    lambda: TracebackType(None, "x", 0, 0),
    "TypeError: traceback() argument 'tb_frame' must be frame, not str",
)
# `_PyArg_BadArgument` names the None singleton itself, not its type.
refuses(
    lambda: TracebackType(None, None, 0, 0),
    "TypeError: traceback() argument 'tb_frame' must be frame, not None",
)
refuses(
    lambda: TracebackType("x", frame, 0, 0),
    "TypeError: expected traceback object or None, got 'str'",
)
refuses(
    lambda: TracebackType(None, frame, "a", 0),
    "TypeError: 'str' object cannot be interpreted as an integer",
)

print("OK")
