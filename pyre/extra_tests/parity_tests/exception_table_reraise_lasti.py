"""RERAISE re-raises the same exception object without loss.

`pypy/interpreter/pyopcode.py:1361-1376 RERAISE` reads the original
raise-site lasti via `peekvalue(oparg)` and threads it through
`RaiseWithExplicitTraceback` so the unwound frame's `last_instr` points
back to the original raise — needed for `f_lineno`.  Without the lasti
field on the carrier, the bare-`raise` in a handler loses the
information.  We exercise the round-trip at the value level: both
exception identity and message must survive a nested re-raise.
"""


sentinel = ValueError("orig-payload")


def inner():
    raise sentinel


def middle():
    try:
        inner()
    except ValueError:
        # RERAISE (bare `raise` inside except) — must preserve identity.
        raise


def outer():
    try:
        middle()
    except ValueError as caught:
        return caught


got = outer()
assert got is sentinel, ("identity lost across RERAISE", id(got), id(sentinel))
assert str(got) == "orig-payload", str(got)

# Now exercise a deeper RERAISE chain — the inner reraise carries
# through two handlers.
def chain():
    try:
        try:
            try:
                raise KeyError("k")
            except KeyError:
                raise  # RERAISE 1
        except KeyError:
            raise  # RERAISE 2
    except KeyError as e:
        return e


got2 = chain()
assert isinstance(got2, KeyError), type(got2)
assert got2.args == ("k",), got2.args

print("OK")
