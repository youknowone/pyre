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
# through two handlers.  Capture the original exception in the
# innermost handler so the outermost return can also be checked for
# object identity, not just type/args.
def chain():
    first = None
    try:
        try:
            try:
                raise KeyError("k")
            except KeyError as e:
                first = e
                raise  # RERAISE 1
        except KeyError:
            raise  # RERAISE 2
    except KeyError as e:
        return first, e


first2, got2 = chain()
assert got2 is first2, ("identity lost across nested RERAISE", id(got2), id(first2))
assert isinstance(got2, KeyError), type(got2)
assert got2.args == ("k",), got2.args


# Verify `pyopcode.py:181-184` no-handler propagation: the unwound
# frame's `last_instr` is restored to the original raise-site offset
# (visible via `tb.tb_lineno` of the deepest frame).  Without the
# restoration the traceback would point at the RERAISE site, not the
# original `raise`.
def raise_here():
    raise RuntimeError("original")  # ORIGINAL_RAISE_LINE


_ORIGINAL_RAISE_LINE = raise_here.__code__.co_firstlineno + 1


def reraise_here():
    try:
        raise_here()
    except RuntimeError:
        raise  # propagate via bare-`raise`; line != _ORIGINAL_RAISE_LINE


def collect_tb():
    try:
        reraise_here()
    except RuntimeError as e:
        return e.__traceback__


tb = collect_tb()
# Walk to the deepest frame in the traceback chain.
deepest = tb
while deepest.tb_next is not None:
    deepest = deepest.tb_next
assert deepest.tb_lineno == _ORIGINAL_RAISE_LINE, (
    "tb_lineno should point to the original raise, not the RERAISE",
    deepest.tb_lineno,
    _ORIGINAL_RAISE_LINE,
)


# Exercise RERAISE oparg > 0 via a `with` block: the __exit__ returning
# a falsy value re-raises with the saved lasti pushed by the exception
# table.  This is the path that requires `reraise_lasti` threading
# (pyopcode.py:1361 `peekvalue(oparg)` with oparg != 0).
class _NoSuppress:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        return None  # falsy → re-raise


def raise_in_with():
    raise IndexError("from-with")  # WITH_RAISE_LINE


_WITH_RAISE_LINE = raise_in_with.__code__.co_firstlineno + 1


def with_block():
    with _NoSuppress():
        raise_in_with()


def collect_with_tb():
    try:
        with_block()
    except IndexError as e:
        return e.__traceback__


tb_with = collect_with_tb()
deepest_with = tb_with
while deepest_with.tb_next is not None:
    deepest_with = deepest_with.tb_next
assert deepest_with.tb_lineno == _WITH_RAISE_LINE, (
    "with-block reraise lost the original raise lineno",
    deepest_with.tb_lineno,
    _WITH_RAISE_LINE,
)

print("OK")
