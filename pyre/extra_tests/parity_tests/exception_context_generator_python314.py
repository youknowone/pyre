"""Implicit ``__context__`` recording across generator resume boundaries.

Two rules interact here.  An exception a frame raises itself takes the
*logical* handled exception, walking to whichever generator parked one; an
exception *thrown into* a resumed generator takes only that generator's own,
because the thrower merely delivered it rather than raising it under a live
handler.  And either answer -- including "no context" -- is recorded once, at
the frame where the error first surfaces, so an outer frame handling something
of its own never overwrites it.
"""


def gen_plain_raise():
    yield
    raise KeyError("k")


def gen_no_handler():
    yield


def gen_own_except():
    try:
        raise IndexError("inner")
    except IndexError:
        yield


def gen_own_except_then_raise():
    try:
        raise IndexError("inner2")
    except IndexError:
        yield
        raise TypeError("t")


def context_of(callback, exc_type):
    try:
        callback()
    except exc_type as caught:
        return caught.__context__
    raise AssertionError(f"expected {exc_type.__name__}")


def resumed(generator):
    next(generator)
    return generator


# No handler is live anywhere: both the raise and the throw record no context.
assert context_of(lambda: next(resumed(gen_plain_raise())), KeyError) is None
g = resumed(gen_no_handler())
assert context_of(lambda: g.throw(ValueError("v")), ValueError) is None

# The plain nested-raise control, with no generator involved.
try:
    raise ValueError("v3")
except ValueError as outer:
    assert context_of(lambda: exec('raise KeyError("k3")'), KeyError) is outer

# A generator parked inside its own `except` supplies the context for an
# exception thrown into it, and the caller's live handler stays out of it.
g = resumed(gen_own_except())
try:
    raise ValueError("caller")
except ValueError:
    thrown_context = context_of(lambda: g.throw(KeyError("k")), KeyError)
assert isinstance(thrown_context, IndexError), thrown_context
assert thrown_context.args == ("inner",), thrown_context

# The same generator state, but the generator raises on resume instead: its own
# handled exception still wins over the caller's.
g = resumed(gen_own_except_then_raise())
try:
    raise ValueError("caller2")
except ValueError:
    raised_context = context_of(lambda: next(g), TypeError)
assert isinstance(raised_context, IndexError), raised_context
assert raised_context.args == ("inner2",), raised_context

# The recorded-once rule.  The generator holds no handled exception, so the
# throw records `None` -- and the caller, which is inside its own `except`, must
# not overwrite that as the exception unwinds back out.
g = resumed(gen_no_handler())
try:
    raise ValueError("v4")
except ValueError:
    assert context_of(lambda: g.throw(KeyError("k4")), KeyError) is None

# A context recorded by a throw keeps its own chain: the generator's handled
# exception was itself raised with nothing live, so it ends the chain.
g = resumed(gen_own_except())
try:
    raise IndexError("caller3")
except IndexError:
    nested = context_of(lambda: g.throw(TypeError("t")), TypeError)
assert isinstance(nested, IndexError) and nested.args == ("inner",), nested
assert nested.__context__ is None, nested.__context__

print("ok")
