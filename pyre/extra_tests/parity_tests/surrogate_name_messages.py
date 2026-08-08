"""A lone surrogate in a name survives into every message that quotes the name.

`surrogateescape` puts an unpaired surrogate into a `str` for every undecodable
filesystem byte, and a name assigned from such a string keeps it.  Every surface
that quotes the name -- a repr, the argument binder's TypeErrors, the text handed
to `sys.stderr` -- rebuilds a `str` from it, and assembling that through a lossy
UTF-8 encode substitutes U+FFFD instead, silently changing the value.

The reprs below read the surrogate out of a `__repr__` rather than out of a
plain `str`, because `str.__repr__` backslash-escapes a surrogate and would hide
the difference.
"""

import contextvars
import io
import sys

S = "q\udcffn"
FFFD = "�"


def at(obj):
    """The address a repr quotes, in the spelling this platform's repr uses.

    `PyUnicode_FromFormat`'s `%p` hands the pointer to the platform's own
    `printf` and normalizes only the prefix, so the MSVC runtime's padding and
    upper case are part of the repr there and glibc's bare lower case is part
    of it everywhere else.
    """
    if sys.platform == "win32":
        return "0x%0*X" % (2 * (8 if sys.maxsize > 2**32 else 4), id(obj))
    return "0x%x" % id(obj)


class Repr:
    def __repr__(self):
        return S


def check_function_qualname():
    def f(a):
        return a

    f.__qualname__ = S
    assert repr(f) == "<function %s at %s>" % (S, at(f)), ascii(repr(f))

    try:
        f(1, 2)
    except TypeError as e:
        assert str(e) == S + "() takes 1 positional argument but 2 were given", ascii(str(e))
    else:
        raise AssertionError("no TypeError")

    try:
        f()
    except TypeError as e:
        assert str(e) == S + "() missing 1 required positional argument: 'a'", ascii(str(e))
    else:
        raise AssertionError("no TypeError")

    try:
        f(1, zz=2)
    except TypeError as e:
        assert str(e) == S + "() got an unexpected keyword argument 'zz'", ascii(str(e))
    else:
        raise AssertionError("no TypeError")

    # A defaulted parameter takes the binder down its other arm, which builds
    # the same two messages from the same name.
    def g(a, b=1):
        return a, b

    g.__qualname__ = S

    try:
        g(1, 2, 3)
    except TypeError as e:
        expected = S + "() takes from 1 to 2 positional arguments but 3 were given"
        assert str(e) == expected, ascii(str(e))
    else:
        raise AssertionError("no TypeError")

    try:
        g(1, a=2)
    except TypeError as e:
        assert str(e) == S + "() got multiple values for argument 'a'", ascii(str(e))
    else:
        raise AssertionError("no TypeError")


def check_bound_method_repr():
    class C:
        def m(self, a):
            return a

    C.m.__qualname__ = S
    o = C()
    assert repr(o.m).startswith("<bound method %s of " % S), ascii(repr(o.m))
    assert FFFD not in repr(o.m), ascii(repr(o.m))

    try:
        o.m()
    except TypeError as e:
        assert str(e) == S + "() missing 1 required positional argument: 'a'", ascii(str(e))
    else:
        raise AssertionError("no TypeError")


def check_unhashable_key_wrapper():
    # The wrapped hash error becomes this exception's own args[0], so the
    # surrogate is the value it carries -- escaping belongs to the display.
    class BadHash:
        def __hash__(self):
            raise TypeError(S)

    # Only the wrapped hash message is pinned here.  The type is spelled by
    # __name__ rather than __qualname__, which is a separate divergence from
    # the one this checks.
    for build, kind in ((lambda: {}[BadHash()], "dict key"), (lambda: {BadHash()}, "set element")):
        try:
            build()
        except TypeError as e:
            tail = " as a %s (%s)" % (kind, S)
            assert str(e).startswith("cannot use '"), ascii(str(e))
            assert str(e).endswith(tail), ascii(str(e))
            assert e.args[0] == str(e), ascii(e.args[0])
            assert FFFD not in str(e), ascii(str(e))
        else:
            raise AssertionError("no TypeError")


def check_generator_qualname():
    def g():
        yield 1

    g.__qualname__ = S
    assert g().__qualname__ == S, ascii(g().__qualname__)


def check_contextvars():
    var = contextvars.ContextVar("n", default=Repr())
    var_repr = repr(var)
    assert var_repr == "<ContextVar name='n' default=%s at %s>" % (S, at(var)), ascii(var_repr)

    token = var.set("x")
    assert repr(token) == "<Token var=%s at %s>" % (var_repr, at(token)), ascii(repr(token))
    var.reset(token)
    used_repr = repr(token)
    assert used_repr == "<Token used var=%s at %s>" % (var_repr, at(token)), ascii(used_repr)
    try:
        var.reset(token)
    except RuntimeError as e:
        assert str(e) == used_repr + " has already been used once", ascii(str(e))
    else:
        raise AssertionError("no RuntimeError")

    # A variable with no default names itself by repr in the LookupError, so
    # the message carries whatever its own repr does.  The name holds the
    # surrogate, so a lossy conversion anywhere on that path shows up here.
    unset = contextvars.ContextVar(S)
    try:
        unset.get()
    except LookupError as e:
        assert str(e) == repr(unset), ascii(str(e))
        assert FFFD not in str(e), ascii(str(e))
    else:
        raise AssertionError("no LookupError")


def check_excepthook():
    buf = io.StringIO()
    saved = sys.stderr
    sys.stderr = buf
    try:
        raise ValueError(S)
    except ValueError:
        sys.excepthook(*sys.exc_info())
    finally:
        sys.stderr = saved
    rendered = buf.getvalue()
    assert rendered.endswith("ValueError: %s\n" % S), ascii(rendered)
    assert FFFD not in rendered, ascii(rendered)


def check_base_exception_str():
    assert str(BaseException(S)) == S, ascii(str(BaseException(S)))
    assert BaseException.__str__(ValueError(S)) == S, ascii(BaseException.__str__(ValueError(S)))


check_function_qualname()
check_bound_method_repr()
check_unhashable_key_wrapper()
check_generator_qualname()
check_contextvars()
check_excepthook()
check_base_exception_str()
print("OK")
