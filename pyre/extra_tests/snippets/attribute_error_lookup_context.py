# PyPy 9883bb2a9d `enrich_attribute_error`, with CPython 3.14's distinction
# between an omitted slot and an explicitly supplied None.  An inner lookup's
# more specific context wins over the outer `__getattr__` receiver, and getattr
# keeps the caller's name object.


class Empty:
    def __getattr__(self, name):
        raise AttributeError()


empty = Empty()
name = "".join(["mis", "sing"])
try:
    getattr(empty, name)
except AttributeError as exc:
    assert exc.name is name
    assert exc.obj is empty
else:
    raise AssertionError("getattr on a raising __getattr__ must fail")


class ExplicitNone:
    def __getattr__(self, name):
        raise AttributeError(name=None, obj=None)


try:
    ExplicitNone().missing
except AttributeError as exc:
    assert exc.name is None
    assert exc.obj is None
else:
    raise AssertionError("an explicit AttributeError must propagate")


class InnerLookup:
    def __getattr__(self, name):
        return getattr(list, name + "nd")


try:
    InnerLookup().app
except AttributeError as exc:
    assert exc.name == "appnd"
    assert exc.obj is list
else:
    raise AssertionError("the inner lookup must fail and win the context")
