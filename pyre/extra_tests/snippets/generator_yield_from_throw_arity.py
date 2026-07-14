class ThrowArity:
    def __iter__(self):
        return self

    def __next__(self):
        return 1

    def throw(self, *args):
        return args


def delegate_throw_arity():
    yield from ThrowArity()


g = delegate_throw_arity()
next(g)
assert g.throw(ValueError) == (ValueError,)

g = delegate_throw_arity()
next(g)
assert g.throw(ValueError, "value") == (ValueError, "value")

g = delegate_throw_arity()
next(g)
assert g.throw(ValueError, "value", None) == (ValueError, "value", None)


class SingleArgThrow:
    def __iter__(self):
        return self

    def __next__(self):
        return 1

    def throw(self, typ):
        raise RuntimeError("got:" + typ.__name__)


def delegate_single_arg_throw():
    yield from SingleArgThrow()


g = delegate_single_arg_throw()
next(g)
try:
    g.throw(ValueError)
except RuntimeError as e:
    assert str(e) == "got:ValueError"
else:
    raise AssertionError("one-argument delegate throw was not called")
