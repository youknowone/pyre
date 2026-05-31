# A metaclass __new__ receives the keyword arguments from the class
# definition (`class C(metaclass=Meta, key=value)`).  Self-contained
# (no testutils import) so it runs on every backend.


class CollectMeta(type):
    def __new__(mcs, name, bases, ns, **kw):
        ns["meta_kwargs"] = dict(kw)
        return super().__new__(mcs, name, bases, ns)


class WithKwargs(metaclass=CollectMeta, a=1, b=2):
    pass


assert WithKwargs.meta_kwargs == {"a": 1, "b": 2}


# Keyword-only parameters on the metaclass __new__ bind by name, and
# fall back to their defaults when the class definition omits them.
class KindMeta(type):
    def __new__(mcs, name, bases, ns, *, kind="default"):
        ns["kind"] = kind
        return super().__new__(mcs, name, bases, ns)


class Special(metaclass=KindMeta, kind="special"):
    pass


class Plain(metaclass=KindMeta):
    pass


assert Special.kind == "special"
assert Plain.kind == "default"


# A metaclass with no class keyword arguments still constructs normally.
class BareMeta(type):
    pass


class Bare(metaclass=BareMeta):
    pass


assert type(Bare) is BareMeta


# A keyword the metaclass __new__ signature cannot accept raises TypeError.
class StrictMeta(type):
    def __new__(mcs, name, bases, ns):
        return super().__new__(mcs, name, bases, ns)


raised = False
try:
    class _Bad(metaclass=StrictMeta, bad=1):
        pass
except TypeError:
    raised = True
assert raised, "unexpected metaclass keyword should raise TypeError"

print("class_metaclass_kwargs: OK")
