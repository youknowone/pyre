# pyre-check: max-pypy-ratio=40
N = 60000
M = 4000

# `cls.__name__` read from inside a method the enclosing `for` loop inlines.
# The read folds to the class's name slot under two guards -- the receiver is a
# type, and its metaclass is `type` -- so what this pins is the ways that can
# be wrong: a rename must be seen, because the slot is read live rather than
# baked, and its object identity must survive; two classes sharing the trace
# must each report their own name, since the fold deliberately does not pin
# which class arrived; a class dict entry of the same name must lose, because
# the metatype descriptor is consulted first; and any metaclass other than
# `type` must not be folded past at all.


class Base:
    @classmethod
    def tag(cls, i):
        return i + len(cls.__name__)


class Derived(Base):
    pass


class Longer(Base):
    pass


class Shadowed(Base):
    # `type.__name__` is a data descriptor on the metatype, so this entry is
    # what an INSTANCE of the class reads and never what the class does.
    __name__ = 'shadow-entry'


class GetattrMeta(type):
    def __getattribute__(cls, name):
        # 8 characters, which no class here is named, so the total says whether
        # this ran.
        if name == '__name__':
            return 'via-meta'
        return type.__getattribute__(cls, name)


class NameMeta(type):
    # A metaclass may define `__name__` itself, and then it is the one that
    # answers -- also 8 characters.
    @property
    def __name__(cls):
        return 'metaname'


class PlainMeta(type):
    # Answers exactly as `type` would, and still must not be folded past: the
    # fold's precondition is the metaclass, not what it happens to do.
    pass


class ByGetattr(metaclass=GetattrMeta):
    @classmethod
    def tag(cls, i):
        return i + len(cls.__name__)


class ByNameProp(metaclass=NameMeta):
    pass


class ByPlain(metaclass=PlainMeta):
    pass


class Box:
    def __init__(self, v):
        self.v = v

    # The instance-attribute twin: the same inline the type-name fold reaches,
    # over the mapdict fold instead.
    def at(self, i):
        return self.v + i


def folded():
    box = Box(4)
    total = 0
    for i in range(N):
        # Three classes down one trace; each must answer with its own name.
        total = total + Derived.tag(i) - i
        total = total + Longer.tag(i) - i
        total = total + Shadowed.tag(i) - i
        total = total + box.at(i) - i
        # Renaming mid-loop, after the loop is compiled: the fold reads the
        # name slot rather than a constant, so the shorter name is seen from
        # the next iteration on.
        if i == N // 2:
            Derived.__name__ = 'D'
    return total


def declined():
    # Every receiver here has a metaclass that is not `type`, so the fold must
    # not run.  Long enough to compile the loop and reach that decision, and no
    # longer -- what this pins is the decision, not a speed.
    total = 0
    for i in range(M):
        total = total + ByGetattr.tag(i) - i
        total = total + len(ByNameProp.__name__)
        total = total + len(ByPlain.__name__)
    return total


print(folded())
print(declined())
print(Derived.__name__, Longer.__name__, Shadowed.__name__)
print(Shadowed.__dict__['__name__'], Shadowed().__name__)
print(ByGetattr.__name__, ByNameProp.__name__, ByPlain.__name__)

# A rename between two compiled loops, rather than inside one: the second loop
# reads the slot the first one's trace was built against, so a baked name would
# survive here too.
Box.__name__ = 'Renamed'
seen = None
for _ in range(M):
    seen = Box.__name__
print(seen)
