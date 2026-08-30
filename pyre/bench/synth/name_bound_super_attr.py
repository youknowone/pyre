# pyre-check: spec-folds=two_arg_super_call,load_attr_on_super
# `super(C, self)` bound to a name, then read — the spelling LOAD_SUPER_ATTR
# does not cover, because the name binding splits the proxy construction and
# the attribute load into two opcodes the compiler never fuses.
#
# The construction arrives as an opaque may-force `bh_call_fn`, which costs
# more than the four-word allocation it performs: the walk publishes a vref
# for the executing frame ahead of it and re-checks it after, nine ops around
# one.  The read arrives as an opaque MRO walk that also wipes the trace's
# heap-field cache.  Folded, the proxy is `New` + `SetfieldGc` whose reads the
# optimizer answers from the emission itself, so the allocation dies and this
# shape costs what `super(C, self).val(x)` costs.
#
try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

N = 10000


class Base:
    def val(self, x):
        return x + 1


class Child(Base):
    def val(self, x):
        su = super(Child, self)
        return su.val(x)


def run(o, n):
    acc = 0
    for _ in range(n):
        acc = o.val(acc)
    return acc


print(run(Child(), N))
