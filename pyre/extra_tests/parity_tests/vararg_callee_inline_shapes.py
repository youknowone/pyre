# CPython-suite gap: call tests do not trace these *args callee seed shapes.
# parity-tests reason: this guards pyre's inlined vararg frame construction.

# Hot calls into a `*args` callee, across the shapes the inline seeding must
# reproduce and the ones it must decline.  The vararg local is
# `scope_w[co_argcount]` and holds `newtuple(starargs_w)`, so what is asserted
# here is the tuple's length, order, contents and the receiver's place in it.


def bare(*args):
    return args


def leading(a, *args):
    return (a, args)


def defaulted(a, b=5, *args):
    return (a, b, args)


def kwonly(a, *args, k=3):
    return (a, args, k)


def starstar(a, *args, **kw):
    return (a, args, sorted(kw))


class Holder:
    def method(self, *args):
        return (self.tag, args)

    def receiverless(*args):
        # No positional parameter to hold the receiver, so it becomes the
        # first element of the vararg tuple instead.
        return (args[0].tag, args[1:])

    def __init__(self):
        self.tag = 7


def one_surplus(rounds):
    acc = 0
    for i in range(rounds):
        acc = (acc * 31 + bare(i)[0]) & 0xFFFFFFFF
    return acc


def many_surplus(rounds):
    acc = 0
    for i in range(rounds):
        args = bare(i, i + 1, i + 2, i + 3)
        acc = (acc * 31 + len(args) + args[0] + args[3]) & 0xFFFFFFFF
    return acc


def no_surplus(rounds):
    # The empty tuple is a singleton, so this call must keep producing the
    # very same object.
    acc = 0
    first = bare()
    for i in range(rounds):
        acc = (acc * 31 + (1 if bare() is first else 0)) & 0xFFFFFFFF
    return acc


def with_leading(rounds):
    acc = 0
    for i in range(rounds):
        a, rest = leading(i, i + 1, i + 2)
        acc = (acc * 31 + a + len(rest) + rest[1]) & 0xFFFFFFFF
    return acc


def with_default(rounds):
    acc = 0
    for i in range(rounds):
        a, b, rest = defaulted(i, i + 1, i + 2)
        acc = (acc * 31 + a + b + len(rest)) & 0xFFFFFFFF
        a, b, rest = defaulted(i)
        acc = (acc * 31 + a + b + len(rest)) & 0xFFFFFFFF
    return acc


def with_kwonly(rounds):
    acc = 0
    for i in range(rounds):
        a, rest, k = kwonly(i, i + 1)
        acc = (acc * 31 + a + len(rest) + k) & 0xFFFFFFFF
    return acc


def with_starstar(rounds):
    acc = 0
    for i in range(rounds):
        a, rest, names = starstar(i, i + 1, z=2)
        acc = (acc * 31 + a + len(rest) + len(names)) & 0xFFFFFFFF
    return acc


def bound_method(rounds):
    holder = Holder()
    acc = 0
    for i in range(rounds):
        tag, rest = holder.method(i, i + 1)
        acc = (acc * 31 + tag + len(rest) + rest[0]) & 0xFFFFFFFF
    return acc


def bound_method_receiverless(rounds):
    holder = Holder()
    acc = 0
    for i in range(rounds):
        tag, rest = holder.receiverless(i)
        acc = (acc * 31 + tag + len(rest) + rest[0]) & 0xFFFFFFFF
    return acc


def detached_method(rounds):
    holder = Holder()
    unbound = Holder.method
    acc = 0
    for i in range(rounds):
        tag, rest = unbound(holder, i)
        acc = (acc * 31 + tag + len(rest) + rest[0]) & 0xFFFFFFFF
    return acc


def escaping(rounds):
    kept = []
    acc = 0
    for i in range(rounds):
        args = bare(i & 3, 1)
        kept.append(args)
        acc = (acc * 31 + hash(args)) & 0xFFFFFFFF
    return (acc, kept[0], kept[0] == (0, 1), len(kept))


def as_dict_key(rounds):
    table = {(k, 1): k for k in range(4)}
    acc = 0
    for i in range(rounds):
        acc = (acc * 31 + table[bare(i & 3, 1)]) & 0xFFFFFFFF
    return acc


def alternating(rounds):
    # The same call site alternating between surplus counts, so the trace
    # cannot pin one arity.
    acc = 0
    for i in range(rounds):
        args = bare(i) if i & 1 else bare(i, i + 1, i + 2)
        acc = (acc * 31 + len(args) + args[0]) & 0xFFFFFFFF
    return acc


def forwarding(rounds):
    acc = 0
    for i in range(rounds):
        acc = (acc * 31 + len(bare(*bare(i, i + 1)))) & 0xFFFFFFFF
    return acc


for fn in (one_surplus, many_surplus, no_surplus, with_leading, with_default,
           with_kwonly, with_starstar, bound_method,
           bound_method_receiverless, detached_method, escaping, as_dict_key,
           alternating, forwarding):
    print(fn.__name__, fn(3000))
print("OK")
