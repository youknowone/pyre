# CPython-suite gap: no suite test calls a keyword-only callee in a loop hot
# enough to trace and then mutates `__kwdefaults__` under it.
# parity-tests reason: this guards pyre's inlined keyword-only frame seeding,
# which fills each such local by reading `__kwdefaults__` live rather than by
# binding the value the definition happened to install.

# The seeding reads the dict on every call.  What that has to preserve is that
# a write to `__kwdefaults__` -- rebinding it, or storing into it in place --
# is visible to the very next call, and every runtime makes it visible
# immediately.  Baking the value the trace was recorded with would satisfy
# every assertion about the *unmutated* default and still be wrong here.
#
# N is what puts each loop past the trace threshold; at 300 the loops run
# interpreted and cover nothing this file exists to cover.

N = 30000


def only_kw(a, *, p=1):
    return (a, p)


def many_kw(a, *, p=1, q=2, r=3, s=4, t=5):
    return p + q + r + s + t


def star_and_kw(a, *rest, p=7):
    return (a, rest, p)


def pos_default_and_kw(a, b=2, *, p=3):
    return (a, b, p)


class Holder:
    def method(self, a, *, p=11):
        return (self.tag, a, p)

    def __init__(self):
        self.tag = 4


def plain_default(rounds):
    acc = 0
    for i in range(rounds):
        a, p = only_kw(i)
        acc = (acc * 31 + a + p) & 0xFFFFFFFF
    return acc


def several_defaults(rounds):
    acc = 0
    for i in range(rounds):
        acc = (acc * 31 + many_kw(i)) & 0xFFFFFFFF
    return acc


def with_vararg(rounds):
    # The vararg slot sits past the keyword-only block, so this is where a
    # wrong slot index shows up: `rest` and `p` would swap.
    empty = ()
    acc = 0
    for i in range(rounds):
        a, rest, p = star_and_kw(i)
        assert rest is empty
        acc = (acc * 31 + a + p) & 0xFFFFFFFF
    for i in range(rounds):
        a, rest, p = star_and_kw(i, i + 1, i + 2)
        assert rest == (i + 1, i + 2)
        acc = (acc * 31 + a + p + len(rest)) & 0xFFFFFFFF
    return acc


def with_positional_default(rounds):
    acc = 0
    for i in range(rounds):
        a, b, p = pos_default_and_kw(i)
        acc = (acc * 31 + a + b + p) & 0xFFFFFFFF
    return acc


def bound(rounds):
    holder = Holder()
    acc = 0
    for i in range(rounds):
        tag, a, p = holder.method(i)
        acc = (acc * 31 + tag + a + p) & 0xFFFFFFFF
    return acc


def passed_by_keyword(rounds):
    # A keyword-only parameter actually supplied at the call site: the default
    # must not win over it.
    acc = 0
    for i in range(rounds):
        a, p = only_kw(i, p=i + 1)
        assert p == i + 1
        acc = (acc * 31 + a + p) & 0xFFFFFFFF
    return acc


def mutated_in_place(rounds):
    def f(a, *, p=1):
        return p

    seen = []
    last = None
    for i in range(rounds):
        v = f(i)
        if v != last:
            seen.append((i, v))
            last = v
        if i == rounds // 2:
            f.__kwdefaults__['p'] = 99
    return seen


def rebound(rounds):
    def g(a, *, q=1):
        return q

    seen = []
    last = None
    for i in range(rounds):
        v = g(i)
        if v != last:
            seen.append((i, v))
            last = v
        if i == rounds // 2:
            g.__kwdefaults__ = {'q': 77}
    return seen


def mod_inplace_target(a, *, p=1):
    return p


def mod_rebound_target(a, *, q=1):
    return q


def module_level_mutated_in_place(rounds):
    # A module-level callee reaches the inline with its function baked, which
    # is a different arm from the closures above: nothing watches
    # `__kwdefaults__` on a baked callee, so the mapping is pinned by identity
    # instead.  An in-place store keeps that same mapping and has to stay
    # visible straight through the pin.
    seen = []
    last = None
    for i in range(rounds):
        v = mod_inplace_target(i)
        if v != last:
            seen.append((i, v))
            last = v
        if i == rounds // 2:
            mod_inplace_target.__kwdefaults__['p'] = 99
    return seen


def module_level_rebound(rounds):
    # Rebinding swaps out the very mapping that pin names, so this is the case
    # the pin exists to catch.
    seen = []
    last = None
    for i in range(rounds):
        v = mod_rebound_target(i)
        if v != last:
            seen.append((i, v))
            last = v
        if i == rounds // 2:
            mod_rebound_target.__kwdefaults__ = {'q': 77}
    return seen


def grown_after_delete(rounds):
    # Deleting the entry leaves the parameter with no default at all, so the
    # next call is a TypeError -- the inline must not keep answering with the
    # value it recorded.
    def h(a, *, p=5):
        return p

    ok = 0
    for i in range(rounds):
        ok += h(i)
        if i == rounds // 2:
            del h.__kwdefaults__['p']
            try:
                h(i)
            except TypeError:
                ok += 1
            h.__kwdefaults__['p'] = 5
    return ok


for fn in (plain_default, several_defaults, with_vararg,
           with_positional_default, bound, passed_by_keyword,
           mutated_in_place, rebound,
           module_level_mutated_in_place, module_level_rebound,
           grown_after_delete):
    print(fn.__name__, fn(N))
print("OK")
