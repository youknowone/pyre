# CPython-suite gap: the suite checks what `__kwdefaults__` *contains*, never
# what kind of mapping it is.  Nothing there would notice a runtime that builds
# it as a specialised dict flavour instead of the one the bytecode produced.
# parity-tests reason: a function's keyword-only defaults live in a mapping the
# runtime is free to choose, and pyre chooses one whose storage differs from a
# plain dict's.  Everything below is what user code can still observe through
# it, so it has to answer exactly as CPython does whichever flavour is behind
# the attribute.

# The loop is what puts the reading callee past the trace threshold: a folded
# read of a default has to keep agreeing with the mapping the attribute hands
# back, so the surface is exercised while the caller is compiled rather than
# only from cold code.

N = 30000


def kw(a, *, p=1, q=2):
    return (a, p, q)


def no_kw(a, b=1):
    return (a, b)


def identity_and_type():
    d = kw.__kwdefaults__
    e = kw.__kwdefaults__
    return (type(d) is dict, isinstance(d, dict), d is e, len(d))


def equality_and_order():
    d = kw.__kwdefaults__
    plain = {'p': 1, 'q': 2}
    reordered = {'q': 2, 'p': 1}
    return (d == plain, plain == d, d == reordered, list(d), list(d.items()),
            list(d.keys()), list(d.values()), repr(d))


def copying():
    d = kw.__kwdefaults__
    c = d.copy()
    b = dict(d)
    u = {**d}
    return (type(c) is dict, c == d, c is not d,
            type(b) is dict, b == d, u == d)


def absent_when_no_kwonly():
    return (no_kw.__kwdefaults__, no_kw(1))


def mutating_surface():
    def g(a, *, p=5):
        return p

    out = []
    d = g.__kwdefaults__
    out.append(d.setdefault('p', 99))
    out.append(d.setdefault('z', 7))
    out.append(sorted(d))
    out.append(d.pop('z'))
    d['p'] = 6
    out.append(g(0))
    del d['p']
    out.append('p' in d)
    try:
        g(0)
    except TypeError:
        out.append('TypeError')
    d['p'] = 5
    out.append(g(0))
    out.append(g.__kwdefaults__ is d)
    return out


def non_str_key():
    # A non-string key is what forces a specialised str-keyed storage back to a
    # general one.  It is legal here -- the mapping is an ordinary dict as far
    # as user code is concerned -- and the keyword-only fill must keep working
    # across the switch.
    #
    # The key list is reported unsorted on purpose: rebuilding the storage is
    # the one step that could reorder, and sorting would hide exactly that.
    def g(a, *, p=5):
        return p

    d = g.__kwdefaults__
    d['q'] = 'two'
    d[1] = 'one'
    d[(2, 3)] = 'tuple'
    total = 0
    for i in range(N):
        total += g(i)
    return (total, d[1], d[(2, 3)], list(d), [str(k) for k in d.keys()],
            list(d.values()), g(0))


def post_switch_surface():
    # Everything in `copying` and `equality_and_order`, re-asked once the
    # mapping has been forced off the str-keyed storage: the general storage is
    # a second implementation of each of these and nothing else covers it.
    def g(a, *, p=5):
        return p

    d = g.__kwdefaults__
    d[1] = 'one'
    c = d.copy()
    b = dict(d)
    u = {**d}
    plain = {'p': 5, 1: 'one'}
    return (type(c) is dict, c == d, c is not d, list(c.items()),
            type(b) is dict, b == d, u == d,
            d == plain, plain == d, repr(d), len(d), d.get('p'), d.get('zz'),
            list(reversed(d)))


def destination_mutators():
    # `d` on the LEFT of an update is a separate path from every read-side copy
    # above: the fast path that adopts a source's storage wholesale is steered
    # away from for this mapping, so the generic item loop has to agree.
    def g(a, *, p=5):
        return p

    out = []
    d = g.__kwdefaults__
    d.update({'q': 6})
    d.update([('r', 7)], s=8)
    out.append(list(d.items()))
    d |= {'t': 9}
    out.append(list(d))
    out.append((d | {'u': 10})['u'])
    out.append(({'u': 10} | d)['p'])
    out.append(d.popitem())
    d.clear()
    out.append((len(d), list(d), g.__kwdefaults__ is d))
    try:
        g(0)
    except TypeError:
        out.append('TypeError')
    d['p'] = 5
    out.append(g(0))
    return out


def mutation_during_iteration():
    # Overwriting a value that is already there must stay legal while adding a
    # key must not -- the two are one bump apart in the storage, so a mapping
    # that got this wrong in either direction would still pass every test that
    # does not iterate.
    def g(a, *, p=5):
        return p

    d = g.__kwdefaults__
    d['q'] = 1
    value_only = 'ok'
    try:
        for k in d:
            d[k] = 0
    except RuntimeError:
        value_only = 'RuntimeError'
    grow = 'no-raise'
    try:
        for _ in d:
            d['fresh'] = 1
    except RuntimeError:
        grow = 'RuntimeError'
    return (value_only, grow, sorted(d))


def serialisation():
    import copy as copy_mod
    import pickle

    def g(a, *, p=5, q=[1, 2]):
        return p

    d = g.__kwdefaults__
    round_tripped = pickle.loads(pickle.dumps(d))
    deep = copy_mod.deepcopy(d)
    return (type(round_tripped) is dict, round_tripped == d,
            type(deep) is dict, deep == d, deep['q'] is not d['q'])


def rebound_flavour():
    def g(a, *, p=5):
        return p

    g.__kwdefaults__ = {'p': 8}
    d = g.__kwdefaults__
    first = (type(d) is dict, d == {'p': 8}, g(0))
    total = 0
    for i in range(N):
        total += g(i)
    g.__kwdefaults__ = None
    try:
        g(0)
    except TypeError:
        second = 'TypeError'
    else:
        second = 'no-raise'
    return (first, total, second, g.__kwdefaults__)


def hot_read_agrees_with_attribute():
    # The compiled caller and the attribute have to answer from the same
    # mapping: read the default through a call and through the attribute on the
    # same iteration, and change it under both halfway through.
    def g(a, *, p=1):
        return p

    seen = []
    for i in range(N):
        through_call = g(i)
        through_attr = g.__kwdefaults__['p']
        if through_call != through_attr:
            seen.append((i, through_call, through_attr))
        if i == N // 2:
            g.__kwdefaults__['p'] = 42
    return (seen, g(0), g.__kwdefaults__['p'])


for fn in (identity_and_type, equality_and_order, copying,
           absent_when_no_kwonly, mutating_surface, non_str_key,
           post_switch_surface, destination_mutators,
           mutation_during_iteration, serialisation,
           rebound_flavour, hot_read_agrees_with_attribute):
    print(fn.__name__, fn())
print("OK")
