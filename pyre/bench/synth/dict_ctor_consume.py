# dict.__new__ allocates and ignores its arguments; filling the instance is
# __init__'s job. So dict(x) walks x exactly once: a mapping's keys() and
# __getitem__ each run one time per key, a one-shot iterable is not re-entered,
# and a key is hashed once. A warmup loop exercises the pairs and mapping
# constructor paths.
def warm(n):
    acc = 0
    for i in range(n):
        d = dict([(i % 8, 1), ((i + 1) % 8, 2)])
        e = dict({i % 4: 3})
        f = dict(a=i)
        acc += len(d) + len(e) + len(f)
    return acc


def m(label, fn):
    try:
        print(label, "->", repr(fn()))
    except BaseException as e:
        print(label, "!!", type(e).__name__, repr(str(e)))


K = []


class Mapping:
    def keys(self):
        K.append("keys")
        return ["a"]

    def __getitem__(self, k):
        K.append("getitem")
        return 1


def calls(label, fn):
    K[:] = []
    try:
        fn()
        print(label, "->", K)
    except BaseException as e:
        print(label, "!!", type(e).__name__, K)


class Once:
    """Refuses a second walk, so a double-consume surfaces as a raise."""

    def __init__(self):
        self.n = 0

    def __iter__(self):
        self.n += 1
        if self.n > 1:
            raise RuntimeError("consumed twice")
        return iter([("a", 1)])


N = [0]


class Counted:
    def __hash__(self):
        N[0] += 1
        return 7

    def __eq__(self, other):
        return self is other


def hashes(label, fn):
    N[0] = 0
    try:
        fn()
        print(label, "->", N[0])
    except BaseException as e:
        print(label, "!!", type(e).__name__, N[0])


class Sub(dict):
    pass


def main():
    print("warm", warm(15000))
    # the source is walked once, not once by __new__ and again by __init__
    calls("mapping_ctor", lambda: dict(Mapping()))
    calls("mapping_update", lambda: {}.update(Mapping()))
    calls("mapping_ctor_kw", lambda: dict(Mapping(), z=9))
    m("once_ctor", lambda: sorted(dict(Once()).items()))
    m("once_update", lambda: (lambda d: (d.update(Once()), sorted(d.items()))[1])({}))
    # a key is hashed once per store
    hashes("h_ctor_list_pairs", lambda: dict([(Counted(), 1)]))
    hashes("h_ctor_tuple_pairs", lambda: dict(((Counted(), 1),)))
    hashes("h_ctor_gen_pairs", lambda: dict((Counted(), 1) for _ in range(1)))
    hashes("h_ctor_dict", lambda: dict({Counted(): 1}))
    hashes("h_ctor_mapping", lambda: dict({}))
    hashes("h_copy", lambda: {Counted(): 1}.copy())
    # every constructor shape still fills
    m("ctor_pairs", lambda: sorted(dict([("a", 1), ("b", 2)]).items()))
    m("ctor_tuple_pairs", lambda: sorted(dict((("a", 1),)).items()))
    m("ctor_gen_pairs", lambda: sorted(dict((k, 1) for k in "xy").items()))
    m("ctor_mapping", lambda: sorted(dict({"b": 2}).items()))
    m("ctor_dict_dup", lambda: sorted(dict({"b": 2, "c": 3}).items()))
    m("ctor_kwargs", lambda: sorted(dict(c=3).items()))
    m("ctor_pairs_kwargs", lambda: sorted(dict([("a", 1)], d=4).items()))
    m("ctor_mapping_kwargs", lambda: sorted(dict({"a": 1}, d=4).items()))
    m("ctor_kwargs_override", lambda: sorted(dict([("a", 1)], a=9).items()))
    m("ctor_empty", lambda: dict())
    m("ctor_str_keys", lambda: sorted(dict(zip("ab", [1, 2])).items()))
    m("ctor_int_keys", lambda: sorted(dict([(1, "x"), (2, "y")]).items()))
    m("ctor_dup_keys", lambda: sorted(dict([("a", 1), ("a", 2)]).items()))
    # a dict subclass fills through the inherited __init__
    m("subclass_pairs", lambda: sorted(Sub([("a", 1)]).items()))
    m("subclass_kwargs", lambda: sorted(Sub(b=2).items()))
    m("subclass_empty", lambda: sorted(Sub().items()))
    # __new__ on its own allocates an empty dict, arguments ignored
    m("new_ignores_args", lambda: dict.__new__(dict, [("a", 1)]))
    m("ctor_not_iterable", lambda: dict(5))


main()
