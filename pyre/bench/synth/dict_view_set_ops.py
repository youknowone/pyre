# A set operation on a dict view builds a set from the left operand and runs
# the matching in-place set method against the right one, so the result is a
# plain set. The reflected forms build the set from the other operand, keeping
# the operand order of the non-commutative `-` and `&`. A warmup loop
# exercises the view set-op path.
def warm(n):
    a = {"a": 1, "b": 2}
    b = {"b": 2, "c": 3}
    acc = 0
    for _ in range(n):
        acc += len(a.keys() & b.keys()) + len(a.keys() | b.keys())
    return acc


def m(label, fn):
    try:
        print(label, "->", repr(fn()))
    except BaseException as e:
        print(label, "!!", type(e).__name__, repr(str(e)))


def s(x):
    return sorted(x, key=repr)


K = {"a": 1, "b": 2}
K2 = {"b": 2, "c": 3}


def main():
    print("warm", warm(15000))
    # forward ops, view vs view
    m("keys_sub", lambda: s(K.keys() - K2.keys()))
    m("keys_and", lambda: s(K.keys() & K2.keys()))
    m("keys_or", lambda: s(K.keys() | K2.keys()))
    m("keys_xor", lambda: s(K.keys() ^ K2.keys()))
    # view vs set / other iterable
    m("keys_sub_set", lambda: s(K.keys() - {"a"}))
    m("keys_and_set", lambda: s(K.keys() & {"a"}))
    m("keys_or_list", lambda: s(K.keys() | ["z"]))
    m("keys_xor_set", lambda: s(K.keys() ^ {"a"}))
    # reflected: `-` and `&` are not commutative, so operand order matters
    m("rsub_set_keys", lambda: s({"a", "z"} - K.keys()))
    m("rand_set_keys", lambda: s({"a", "z"} & K.keys()))
    m("ror_set_keys", lambda: s({"z"} | K.keys()))
    m("rxor_set_keys", lambda: s({"a", "z"} ^ K.keys()))
    # items view
    m("items_sub", lambda: s(K.items() - K2.items()))
    m("items_and", lambda: s(K.items() & K2.items()))
    m("items_or", lambda: s(K.items() | K2.items()))
    m("items_xor", lambda: s(K.items() ^ K2.items()))
    # the result is a plain set
    m("result_type", lambda: type(K.keys() - K2.keys()).__name__)
    # an unhashable element in the view raises through the set constructor
    m("items_and_unhashable", lambda: {"a": []}.items() & {("a", 1)})
    # a non-iterable right operand
    m("keys_sub_int", lambda: K.keys() - 1)
    m("keys_and_none", lambda: K.keys() & None)
    # empty view
    m("empty_keys_and", lambda: s({}.keys() & {1}))
    # the values view is not set-like and has no set operations
    m("values_has_no_setop", lambda: K.values() & {1})


main()
