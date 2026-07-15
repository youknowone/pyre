# dict.fromkeys fills a plain dict through the dict's own setitem, so a key
# whose __hash__ raises propagates that exception instead of being stored
# unhashed. Only the keys and items views are set-like and therefore
# unhashable; the values view keeps object.__hash__. A warmup loop exercises
# the ordinary fromkeys / view paths.
def warm(n):
    acc = 0
    for i in range(n):
        d = dict.fromkeys((i % 4, (i + 1) % 4), i)
        acc += len(d) + len(d.keys())
    return acc


def m(label, fn):
    try:
        print(label, "->", repr(fn()))
    except BaseException as e:
        print(label, "!!", type(e).__name__, repr(str(e)))


class RaisingHash:
    def __hash__(self):
        raise ValueError("nope")


def main():
    print("warm", warm(15000))
    # fromkeys must run the hash protocol: a raising __hash__ propagates
    m("fromkeys_raising_hash", lambda: dict.fromkeys([RaisingHash()]))
    m("fromkeys_raising_hash_2nd", lambda: dict.fromkeys([1, RaisingHash()]))
    m("fromkeys_raising_hash_value", lambda: dict.fromkeys([RaisingHash()], 7))
    # ordinary fromkeys still works
    m("fromkeys_ok", lambda: dict.fromkeys([1, 2]))
    m("fromkeys_ok_value", lambda: dict.fromkeys([1, 2], 0))
    m("fromkeys_str", lambda: dict.fromkeys("ab", 1))
    m("fromkeys_empty", lambda: dict.fromkeys([]))
    m("fromkeys_dedup", lambda: dict.fromkeys([1, 1, 2]))
    m("fromkeys_generator", lambda: dict.fromkeys((i for i in range(3)), "v"))
    # the keys/items views are set-like -> unhashable, named by their own type
    m("hash_dict_keys", lambda: hash({"a": 1}.keys()))
    m("hash_dict_items", lambda: hash({"a": 1}.items()))
    # the values view is NOT set-like -> keeps object.__hash__ (value differs
    # per implementation, so only its type is compared)
    m("hash_dict_values_isint", lambda: isinstance(hash({"a": 1}.values()), int))
    m("hash_dict_values_stable", lambda: (lambda v: hash(v) == hash(v))({"a": 1}.values()))
    # view type names
    m("name_keys", lambda: type({}.keys()).__name__)
    m("name_values", lambda: type({}.values()).__name__)
    m("name_items", lambda: type({}.items()).__name__)


main()
