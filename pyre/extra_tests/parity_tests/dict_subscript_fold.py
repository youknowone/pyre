N = 3000


def read_tag(d, k):
    n = 0
    for _ in range(N):
        n += d[k]
    return n


def read_last(d, k):
    out = None
    for _ in range(N):
        out = d[k]
    return out


def read_mixed(d, k):
    out = None
    for _ in range(N):
        out = d[k]
    return out


d = {"tag": 1}
assert read_tag(d, "tag") == N
d["tag"] = 2
assert read_tag(d, "tag") == 2 * N

gone = {"tag": 3}
assert read_tag(gone, "tag") == 3 * N
del gone["tag"]
try:
    read_last(gone, "tag")
except KeyError as exc:
    assert exc.args == ("tag",)
else:
    raise AssertionError("deleted key still readable")


class MissingDict(dict):
    def __missing__(self, key):
        return 41


missing = MissingDict()
assert read_last(missing, "tag") == 41

mixed = {1: "int", "1": "str"}
assert read_mixed(mixed, 1) == "int"
assert read_mixed(mixed, "1") == "str"

computed_key = "".join(["ta", "g"])
assert computed_key == "tag"
assert read_last({"tag": "computed"}, computed_key) == "computed"


class MyStr(str):
    def __hash__(self):
        return str.__hash__(self)

    def __eq__(self, other):
        return str.__eq__(self, other)


incoming_sub = MyStr("tag")
assert read_last({"tag": "incoming"}, incoming_sub) == "incoming"

stored_sub = MyStr("tag")
stored_sub_dict = {stored_sub: "stored"}
assert next(iter(stored_sub_dict)) is stored_sub
assert read_last(stored_sub_dict, "tag") == "stored"

surrogate = "\ud800"
assert read_last({surrogate: "surrogate"}, surrogate) == "surrogate"

devolved = {"tag": 5}
assert read_tag(devolved, "tag") == 5 * N
devolved[1] = 99
assert read_tag(devolved, "tag") == 5 * N


class CountingKey:
    hash_count = 0
    eq_count = 0

    def __init__(self, value):
        self.value = value

    def __hash__(self):
        type(self).hash_count += 1
        return 123

    def __eq__(self, other):
        type(self).eq_count += 1
        return isinstance(other, CountingKey) and self.value == other.value


stored = CountingKey("tag")
probe = CountingKey("tag")
counted = {stored: 7}
CountingKey.hash_count = 0
CountingKey.eq_count = 0
assert read_tag(counted, probe) == 7 * N
assert CountingKey.hash_count == N
assert CountingKey.eq_count == N

CountingKey.hash_count = 0
CountingKey.eq_count = 0
assert read_tag(counted, stored) == 7 * N
assert CountingKey.hash_count == N
assert CountingKey.eq_count == 0
