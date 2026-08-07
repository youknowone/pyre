import types


def assert_empty_popitem(d):
    try:
        d.popitem()
    except KeyError as exc:
        assert exc.args == ("popitem(): dictionary is empty",)
    else:
        raise AssertionError("popitem on empty dict did not raise")


def assert_lifo(label, d, expected):
    out = []
    while d:
        out.append(d.popitem())
    assert out == expected, (label, out, expected)
    assert_empty_popitem(d)


def make_kwargs(**kwargs):
    return kwargs


class DictSubclass(dict):
    pass


class Carrier:
    pass


assert_empty_popitem({})

single = {"only": 1}
assert single.popitem() == ("only", 1)
assert single == {}

reused = {1: "a", 2: "b"}
assert reused.popitem() == (2, "b")
reused[3] = "c"
assert reused.popitem() == (3, "c")
assert reused.popitem() == (1, "a")
assert_empty_popitem(reused)

assert_lifo("int", {1: "a", 2: "b", 3: "c"}, [(3, "c"), (2, "b"), (1, "a")])
assert_lifo("str", {"a": 1, "b": 2, "c": 3}, [("c", 3), ("b", 2), ("a", 1)])
assert_lifo("object", {1: "a", "b": 2, (3,): "c"}, [((3,), "c"), ("b", 2), (1, "a")])

kw = make_kwargs(a=1, b=2, c=3)
assert_lifo("kwargs", kw, [("c", 3), ("b", 2), ("a", 1)])

module_ns = globals()
module_ns["popitem_strategy_mod_a"] = 1
module_ns["popitem_strategy_mod_b"] = 2
assert module_ns.popitem() == ("popitem_strategy_mod_b", 2)
assert module_ns.popitem() == ("popitem_strategy_mod_a", 1)

obj = Carrier()
obj.first = 1
obj.second = 2
surrogate = "\ud800"
setattr(obj, surrogate, 3)
assert obj.__dict__.popitem() == (surrogate, 3)
assert obj.__dict__.popitem() == ("second", 2)
assert obj.__dict__.popitem() == ("first", 1)
assert_empty_popitem(obj.__dict__)

sub = DictSubclass()
sub["x"] = 1
sub["y"] = 2
assert_lifo("subclass", sub, [("y", 2), ("x", 1)])

# A non-str key moves a module dict off cell storage; popitem has a separate
# arm for that storage half, and a reader of a popped global must stop seeing it.
switched = types.ModuleType("popitem_strategy_switched")
exec("def read_g():\n    return g\n", switched.__dict__)
read_g = switched.read_g
switched.__dict__[42] = "not a str key"
switched.__dict__["g"] = "before"
assert read_g() == "before"
assert switched.__dict__.popitem() == ("g", "before")
try:
    read_g()
except NameError:
    pass
else:
    raise AssertionError("read_g must not see the popped global")
switched.__dict__["g"] = "after"
assert read_g() == "after"
assert switched.__dict__.popitem() == ("g", "after")
assert switched.__dict__.popitem() == (42, "not a str key")

print("OK")
