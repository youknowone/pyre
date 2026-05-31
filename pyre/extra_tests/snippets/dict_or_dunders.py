# dict.__or__ / __ror__ / __ior__ / __reversed__ exposed as methods.

a = {1: "a", 2: "b"}
b = {2: "B", 3: "c"}

# __or__ — copy left, overlay right.
assert a.__or__(b) == {1: "a", 2: "B", 3: "c"}
assert (a | b) == {1: "a", 2: "B", 3: "c"}
# left operand unchanged
assert a == {1: "a", 2: "b"}

# __ror__ — copy right-hand base (other), overlay self.
assert a.__ror__(b) == {2: "b", 3: "c", 1: "a"}

# non-dict operand → NotImplemented
assert a.__or__(3) is NotImplemented
assert a.__ror__("x") is NotImplemented

# __ior__ — in-place update, returns the same object.
c = {1: "a"}
res = c.__ior__({1: "z", 4: "d"})
assert res is c
assert c == {1: "z", 4: "d"}

d = {1: "a"}
d |= {5: "e"}
assert d == {1: "a", 5: "e"}

# __reversed__ — reverse iterator over keys (insertion order reversed).
keys = list({1: 0, 2: 0, 3: 0}.__reversed__())
assert keys == [3, 2, 1]
assert list(reversed({"x": 1, "y": 2})) == ["y", "x"]

print("dict_or_dunders ok")
