assert bool(b"") is False
assert bool(bytearray()) is False
assert bool(b"x") is True
assert bool(bytearray(b"x")) is True
assert (not b"") is True
assert (not bytearray()) is True
# used in conditionals (exercise the JIT conditional-jump path too)
def f(x):
    if x:
        return "truthy"
    return "falsy"
assert f(b"") == "falsy"
assert f(bytearray()) == "falsy"
assert f(b"ab") == "truthy"
# while-loop consume (Unpickler-style): truthiness drives loop exit at empty
buf = bytearray(b"abc"); seen = []
while buf:
    seen.append(buf.pop(0))
assert seen == [97, 98, 99], seen
# regression: other empties stay correct
for v in ["", [], (), {}, set(), 0, 0.0]:
    assert bool(v) is False, v
print("builtin_bytes_truthiness OK")
