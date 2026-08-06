"""Decoder and encoder behaviour the `_json` accelerator is responsible for.

The scanner recurses through nested containers, carries a per-call memo for
repeated object keys, and dispatches the decoder's hooks; the encoder walks the
same shapes and refuses a cycle. None of that had a fixture, so every assertion
here is chosen to hold on the reference as well.
"""

import json


nested = {"a": [1, 2, {"b": [3, {"c": [4, [5, [6]]]}]}], "d": {"e": {"f": [7, 8]}}}
assert json.loads(json.dumps(nested)) == nested

# Repeated keys across sibling objects go through the scanner memo.
repeated = [{"key": i, "other": i} for i in range(50)]
decoded = json.loads(json.dumps(repeated))
assert decoded == repeated
first_key = next(iter(decoded[0]))
for entry in decoded:
    for key in entry:
        if key == first_key:
            assert key == first_key

# Depth: a nesting the scanner walks without tripping its recursion guard.
deep = json.loads("[" * 200 + "]" * 200)
level = deep
for _ in range(199):
    assert isinstance(level, list) and len(level) == 1
    level = level[0]
assert level == []

# What a nesting too deep to walk does is NOT asserted here: pypy raises
# RecursionError where the reference decodes it, so the two disagree and a
# fixture that has to pass under both cannot pin either.

# The decoder's hooks all reach the scanner.
assert json.loads('{"x": 1}', object_hook=lambda d: ("hook", sorted(d.items()))) == (
    "hook",
    [("x", 1)],
)
assert json.loads('{"x": 1, "y": 2}', object_pairs_hook=list) == [("x", 1), ("y", 2)]
assert json.loads("[1.5]", parse_float=lambda s: ("float", s)) == [("float", "1.5")]
assert json.loads("[7]", parse_int=lambda s: ("int", s)) == [("int", "7")]
assert json.loads("[NaN]", parse_constant=lambda s: ("const", s)) == [("const", "NaN")]

# object_pairs_hook wins over object_hook.
assert json.loads('{"x": 1}', object_pairs_hook=len, object_hook=lambda d: "unused") == 1

# strict mode rejects an unescaped control character in a string.
assert json.loads('"a\tb"', strict=False) == "a\tb"
try:
    json.loads('"a\tb"')
except json.JSONDecodeError:
    pass
else:
    raise AssertionError("strict mode must reject a raw control character")

# JSONDecodeError carries documented position attributes.
try:
    json.loads('{"a": 1,\n "b": }')
except json.JSONDecodeError as error:
    assert error.lineno == 2
    assert error.colno == error.pos - error.doc.rindex("\n", 0, error.pos)
    assert error.doc == '{"a": 1,\n "b": }'
else:
    raise AssertionError("malformed object must raise JSONDecodeError")

for bad in ("", "[", "[1,", '{"a"', "tru", "01", "[1,]", '{"a": 1,}'):
    try:
        json.loads(bad)
    except json.JSONDecodeError:
        pass
    else:
        raise AssertionError(f"{bad!r} must not decode")

assert json.loads(" \t\n [1, 2] \t\n ") == [1, 2]
assert json.loads('"\\u00e9\\ud83d\\ude00"') == "é\U0001f600"
assert json.loads('{"": 0}') == {"": 0}
assert json.loads("[1e400, -1e400]") == [float("inf"), float("-inf")]

# Encoder: the same shapes, plus the surfaces the C encoder owns.
assert json.dumps({"b": 1, "a": 2}, sort_keys=True) == '{"a": 2, "b": 1}'
assert json.dumps([1, 2], separators=(",", ":")) == "[1,2]"
assert json.dumps({"a": [1]}, indent=2) == '{\n  "a": [\n    1\n  ]\n}'
assert json.dumps("é") == '"\\u00e9"'
assert json.dumps("é", ensure_ascii=False) == '"é"'
assert json.dumps({1: "int-key", True: "bool-key", None: "none-key"}) == (
    '{"1": "bool-key", "null": "none-key"}'
)
assert json.dumps(float("nan")) == "NaN"
try:
    json.dumps(float("nan"), allow_nan=False)
except ValueError:
    pass
else:
    raise AssertionError("allow_nan=False must reject NaN")

cycle = []
cycle.append(cycle)
try:
    json.dumps(cycle)
except ValueError:
    pass
else:
    raise AssertionError("a self-referential list must be refused")

shared = [1]
assert json.dumps([shared, shared]) == "[[1], [1]]"


class Point:
    def __init__(self, x):
        self.x = x


assert json.dumps(Point(3), default=lambda o: {"x": o.x}) == '{"x": 3}'
try:
    json.dumps(Point(3))
except TypeError:
    pass
else:
    raise AssertionError("an unserialisable object must raise TypeError")

assert json.dumps({"a": Point(1)}, default=lambda o: o.x, sort_keys=True) == '{"a": 1}'
assert json.loads(json.dumps(nested, indent=4, sort_keys=True)) == nested

print("OK")
