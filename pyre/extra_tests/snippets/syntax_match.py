i = 0
z = 1
match i:
    case 0:
        z = 0
    case 1:
        z = 2
    case _:
        z = 3

assert z == 0
# Test enum
from enum import Enum


class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3


def test_color(color):
    z = -1
    match color:
        case Color.RED:
            z = 1
        case Color.GREEN:
            z = 2
        case Color.BLUE:
            z = 3
    assert z == color.value


for color in Color:
    test_color(color)


# test or
def test_or(i):
    z = -1
    match i:
        case 0 | 1:
            z = 0
        case 2 | 3:
            z = 1
        case _:
            z = 2
    return z


assert test_or(0) == 0
assert test_or(1) == 0
assert test_or(2) == 1
assert test_or(3) == 1
assert test_or(4) == 2

# test mapping
data = {"a": 1, "b": 2}
match data:
    case {"a": x}:
        assert x == 1
    case _:
        assert False

match data:
    case {"a": x, "b": y}:
        assert x == 1, x
        assert y == 2, y
    case _:
        assert False

# test mapping with rest
match data:
    case {"a": x, **rest}:
        assert x == 1
        assert rest == {"b": 2}
    case _:
        assert False

# test empty rest
data2 = {"a": 1}
match data2:
    case {"a": x, **rest}:
        assert x == 1
        assert rest == {}
    case _:
        assert False

# test rest with multiple keys
data3 = {"a": 1, "b": 2, "c": 3, "d": 4}
match data3:
    case {"a": x, "b": y, **rest}:
        assert x == 1
        assert y == 2
        assert rest == {"c": 3, "d": 4}
    case _:
        assert False

match data3:
    case {"a": x, "b": y, "c": z, **rest}:
        assert x == 1
        assert y == 2
        assert z == 3
        assert rest == {"d": 4}
    case _:
        assert False

# test mapping pattern with wildcard fallback (reproduces wheelinfo.py issue)
test_dict = {"sha256": "abc123"}
result = None
match test_dict:
    case {"sha256": checksum}:
        result = checksum
    case _:
        result = "no checksum"
assert result == "abc123"

# test with no match
test_dict2 = {"md5": "xyz789"}
match test_dict2:
    case {"sha256": checksum}:
        result = checksum
    case _:
        result = "no checksum"
assert result == "no checksum"


# test mapping patterns - comprehensive tests
def test_mapping_comprehensive():
    # Single key capture
    data = {"a": 1}
    match data:
        case {"a": x}:
            captured = x
        case _:
            captured = None
    assert captured == 1, f"Expected 1, got {captured}"

    # Multiple keys
    data = {"a": 1, "b": 2}
    match data:
        case {"a": x, "b": y}:
            cap_x = x
            cap_y = y
        case _:
            cap_x = cap_y = None
    assert cap_x == 1, f"Expected x=1, got {cap_x}"
    assert cap_y == 2, f"Expected y=2, got {cap_y}"

    # A dict subclass with __missing__ must not satisfy a key it does not hold.
    # MATCH_KEYS looks keys up with get(key, sentinel), so __missing__ never runs
    # and no entry is fabricated.
    class DefaultMap(dict):
        def __missing__(self, key):
            return f"default-{key}"

    dm = DefaultMap(present=1)
    match dm:
        case {"absent": _}:
            missed = "matched"
        case _:
            missed = "fell-through"
    assert missed == "fell-through", f"Expected fell-through, got {missed}"

    match dm:
        case {"present": v}:
            got = v
        case _:
            got = None
    assert got == 1, f"Expected 1, got {got}"

    # A value-pattern key equal to a literal key is a runtime duplicate. The
    # subject holds enough keys to clear the length guard ahead of MATCH_KEYS.
    class Key:
        name = "a"

    raised = False
    try:
        match {"a": 1, "b": 2}:
            case {Key.name: _, "a": _}:
                pass
    except ValueError as exc:
        raised = "duplicate key" in str(exc)
    assert raised, "Expected ValueError for duplicate mapping key"


test_mapping_comprehensive()
