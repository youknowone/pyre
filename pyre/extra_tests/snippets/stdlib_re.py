import re

haystack = "Hello world"
needle = "ello"

mo = re.search(needle, haystack)
print(mo)

# Does not work on python 3.6:
assert isinstance(mo, re.Match)
assert mo.start() == 1
assert mo.end() == 5

assert re.escape("python.exe") == "python\\.exe"

p = re.compile("ab")
s = p.sub("x", "abcabca")
# print(s)
assert s == "xcxca"

idpattern = r"([_a-z][_a-z0-9]*)"

mo = re.search(idpattern, "7382 _boe0+2")
assert mo.group(0) == "_boe0"

# tes op range
assert re.compile("[a-z]").match("a").span() == (0, 1)
assert re.compile("[a-z]").fullmatch("z").span() == (0, 1)

# test op charset
assert re.compile("[_a-z0-9]*").match("_09az").group() == "_09az"

# test op bigcharset
assert re.compile("[你好a-z]*").match("a好z你?").group() == "a好z你"
assert re.compile("[你好a-z]+").search("1232321 a好z你 !!?").group() == "a好z你"

# test op repeat one
assert re.compile("a*").match("aaa").span() == (0, 3)
assert re.compile("abcd*").match("abcdddd").group() == "abcdddd"
assert re.compile("abcd*").match("abc").group() == "abc"
assert re.compile("abcd*e").match("abce").group() == "abce"
assert re.compile("abcd*e+").match("abcddeee").group() == "abcddeee"
assert re.compile("abcd+").match("abcddd").group() == "abcddd"

# test op mark
assert re.compile("(a)b").match("ab").group(0, 1) == ("ab", "a")
assert re.compile("a(b)(cd)").match("abcd").group(0, 1, 2) == ("abcd", "b", "cd")

# test op repeat
assert re.compile("(ab)+").match("abab")
assert re.compile("(a)(b)(cd)*").match("abcdcdcd").group(0, 1, 2, 3) == (
    "abcdcdcd",
    "a",
    "b",
    "cd",
)
assert re.compile("ab()+cd").match("abcd").group() == "abcd"
assert re.compile("(a)+").match("aaa").groups() == ("a",)
assert re.compile("(a+)").match("aaa").groups() == ("aaa",)

# test Match object method
assert re.compile("(a)(bc)").match("abc")[1] == "a"
assert re.compile("a(b)(?P<a>c)d").match("abcd").groupdict() == {"a": "c"}

# Keep later dynamically-created selectors alive while each earlier group
# slice allocates.  Repetition crosses the moving nursery boundary, exercising
# the gateway argument-rooting path rather than only immortal integer indices.
named_match = re.compile("(?P<left>a)(?P<right>bc)").match("abc")
for i in range(4096):
    left = ("left" + str(i))[:4]
    right = ("right" + str(i))[:5]
    assert named_match.group(left, right) == ("a", "bc")

# test op branch
assert re.compile(r"((?=\d|\.\d)(?P<int>\d*)|a)").match("123.2132").group() == "123"

assert re.sub(r"^\s*", "X", "test") == "Xtest"

assert re.match(r"\babc\b", "abc").group() == "abc"

urlpattern = re.compile("//([^/#?]*)(.*)", re.DOTALL)
url = "//www.example.org:80/foo/bar/baz.html"
assert urlpattern.match(url).group(1) == "www.example.org:80"

assert re.compile("(?:\w+(?:\s|/(?!>))*)*").match("a /bb />ccc").group() == "a /bb "
assert re.compile("(?:(1)?)*").match("111").group() == "111"

# Test of fix re.fullmatch POSSESSIVE_REPEAT, issue #7183
assert re.fullmatch(r"([0-9]++(?:\.[0-9]+)*+)", "1.25.38")
assert re.fullmatch(r"([0-9]++(?:\.[0-9]+)*+)", "1.25.38").group(0) == "1.25.38"

# Combining characters; issue #7518
assert not re.match(r"\w", "\u0345"), r"\w should not match U+0345 (category Mn)"


# A group selector is taken as a number when its type has `__index__`, and as a
# name otherwise, so a duck-typed operand reaches every span accessor.
class DuckIndex:
    def __index__(self):
        return 1


mo = re.compile("(a+)(b+)").match("aabb")
assert mo.group(DuckIndex()) == "aa"
assert mo[DuckIndex()] == "aa"
assert mo.start(DuckIndex()) == 0
assert mo.end(DuckIndex()) == 2
assert mo.span(DuckIndex()) == (0, 2)
assert mo.group(DuckIndex(), DuckIndex()) == ("aa", "aa")

mo = re.compile("(?P<first>a+)(?P<second>b+)").match("aabb")
assert mo.group("second") == "bb"
