"""A character the tokenizer cannot lex names itself in the SyntaxError message.

`pytokenizer.py:130-140` splits on printability: a non-printable character
reports only its code point, a printable one is quoted as well. The hex is
uppercase and padded to at least four digits, so an astral character widens
rather than truncating.

A printable *ASCII* character reports plain `invalid syntax` — `test_syntax.py`
asserts that for `1 $ 2` and keeps `invalid character` for the non-ASCII case.
"""


def msg_of(source):
    try:
        compile(source, "<t>", "exec")
    except SyntaxError as exc:
        return exc.msg
    raise AssertionError(f"{source!r} compiled")


def error_of(source):
    try:
        compile(source, "<t>", "exec")
    except SyntaxError as exc:
        return exc
    raise AssertionError(f"{source!r} compiled")


# Printable ASCII: the message carries no character at all.
for char in "?$`":
    assert msg_of(f"x = {char}") == "invalid syntax", char

# Printable non-ASCII: quoted, with the code point beside it.
NAMED = {
    "£": "invalid character '£' (U+00A3)",
    "€": "invalid character '€' (U+20AC)",
    "☃": "invalid character '☃' (U+2603)",
    "“": "invalid character '“' (U+201C)",
    "？": "invalid character '？' (U+FF1F)",
}
for char, expected in NAMED.items():
    assert msg_of(f"x = {char}") == expected, (char, msg_of(f"x = {char}"))

# An astral character needs five hex digits; four would truncate it to U+F600.
assert msg_of("x = \U0001f600") == "invalid character '\U0001f600' (U+1F600)"

# Non-printable: the code point alone, with no quoted character.
assert msg_of("x = \xa0") == "invalid non-printable character U+00A0"
assert msg_of("x = \x07") == "invalid non-printable character U+0007"

# The location is untouched by the message split.
assert error_of("x = ?").offset == 5
assert error_of("x = ☃").offset == 5
assert error_of("x = ?").lineno == 1

# A non-ASCII identifier still lexes; only unlexable characters take the arms
# above.
scope = {}
exec("α = 1", scope)
assert scope["α"] == 1

print("OK")
