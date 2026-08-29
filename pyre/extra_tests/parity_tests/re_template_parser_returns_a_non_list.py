# pyre-check: pypy-diverges: pypy3's `_sre` reads the parser's result
# positionally (`literals[index]`) rather than checking its type, so the same
# program raises `IndexError: tuple index out of range` there instead of the
# argument `TypeError`.  The exception's identity is what this pins, so the
# divergence fails the fixture rather than being expressible in it.
#
# CPython-suite gap: `test_re` covers `re.sub` template expansion thoroughly
# and `test_re.test_bug_2537` and neighbours reach `_sre.template` directly,
# but nothing in the suite replaces `re._parser.parse_template` -- the suite
# treats the parser as part of the implementation rather than as a name a
# program can rebind.
#
# parity-tests reason: `re.sub` reaches its template parser by attribute on an
# ordinary imported module, so the result is a program value.  `_sre.template`
# is where upstream refuses one of the wrong type, and a runtime that instead
# reads the list header off whatever came back has no check between a tuple and
# a length.
import re

try:
    re._parser.parse_template = lambda template, pattern: ()
    try:
        re.sub("a", r"\g<0>", "a")
    except TypeError as exc:
        assert "must be list" in str(exc), exc
    else:
        raise AssertionError("a non-list template result was accepted")
finally:
    del re._parser.parse_template

print("OK")
