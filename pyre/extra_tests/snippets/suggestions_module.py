# pyre-check: gate=1
# `_suggestions._generate_suggestions` exposes the misspelling search that
# `traceback._compute_suggestion_error` and `_find_keyword_typos` rank with.
# The search reads every name through `PyUnicode_AsUTF8AndSize` and indexes the
# buffer it gets, so each length and each position it measures is a UTF-8 byte;
# ranking by code points is a different order, not the same one in other units.

import _suggestions

gen = _suggestions._generate_suggestions

assert gen(["hello", "world"], "hell") == "hello"
# A candidate equal to the name is skipped, so it never suggests itself.
assert gen(["hell"], "hell") is None
assert gen(["zzzzzz"], "hell") is None
assert gen([], "hell") is None

# `MAX_CANDIDATE_ITEMS` is 750, and the set is given up on before any name is
# read rather than searched and lost.
assert gen(["hello"] * 749, "hell") == "hello"
assert gen(["hello"] * 750, "hell") is None

# `PyList_CheckExact`: a tuple and a list subclass are both refused.
class MyList(list):
    pass

for wrong_shape in (("hello",), MyList(["hello"])):
    try:
        gen(wrong_shape, "hell")
    except TypeError as error:
        assert str(error) == "candidates must be a list", str(error)
    else:
        raise AssertionError("only an exact list is a candidate set")

try:
    gen(["hello", 1], "hell")
except TypeError as error:
    assert str(error) == "all elements in 'candidates' must be strings", str(error)
else:
    raise AssertionError("a non-string candidate is a TypeError")

# The name is converted by the argument clinic, ahead of the body's list test.
try:
    gen(1, 1)
except TypeError as error:
    assert "argument 2 must be str, not int" in str(error), str(error)
else:
    raise AssertionError("a non-str name is a TypeError")

# A lone surrogate has no UTF-8 encoding, so reading it is what fails -- the
# name is not quietly dropped from the ranking.
for candidates, name in ((["hello", "\udcff"], "hell"), (["hello"], "\udcff")):
    try:
        gen(candidates, name)
    except UnicodeEncodeError as error:
        assert "surrogates not allowed" in str(error), str(error)
    else:
        raise AssertionError("a lone surrogate has no UTF-8 encoding")

# Names outside ASCII are measured in UTF-8 bytes.  Each of these is decided
# differently when the same search counts code points instead.
assert gen(["αaαaa", "日日é"], "αx日é") == "日日é"
assert gen(["日αaa日日", "éaααé"], "aα日é") is None
assert gen(["aaαéαx日", "αax"], "α日x") is None

# `_substitution_cost` charges a case-only difference `CASE_COST` where a real
# substitution costs `MOVE_COST`, so two changes of case beat one of letter.
assert gen(["helLO", "hellp"], "hello") == "helLO"
# The ratio cutoff still applies to those cheap edits: five of them are more
# than a third of the characters involved.
assert gen(["HELLO"], "hello") is None

# `traceback` is the caller this exists for.
try:
    raise AttributeError(name="attrbute", obj=type("C", (), {"attribute": 1})())
except AttributeError as error:
    import traceback

    rendered = "".join(traceback.format_exception_only(error))
    assert "attribute" in rendered, rendered

print("OK")
