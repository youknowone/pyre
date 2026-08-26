# pyre-check: gate=1
# Decoding an escape the syntax does not define reports it once, naming the
# sequence as a literal of the type being produced.  Only the first such
# escape in an input is reported.

import codecs
import warnings

def decoded(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn()
    return result, [(w.category, str(w.message)) for w in caught]

SUFFIX = "Such sequences will not work in the future. "

for source, text, sequence, octal in (
    (br"\z", "\\z", "z", False),
    (br"\8", "\\8", "8", False),
    (br"\501", "Ł", "501", True),
    (br"\777", "ǿ", "777", True),
):
    kind = "an invalid octal escape sequence" if octal else "an invalid escape sequence"

    result, caught = decoded(lambda s=source: s.decode("unicode-escape"))
    assert result == text, (source, result)
    assert len(caught) == 1, (source, caught)
    category, message = caught[0]
    assert category is DeprecationWarning, category
    assert message == f'"\\{sequence}" is {kind}. {SUFFIX}', message

    result, caught = decoded(lambda s=source: codecs.escape_decode(s))
    assert len(caught) == 1, (source, caught)
    category, message = caught[0]
    assert category is DeprecationWarning, category
    # The bytes transform names the sequence as a bytes literal.
    assert message == f'b"\\{sequence}" is {kind}. {SUFFIX}', message

# A defined escape says nothing, and `\377` still fits a byte.
for source in (br"\n", br"\x41", br"\377", br"\\z"):
    for decode in (lambda s: s.decode("unicode-escape"), codecs.escape_decode):
        _, caught = decoded(lambda s=source, d=decode: d(s))
        assert caught == [], (source, caught)

# Only the first offending escape is reported.
_, caught = decoded(lambda: br"\z\q\w".decode("unicode-escape"))
assert len(caught) == 1, caught
assert '"\\z"' in caught[0][1], caught[0][1]
