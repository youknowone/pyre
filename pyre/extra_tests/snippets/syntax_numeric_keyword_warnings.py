# pyre-check: gate=1
"""Tokenizer SyntaxWarnings inside 3.14 interpolated-string fields."""

import warnings


def warnings_for(source):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            compile(source, "<numeric-keyword>", "eval")
        except SyntaxError:
            pass
    return [(item.category, str(item.message)) for item in caught]


for source in (
    'f"{1or 0}"',
    'rf"\\{1or 0}"',
    "f'''{ # !\n1or 0\n}'''",
    "f'''{ # :\n1or 0\n}'''",
):
    assert warnings_for(source) == [(SyntaxWarning, "invalid decimal literal")], source

for source in ('ff"{1or 0}"', 'ft"{1or 0}"'):
    assert warnings_for(source) == [], source

# `#` in a format spec is literal text, not a PEP 701 expression comment; the
# ordinary string after the field must therefore remain ordinary string text.
assert warnings_for('f"{1:#x}" + "{2or 0}"') == []

# A backslash does not stop `{` opening a replacement field.  The literal
# still gets its independent invalid-escape warning, followed by the numeric
# token warning from the field expression.
backslash_field_warnings = warnings_for("f'\\{1or 0}'")
assert len(backslash_field_warnings) == 2
assert set(backslash_field_warnings) == {
    (
        SyntaxWarning,
        '"\\{" is an invalid escape sequence. Such sequences will not work in '
        'the future. Did you mean "\\\\{"? A raw string is also an option.',
    ),
    (SyntaxWarning, "invalid decimal literal"),
}

print("OK")
