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

print("OK")
