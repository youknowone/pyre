# cpyext-fixture: cpyext_str
# cpyext-expect: cpyext-format-ok

# The `str` entry points, and the `%`-format engine three of them share.
#
# Every expectation was taken from CPython 3.14.6 running this same script
# against this same fixture, except where noted: `%p` answers with the
# platform's own spelling of a pointer, and two entry points read an argument
# CPython never checks, which is not behaviour to match.

import cpyext_str as m

class Thing:
    def __str__(self): return 'sté'
    def __repr__(self): return 'repé'

OVERFLOW = ('OverflowError', 'character argument not in range(0x110000)')
LONG = '0123456789abcdef' * 32

EXPECTED = {
    'literal': 'plain',
    '%%': '100%',

    # A code point, so one above 127 is a character and not a byte.
    '%c ascii': '[A]',
    '%c latin': '[é]',
    '%c astral': '[😀]',
    '%c twice': '[AB]',
    '%c negative': OVERFLOW,
    '%c too big': OVERFLOW,

    '%d': '[-42]',
    '%i': '[42]',
    '%u': '[42]',
    '%ld': '[-42]',
    '%lu': '[42]',
    '%lld': '[-42]',
    '%llu': '[42]',
    '%zd': '[-42]',
    '%zu': '[42]',
    '%x': '[ff]',
    '%X': '[FF]',
    '%o': '[10]',
    '%5d': '[   42]',
    '%-5d': '[42   ]',
    '%05d': '[00042]',

    '%s': '[text]',
    '%s utf8': '[été]',
    # Text C hands over need not be valid UTF-8; what is not becomes U+FFFD.
    '%s invalid': '[bad��utf8]',
    '%.2s': '[ab]',
    # The precision bounds the bytes read, so it stops inside a character.
    '%.2s utf8': '[é]',
    '%.0s': '[]',
    '%10s': '[        ab]',
    # The width counts characters, so the padding is three and not one.
    '%6s utf8': '[   été]',

    '%S': '[sté]',
    '%R': '[repé]',
    # `ascii()` rather than `repr()`, which is what tells the two apart.
    '%A': "[rep\\xe9]",
    '%.2S': '[st]',
    '%6S': '[   sté]',
    '%-6S': '[sté   ]',
    '%U': '[abcdef]',
    '%.3U': '[abc]',
    '%V': '[abcdef]',
    '%V null': '[fallback]',

    'two': 'n=7',
    # Longer than any fixed buffer the engine might start with.
    'long': LONG,

    'unknown code': ('SystemError', 'invalid format string: %q]'),
    # The float conversions are not among the ones this describes.
    'float code': ('SystemError', 'invalid format string: %.2f]'),
    'trailing': ('SystemError', 'invalid format string: %'),
    'non-ascii format': (
        'ValueError',
        'PyUnicode_FromFormatV() expects an ASCII-encoded format string, '
        'got a non-ASCII byte: 0xc3'),
}

rows = dict(m.format_rows(Thing()))
missing = set(EXPECTED) - set(rows)
assert not missing, 'no row for %s' % sorted(missing)
extra = set(rows) - set(EXPECTED)
assert not extra, 'no expectation for %s' % sorted(extra)
for name, want in EXPECTED.items():
    assert rows[name] == want, '%s: got %r, want %r' % (name, rows[name], want)

# `%T` names an object's type and `%N` a type. Neither the module a class was
# defined in nor `builtins` is named, so the two spellings agree here.
assert m.format_type(Thing()) == ('[Thing]', '[Thing]'), m.format_type(Thing())
assert m.format_type(3) == ('[int]', '[int]'), m.format_type(3)
assert m.format_type_name(Thing) == '[Thing]', m.format_type_name(Thing)
assert m.format_type_name(3) == ('TypeError', '%N argument must be a type')

# `%p` is the one conversion whose digits are the platform's; only the prefix
# it is guaranteed to carry is compared.
assert m.format_pointer().startswith('0x'), m.format_pointer()

# The engine reached through the error path, with a pending exception it has
# to drop before running the argument's `__str__`.
assert m.format_error(Thing()) == ('ValueError', 'n=7 sté Z'), m.format_error(Thing())

print('cpyext-format-ok')
