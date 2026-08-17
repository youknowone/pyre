//! The `str` entry points, and the `%`-format engine three of them share.
//!
//! Every expectation was taken from CPython 3.14.6 running this same script
//! against this same fixture, except where noted: `%p` answers with the
//! platform's own spelling of a pointer, and two entry points read an argument
//! CPython never checks, which is not behaviour to match.

#![cfg(all(
    feature = "cpyext",
    not(feature = "sandbox"),
    any(target_os = "macos", target_os = "linux")
))]

mod cpyext_fixture;

use cpyext_fixture::Fixtures;

const FORMAT_SCRIPT: &str = r#"
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
"#;

const ENTRY_POINT_SCRIPT: &str = r#"
import cpyext_str as m

class S(str):
    pass

def eq(name, got, want):
    assert got == want, '%s: got %r, want %r' % (name, got, want)

eq('concat', m.str_concat('ab', 'cd'), 'abcd')
eq('concat non-str', m.str_concat('ab', 3),
   ('TypeError', 'can only concatenate str (not "int") to str'))
eq('append', m.str_append('ab', 'cd'), 'abcd')
eq('append_and_del', m.str_append_and_del('ab', 'cd'), 'abcd')

eq('substring', m.str_substring('abcdef', 1, 4), 'bcd')
eq('substring clamp', m.str_substring('abc', 1, 99), 'bc')
eq('substring empty', m.str_substring('abc', 2, 1), '')
eq('substring negative', m.str_substring('abc', -1, 2),
   ('IndexError', 'string index out of range'))

eq('join', m.str_join('-', ['a', 'b', 'c']), 'a-b-c')
eq('join non-str item', m.str_join('-', ['a', 2]),
   ('TypeError', 'sequence item 1: expected str instance, int found'))

eq('findchar forward', m.str_find_char('abcabc', ord('b'), 0, 6, 1), 1)
eq('findchar backward', m.str_find_char('abcabc', ord('b'), 0, 6, -1), 4)
eq('findchar absent', m.str_find_char('abc', ord('z'), 0, 3, 1), -1)
eq('findchar window', m.str_find_char('abcabc', ord('b'), 2, 6, 1), 4)

eq('contains', m.str_contains('abc', 'b'), True)
eq('contains absent', m.str_contains('abc', 'z'), False)
eq('contains non-str', m.str_contains('abc', 3),
   ('TypeError', "'in <string>' requires string as left operand, not int"))

eq('compare less', m.str_compare('a', 'b'), -1)
eq('compare equal', m.str_compare('a', 'a'), 0)
eq('compare greater', m.str_compare('b', 'a'), 1)
eq('compare non-str', m.str_compare('a', 3), ('TypeError', "Can't compare str and int"))
# The C string is read as ISO-8859-1, so it is a comparison of code points
# against bytes and never fails.
eq('compare ascii equal', m.str_compare_ascii('abc', 'abc'), 0)
eq('compare ascii less', m.str_compare_ascii('abc', 'abd'), -1)
eq('compare ascii shorter', m.str_compare_ascii('ab', 'abc'), -1)
eq('compare ascii longer', m.str_compare_ascii('abcd', 'abc'), 1)

eq('rich equal', m.str_rich_compare('a', 'a', 2), True)
eq('rich less', m.str_rich_compare('a', 'b', 0), True)
eq('equal', m.str_equal('ab', 'ab'), True)
eq('equal no', m.str_equal('ab', 'ac'), False)
eq('equal utf8', m.str_equal_utf8('été', 'été'.encode()), (True, True))
eq('equal utf8 no', m.str_equal_utf8('été', b'nope'), (False, False))

eq('ordinal', m.str_from_ordinal(0x1f600), '😀')
eq('ordinal ascii', m.str_from_ordinal(65), 'A')
eq('ordinal too big', m.str_from_ordinal(0x110000),
   ('ValueError', 'chr() arg not in range(0x110000)'))
eq('ordinal negative', m.str_from_ordinal(-1),
   ('ValueError', 'chr() arg not in range(0x110000)'))

# An exact str is answered with itself; a subclass instance is copied.
eq('from_object exact', m.str_from_object('abc'), ('abc', 'str', True))
eq('from_object subclass', m.str_from_object(S('abc')), ('abc', 'str', False))
eq('from_object other', m.str_from_object(3),
   ('TypeError', "Can't convert 'int' object to str implicitly"))

eq('intern', m.str_intern('a-name-nobody-else-uses'), ('a-name-nobody-else-uses', True))
eq('intern in place', m.str_intern_in_place('another-name-nobody-uses'),
   ('another-name-nobody-uses', True))

# The error handler is the interpreter's own, so every one it has is reachable.
eq('decode strict', m.str_decode_utf8(b'\xc3\xa9', None), 'é')
eq('decode strict invalid', m.str_decode_utf8(b'\xff', None),
   ('UnicodeDecodeError',
    "'utf-8' codec can't decode byte 0xff in position 0: invalid start byte"))
eq('decode replace', m.str_decode_utf8(b'a\xffb', 'replace'), 'a�b')
eq('decode ignore', m.str_decode_utf8(b'a\xffb', 'ignore'), 'ab')

print('cpyext-str-ok')
"#;

#[test]
fn every_format_conversion() {
    let fixtures = Fixtures::new("cpyext-format");
    fixtures.compile("cpyext_str");
    fixtures.expect_ok(FORMAT_SCRIPT, &[], "cpyext-format-ok");
}

#[test]
fn the_str_entry_points() {
    let fixtures = Fixtures::new("cpyext-str");
    fixtures.compile("cpyext_str");
    fixtures.expect_ok(ENTRY_POINT_SCRIPT, &[], "cpyext-str-ok");
}
