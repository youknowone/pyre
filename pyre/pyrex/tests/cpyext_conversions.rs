//! The conversions an extension does at its C boundary: the named codecs, the
//! `wchar_t` forms, the filesystem encoding, and the small entry points beside
//! them.
//!
//! Every expectation was taken from CPython 3.14.6 running this same script
//! against this same fixture.

#![cfg(all(
    feature = "cpyext",
    not(feature = "sandbox"),
    any(target_os = "macos", target_os = "linux")
))]

mod cpyext_fixture;

use cpyext_fixture::Fixtures;

const SCRIPT: &str = r#"
import cpyext_conversions as m


def eq(name, got, want):
    assert got == want, '%s: got %r, want %r' % (name, got, want)


class Indexable:
    def __index__(self):
        return 3


class Path:
    def __init__(self, value):
        self.value = value

    def __fspath__(self):
        return self.value


class BadPath:
    def __fspath__(self):
        return 42


# ── the named codecs ───────────────────────────────────────────────────

eq('ascii ok', m.decode_ascii(b'abc', None), ('abc', None))
eq('ascii empty', m.decode_ascii(b'', None), ('', None))
eq('ascii high', m.decode_ascii(b'a\xffz', None),
   ('UnicodeDecodeError',
    "'ascii' codec can't decode byte 0xff in position 1: ordinal not in range(128)"))
eq('ascii replace', m.decode_ascii(b'a\xffz', 'replace'), ('a�z', None))
eq('ascii ignore', m.decode_ascii(b'a\xffz', 'ignore'), ('az', None))
eq('ascii surrogate', m.decode_ascii(b'a\xffz', 'surrogateescape'), ('a\udcffz', None))
eq('ascii bogus handler', m.decode_ascii(b'a\xffz', 'nosuch'),
   ('LookupError', "unknown error handler name 'nosuch'"))
eq('latin1 ok', m.decode_latin1(b'a\xffz', None), ('a\xffz', None))
eq('latin1 empty', m.decode_latin1(b'', None), ('', None))
eq('named utf8', m.decode_named(b'\xc3\xa9', 'utf-8', None), ('\xe9', None))
eq('named utf16', m.decode_named(b'\xff\xfeA\x00', 'utf-16', None), ('A', None))
eq('named bad', m.decode_named(b'\xff', 'utf-8', None),
   ('UnicodeDecodeError',
    "'utf-8' codec can't decode byte 0xff in position 0: invalid start byte"))
eq('named unknown', m.decode_named(b'ab', 'nosuchcodec', None),
   ('LookupError', 'unknown encoding: nosuchcodec'))
# A codec that does not answer with `str` is refused by the codec machinery
# rather than by the entry point.
eq('named not text', m.decode_named(b'ab', 'hex_codec', None),
   ('LookupError',
    "'hex_codec' is not a text encoding; use codecs.decode() to handle arbitrary codecs"))

eq('as ascii', m.as_ascii('abc'), (b'abc', None))
eq('as ascii high', m.as_ascii('a\xe9z'),
   ('UnicodeEncodeError',
    "'ascii' codec can't encode character '\\xe9' in position 1: ordinal not in range(128)"))
eq('as latin1', m.as_latin1('a\xe9z'), (b'a\xe9z', None))
eq('as latin1 wide', m.as_latin1('a一z'),
   ('UnicodeEncodeError',
    "'latin-1' codec can't encode character '\\u4e00' in position 1: ordinal not in range(256)"))
eq('as utf8', m.as_utf8_string('a一z'), (b'a\xe4\xb8\x80z', None))
eq('as utf8 surrogate', m.as_utf8_string('a\ud800z'),
   ('UnicodeEncodeError',
    "'utf-8' codec can't encode character '\\ud800' in position 1: surrogates not allowed"))
eq('as encoded', m.as_encoded('a\xe9z', 'latin-1', None), (b'a\xe9z', None))
eq('as encoded errors', m.as_encoded('a一z', 'ascii', 'replace'), (b'a?z', None))
eq('as encoded unknown', m.as_encoded('ab', 'nosuchcodec', None),
   ('LookupError', 'unknown encoding: nosuchcodec'))
# Every encode spelling wants a `str`, and says so the way `PyErr_BadArgument`
# does rather than by naming the entry point.
BAD = ('TypeError', 'bad argument type for built-in operation')
eq('as ascii nonstr', m.as_ascii(b'abc'), BAD)
eq('as utf8 nonstr', m.as_utf8_string(42), BAD)
eq('as encoded nonstr', m.as_encoded(42, 'ascii', None), BAD)

# A subclass is still a `str`, and what these encode is the string: they
# reach the codec directly, so an `encode` of its own is never looked up.


class Shouty(str):
    def encode(self, *arguments, **keywords):
        raise AssertionError('encode() was looked up')


eq('as ascii subclass', m.as_ascii(Shouty('abc')), (b'abc', None))
eq('as utf8 subclass', m.as_utf8_string(Shouty('a一z')), (b'a\xe4\xb8\x80z', None))
eq('as latin1 subclass', m.as_latin1(Shouty('a\xe9z')), (b'a\xe9z', None))
eq('as encoded subclass', m.as_encoded(Shouty('a一z'), 'ascii', 'replace'), (b'a?z', None))

# ── the `wchar_t` forms ────────────────────────────────────────────────

eq('from wide nul terminated', m.from_wide([104, 105], -1), ('hi', None))
eq('from wide sized', m.from_wide([104, 105], 2), ('hi', None))
eq('from wide short', m.from_wide([104, 105], 1), ('h', None))
eq('from wide empty', m.from_wide([], 0), ('', None))
# A size given outright takes the units as they are, a NUL among them.
eq('from wide embedded nul', m.from_wide([104, 0, 105], 3), ('h\x00i', None))
eq('from wide surrogate', m.from_wide([0xD800], -1), ('\ud800', None))
if m.WCHAR_SIZE == 4:
    eq('from wide astral', m.from_wide([0x1F600], -1), ('\U0001F600', None))

# The buffer takes the trailing NUL only when there is room for it; the count
# answered is the length either way.
eq('as wide exact', m.as_wide('hi', 2), (2, [104, 105]))
eq('as wide room', m.as_wide('hi', 4), (2, [104, 105, 0, 0x7F7F7F7F]))
eq('as wide short', m.as_wide('hi', 1), (1, [104]))
eq('as wide zero', m.as_wide('hi', 0), (0, []))
eq('as wide nonstr', m.as_wide(42, 4), BAD)

eq('as wide string', m.as_wide_string('hi', True), (2, [104, 105]))
# Without a length beside it the block is read to its first NUL, so a string
# holding one has nowhere to be put.
eq('as wide string nul', m.as_wide_string('a\x00b', True), (3, [97]))
eq('as wide string nul unsized', m.as_wide_string('a\x00b', False),
   ('ValueError', 'embedded null character'))
eq('as wide string nonstr', m.as_wide_string(42, True), BAD)

# ── the filesystem encoding ────────────────────────────────────────────

eq('decode fs', m.decode_fs(b'/a/b'), ('/a/b', None))
# A byte with no text spelling comes back as the surrogate escape that
# re-encodes to itself.
eq('decode fs undecodable', m.decode_fs(b'/a/\xff'), ('/a/\udcff', None))
eq('decode fs size', m.decode_fs_size(b'/a/b'), ('/a/b', None))
eq('decode fs size nul', m.decode_fs_size(b'a\x00b'), ('a\x00b', None))
eq('encode fs', m.encode_fs('/a/b'), (b'/a/b', None))
eq('encode fs surrogate', m.encode_fs('/a/\udcff'), (b'/a/\xff', None))
eq('encode fs bytes', m.encode_fs(b'/a/b'), BAD)
eq('encode fs int', m.encode_fs(42), BAD)

NOT_A_PATH = ('TypeError', 'expected str, bytes or os.PathLike object, not int')
eq('fs converter str', m.fs_converter('/a/b'), (b'/a/b', None))
eq('fs converter bytes', m.fs_converter(b'/a/b'), (b'/a/b', None))
eq('fs converter surrogate', m.fs_converter('/a/\udcff'), (b'/a/\xff', None))
eq('fs converter path', m.fs_converter(Path('/a/b')), (b'/a/b', None))
eq('fs converter nul', m.fs_converter('a\x00b'), ('ValueError', 'embedded null byte'))
eq('fs converter int', m.fs_converter(42), NOT_A_PATH)
eq('fs converter bad path', m.fs_converter(BadPath()),
   ('TypeError', 'expected BadPath.__fspath__() to return str or bytes, not int'))
eq('fs decoder str', m.fs_decoder('/a/b'), ('/a/b', None))
eq('fs decoder bytes', m.fs_decoder(b'/a/b'), ('/a/b', None))
eq('fs decoder undecodable', m.fs_decoder(b'/a/\xff'), ('/a/\udcff', None))
eq('fs decoder path', m.fs_decoder(Path(b'/a/b')), ('/a/b', None))
eq('fs decoder nul', m.fs_decoder(b'a\x00b'), ('ValueError', 'embedded null character'))
eq('fs decoder int', m.fs_decoder(42), NOT_A_PATH)

# ── the small entry points ─────────────────────────────────────────────

eq('index int', m.index_check(5), 1)
eq('index bool', m.index_check(True), 1)
eq('index float', m.index_check(1.5), 0)
eq('index str', m.index_check('a'), 0)
eq('index custom', m.index_check(Indexable()), 1)
# The lookup is on the type, so a class defining `__index__` for its instances
# does not answer for itself.
eq('index class', m.index_check(Indexable), 0)

CONSTANTS = [None, False, True, Ellipsis, NotImplemented, 0, 1, '', b'', ()]
for identifier, value in enumerate(CONSTANTS):
    eq('constant %d' % identifier, m.get_constant(identifier, False), (value, None))
    eq('constant %d borrowed' % identifier, m.get_constant(identifier, True), (value, None))
# The same object every time, which is what lets the borrowed spelling exist.
eq('constant identity',
   m.get_constant(9, True)[0] is m.get_constant(9, False)[0], True)
# Only the class: the message names the file and line of the call that made
# the mistake, which no two implementations agree on.
eq('constant out of range', m.get_constant(10, False)[0], 'SystemError')
eq('constant out of range borrowed', m.get_constant(10, True)[0], 'SystemError')

# ── PyOS_snprintf ──────────────────────────────────────────────────────

# 13 either way: the answer is the length the conversion needed, whether or
# not the buffer had room, and `str[size - 1]` is a NUL whatever happened.
eq('snprintf room', m.os_snprintf(32),
   (13, b'left-42-right\x00' + b'x' * 17 + b'\x00'))
eq('snprintf exact', m.os_snprintf(14), (13, b'left-42-right\x00'))
eq('snprintf truncated', m.os_snprintf(13), (13, b'left-42-righ\x00'))
eq('snprintf short', m.os_snprintf(5), (13, b'left\x00'))
eq('snprintf one', m.os_snprintf(1), (13, b'\x00'))

# ── PyArg_Parse ────────────────────────────────────────────────────────

# The argument itself, not a tuple holding it.
eq('parse one', m.arg_parse(7, 'i'), (1, 7, None))
eq('parse str', m.arg_parse('hi', 's'), (1, 'hi', None))
wrong = m.arg_parse('x', 'i')
eq('parse wrong type', (wrong[0], wrong[1], wrong[2][0]), (0, -1, 'TypeError'))
# A parenthesised unit is one argument, so it is the one shape that can carry
# more than one value.
eq('parse nested', m.arg_parse((1, 2), '(ii)'), (1, (1, 2), None))
# More than one unit has nowhere to take the second from.
eq('parse two units', m.arg_parse((1, 2), 'ii'),
   (0, (-1, -1), ('SystemError', 'old style getargs format uses new features')))

# ── the nested unit through the ordinary parser ────────────────────────

eq('nested pair', m.parse_nested((1, 2), '(ii)'), (1, (1, 2), None))
# Any sequence of the right length, `str` excepted.
eq('nested list', m.parse_nested([1, 2], '(ii)'), (1, (1, 2), None))
eq('nested then unit', m.parse_nested((7,), '(i)i'), (1, (7, -1), None))
# Only the class and the values: the two parsers word a bad argument
# differently, which is the argument parser's own question.
for name, argument in [('short', (1,)), ('long', (1, 2, 3)), ('str', 'ab'),
                       ('int', 5), ('none', None)]:
    answer = m.parse_nested(argument, '(ii)')
    eq('nested %s' % name, (answer[0], answer[1], answer[2][0]),
       (0, (-1, -1), 'TypeError'))

print('cpyext-conversions-ok')
"#;

#[test]
fn the_conversion_entry_points() {
    let fixtures = Fixtures::new("cpyext-conversions");
    fixtures.compile("cpyext_conversions");
    fixtures.expect_ok(SCRIPT, &[], "cpyext-conversions-ok");
}
