//! The small entry points an extension reaches for in passing: the unqualified
//! type name, the repr recursion guard, the locale codec, a string built from
//! code points, the buffer hash, `setdefault` and `origin[args]`.
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
import cpyext_small as m


def eq(name, got, want):
    assert got == want, '%s: got %r, want %r' % (name, got, want)


class Named:
    pass


class Boom:
    def __hash__(self):
        raise ValueError('boom')


# ── the unqualified type name ──────────────────────────────────────────

eq('names int', m.type_names(int), ('int', 'int'))
eq('names class', m.type_names(Named), ('Named', 'Named'))
eq('names module', m.type_names(type(m)), ('module', 'module'))

# ── the repr recursion guard ───────────────────────────────────────────

# The entry is recorded, so the second ask says the object is already being
# rendered; the third is after the leave and is the first ask again.
value = []
eq('guard list', m.repr_guard(value), (0, 1, 0))
eq('guard list again', m.repr_guard(value), (0, 1, 0))
eq('guard tuple', m.repr_guard((1, 2)), (0, 1, 0))

# The set is the one the interpreter's own containers consult, so a `tp_repr`
# that entered a list and then asks for its repr gets the elision rather than
# the body.
eq('guarded repr', m.guarded_repr([1, 2]), '[...]')
loop = []
loop.append(loop)
eq('guarded repr cycle', m.guarded_repr(loop), '[...]')

# ── the locale codec ───────────────────────────────────────────────────

DECODE_ERROR = ('UnicodeDecodeError',
                "'locale' codec can't decode byte 0xff in position 1: decoding error")
UNSUPPORTED = ('ValueError', 'unsupported error handler')

eq('decode ascii', m.decode_locale(b'abc', None), ('abc', None))
eq('decode utf8', m.decode_locale(b'a\xc3\xa9z', None), ('a\xe9z', None))
eq('decode empty', m.decode_locale(b'', None), ('', None))
eq('decode undecodable', m.decode_locale(b'a\xffz', None), DECODE_ERROR)
eq('decode strict named', m.decode_locale(b'a\xffz', 'strict'), DECODE_ERROR)
# The byte with no text spelling comes back as the surrogate escape that
# re-encodes to itself.
eq('decode surrogateescape', m.decode_locale(b'a\xffz', 'surrogateescape'),
   ('a\udcffz', None))
# Only those two handlers: the conversion refuses the rest before it runs
# rather than reaching the codec registry.
eq('decode replace', m.decode_locale(b'abc', 'replace'), UNSUPPORTED)
# The block is read to its first NUL, so one before the end has nowhere to go.
eq('decode embedded nul', m.decode_locale(b'a\x00b', None),
   ('ValueError', 'embedded null byte'))
eq('decode str', m.decode_locale_str(b'abc', None), ('abc', None))

ENCODE_ERROR = ('UnicodeEncodeError',
                "'locale' codec can't encode character '\\udcff' in position 1: encoding error")

eq('encode ascii', m.encode_locale('abc', None), (b'abc', None))
eq('encode wide', m.encode_locale('a\xe9z', None), (b'a\xc3\xa9z', None))
eq('encode astral', m.encode_locale('a\U0001F600z', None),
   (b'a\xf0\x9f\x98\x80z', None))
eq('encode empty', m.encode_locale('', None), (b'', None))
eq('encode surrogate strict', m.encode_locale('a\udcffz', None), ENCODE_ERROR)
eq('encode surrogate escape', m.encode_locale('a\udcffz', 'surrogateescape'),
   (b'a\xffz', None))
# A surrogate outside the escape range stands for no byte, so it has no
# spelling under either handler.
eq('encode lone surrogate', m.encode_locale('a\ud800z', 'surrogateescape'),
   ('UnicodeEncodeError',
    "'locale' codec can't encode character '\\ud800' in position 1: encoding error"))
eq('encode nul', m.encode_locale('a\x00b', None),
   ('ValueError', 'embedded null character'))
eq('encode replace', m.encode_locale('abc', 'replace'), UNSUPPORTED)
eq('encode nonstr', m.encode_locale(42, None),
   ('TypeError', 'bad argument type for built-in operation'))

# ── a string from code points ──────────────────────────────────────────

# The units are code points rather than an encoding, so each width reads its
# array as it stands and a surrogate is one character.
eq('kind 1', m.from_kind(1, [104, 105]), ('hi', None))
eq('kind 1 high', m.from_kind(1, [0xE9]), ('\xe9', None))
eq('kind 1 empty', m.from_kind(1, []), ('', None))
eq('kind 2', m.from_kind(2, [104, 0x4E00]), ('h一', None))
eq('kind 2 surrogate', m.from_kind(2, [0xD800]), ('\ud800', None))
eq('kind 4', m.from_kind(4, [104, 0x1F600]), ('h\U0001F600', None))
eq('kind 4 nul', m.from_kind(4, [104, 0, 105]), ('h\x00i', None))
eq('kind 4 empty', m.from_kind(4, []), ('', None))
eq('kind 0', m.from_kind_bad(0, 1), ('SystemError', 'invalid kind'))
eq('kind 3', m.from_kind_bad(3, 1), ('SystemError', 'invalid kind'))
eq('kind negative size', m.from_kind_bad(4, -1),
   ('ValueError', 'size must be positive'))

# ── the buffer hash ────────────────────────────────────────────────────

for data in [b'', b'a', b'hello world', bytes(range(64))]:
    eq('hash %r' % data[:8], m.hash_buffer(data), hash(data))

# ── setdefault ─────────────────────────────────────────────────────────

mapping = {'a': 1}
eq('setdefault present', m.set_default(mapping, 'a', 99), (1, 1, None))
eq('setdefault absent', m.set_default(mapping, 'b', 2), (0, 2, None))
eq('setdefault inserted', mapping, {'a': 1, 'b': 2})
# A NULL result is the caller wanting the insertion and not the value.
eq('setdefault no result', m.set_default_no_result(mapping, 'c', 3), (0, None))
eq('setdefault no result again', m.set_default_no_result(mapping, 'c', 4), (1, None))
eq('setdefault inserted 2', mapping, {'a': 1, 'b': 2, 'c': 3})

# Nothing is handed back when the call fails, whatever the failure was.
unhashable = m.set_default(mapping, [], 1)
eq('setdefault unhashable', (unhashable[0], unhashable[1], unhashable[2][0]),
   (-1, None, 'TypeError'))
raising = m.set_default(mapping, Boom(), 1)
eq('setdefault hash raises', raising, (-1, None, ('ValueError', 'boom')))
# Only the code and the empty result: what is not a dict is refused the way
# every entry point in the family refuses it, which names the function where
# CPython reports a bad internal call.
refused = m.set_default([], 'a', 1)
eq('setdefault not a dict', (refused[0], refused[1]), (-1, None))
eq('setdefault untouched', mapping, {'a': 1, 'b': 2, 'c': 3})

# ── origin[args] ───────────────────────────────────────────────────────

alias, error = m.generic_alias(list, int)
eq('alias error', error, None)
eq('alias str', str(alias), 'list[int]')
eq('alias origin', alias.__origin__ is list, True)
eq('alias args', alias.__args__, (int,))
eq('alias equal', alias == list[int], True)
# A tuple is the whole argument list rather than one argument that is a tuple.
pair, error = m.generic_alias(dict, (str, int))
eq('alias pair error', error, None)
eq('alias pair str', str(pair), 'dict[str, int]')
eq('alias pair args', pair.__args__, (str, int))

print('cpyext-small-ok')
"#;

#[test]
fn the_small_entry_points() {
    let fixtures = Fixtures::new("cpyext-small");
    fixtures.compile("cpyext_small");
    fixtures.expect_ok(SCRIPT, &[], "cpyext-small-ok");
}
