//! The `str` an extension builds piece by piece, and the container entry
//! points that answer whether the key was there beside the value.
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
import cpyext_writer as m


def eq(name, got, want):
    assert got == want, '%s: got %r, want %r' % (name, got, want)


class Missing:
    def __getitem__(self, key):
        raise KeyError(key)


class Angry:
    def __getitem__(self, key):
        raise RuntimeError('no')


class NoText:
    def __str__(self):
        raise RuntimeError('no str')

    def __repr__(self):
        raise RuntimeError('no repr')


class Unhashable:
    __hash__ = None


# ── the str an extension builds piece by piece ─────────────────────────

# Every write in one pass: ASCII bytes, UTF-8 bytes, one code point, an
# array of them, wide characters, an object as text and as its repr, and a
# formatted piece.
eq('write all', m.write_all('ab'),
   ("[olé☃a\U0001f600bA☃ab'ab'<fmt 7 v>]", None))

# A writer given up answers nothing and leaves nothing behind.
eq('discard', m.write_discard(), (0, None))

# The length is a reservation, so there is no negative one to reserve.
eq('bad create', m.write_bad_create(),
   ('ValueError', 'length must be positive'))

# What a refused write leaves behind: the writer still holds what it held
# before, so the caller may still finish it.
eq('refuse char', m.write_refusals('char', '', 0, 0),
   (-1, ('ValueError', 'character must be in range(0x110000)'), 'keep'))
eq('refuse utf8', m.write_refusals('utf8', '', 0, 0),
   (-1, ('UnicodeDecodeError',
         "'utf-8' codec can't decode byte 0xff in position 1: "
         'invalid start byte'), 'keep'))
eq('refuse utf8 cut short', m.write_refusals('utf8_short', '', 0, 0),
   (-1, ('UnicodeDecodeError',
         "'utf-8' codec can't decode byte 0xc3 in position 1: "
         'unexpected end of data'), 'keep'))
eq('refuse ucs4', m.write_refusals('ucs4', '', 0, 0),
   (-1, ('ValueError', 'size must be positive'), 'keep'))
eq('refuse substring type', m.write_refusals('substring', 5, 0, 0),
   (-1, ('TypeError', 'expect str, not int'), 'keep'))
eq('refuse substring start', m.write_refusals('substring', 'abc', -1, 2),
   (-1, ('ValueError', 'invalid start argument'), 'keep'))
eq('refuse substring order', m.write_refusals('substring', 'abc', 2, 1),
   (-1, ('ValueError', 'invalid start argument'), 'keep'))
eq('refuse substring end', m.write_refusals('substring', 'abc', 0, 9),
   (-1, ('ValueError', 'invalid end argument'), 'keep'))

# The two writes that run Python code fail the way that code did.
eq('refuse str', m.write_refusals('str', NoText(), 0, 0),
   (-1, ('RuntimeError', 'no str'), 'keep'))
eq('refuse repr', m.write_refusals('repr', NoText(), 0, 0),
   (-1, ('RuntimeError', 'no repr'), 'keep'))

# The substring bounds are in code points, so a character outside the basic
# plane counts once.
eq('substring', m.write_substring('h☃llo', 1, 4), ('☃ll', None))
eq('substring empty', m.write_substring('abc', 2, 2), ('', None))
eq('substring surrogate', m.write_substring('a\ud800b', 0, 3),
   ('a\ud800b', None))

# ── taking a key out ───────────────────────────────────────────────────

data = {'a': 1, 2: 'b'}
eq('pop hit', m.dict_pop(data, 'a'), (1, 1, None))
eq('pop miss', m.dict_pop(data, 'zz'), (0, None, None))
eq('pop left', data, {2: 'b'})

# An empty dictionary answers the miss without hashing the key at all, so a
# key that cannot be hashed is a miss there and a TypeError anywhere else.
eq('pop empty unhashable', m.dict_pop({}, Unhashable()), (0, None, None))
answer = m.dict_pop({1: 2}, Unhashable())
eq('pop unhashable', (answer[0], answer[1], answer[2][0]),
   (-1, None, 'TypeError'))

# Only the class: the report names the source of whoever made the call, and
# the runtime's own files are not the extension's.
answer = m.dict_pop([], 'a')
eq('pop not a dict', (answer[0], answer[1], answer[2][0]),
   (-1, None, 'SystemError'))

# A NULL result is the caller saying it wants the removal, not the value.
eq('pop no result', m.dict_pop_no_result({'k': 9}, 'k'), (1, None))
eq('pop string', m.dict_pop_string({'k': 9}, 'k'), (1, 9, None))
eq('pop string miss', m.dict_pop_string({'k': 9}, 'j'), (0, None, None))

# The check is `PyDict_Check`, so a subclass instance is reached through its
# concrete mapping rather than through whatever it overrides.
class Loud(dict):
    def pop(self, *arguments):
        raise AssertionError('the override was consulted')


loud = Loud(k=1)
eq('pop a subclass', m.dict_pop(loud, 'k'), (1, 1, None))
eq('pop a subclass left', dict(loud), {})

# ── the read-only view ─────────────────────────────────────────────────

proxy, left = m.dict_proxy({'a': 1})
eq('proxy dict', (dict(proxy), left), ({'a': 1}, None))
eq('proxy str', m.dict_proxy('abc')[1], None)
for value, name in [([1], 'list'), ((1,), 'tuple'), (5, 'int')]:
    eq('proxy %s' % name, m.dict_proxy(value),
       ('TypeError',
        'mappingproxy() argument must be a mapping, not %s' % name))

# ── the read that answers "was it there?" ──────────────────────────────

eq('optional hit', m.optional_item({'a': 1}, 'a'), (1, 1, None))
eq('optional miss', m.optional_item({'a': 1}, 'b'), (0, None, None))

# A KeyError the mapping raised is the miss; every other failure is not.
eq('optional custom miss', m.optional_item(Missing(), 'b'), (0, None, None))
eq('optional custom raise', m.optional_item(Angry(), 'b'),
   (-1, None, ('RuntimeError', 'no')))
answer = m.optional_item({1: 2}, Unhashable())
eq('optional unhashable', (answer[0], answer[1], answer[2][0]),
   (-1, None, 'TypeError'))

# It is the subscript, not a mapping test, so a sequence is read by index.
eq('optional list', m.optional_item([10, 20], 1), (1, 20, None))
eq('optional string key', m.optional_item_string({'a': 1}, 'a'), (1, 1, None))
eq('optional string miss', m.optional_item_string({'a': 1}, 'z'),
   (0, None, None))
eq('optional string on a list', m.optional_item_string([1], 'a'),
   (-1, None, ('TypeError',
               'list indices must be integers or slices, not str')))

# ── clearing and extending a list ──────────────────────────────────────

items = [1, 2, 3]
eq('clear', m.list_clear(items), (0, None))
eq('clear left', items, [])
eq('extend', m.list_extend(items, (4, 5)), (0, None))
eq('extend left', items, [4, 5])
eq('extend a string', m.list_extend(items, 'xy'), (0, None))
eq('extend left again', items, [4, 5, 'x', 'y'])
eq('extend a non-iterable', m.list_extend(items, 5),
   (-1, ('TypeError', "'int' object is not iterable")))
answer = m.list_clear(())
eq('clear not a list', (answer[0], answer[1][0]), (-1, 'SystemError'))
answer = m.list_extend((), (1,))
eq('extend not a list', (answer[0], answer[1][0]), (-1, 'SystemError'))

# A list subclass is a list, and the concrete methods are what run.
class Chatty(list):
    def clear(self):
        raise AssertionError('the override was consulted')

    def extend(self, other):
        raise AssertionError('the override was consulted')


chatty = Chatty([1, 2])
eq('extend a subclass', m.list_extend(chatty, (3,)), (0, None))
eq('extend a subclass left', list(chatty), [1, 2, 3])
eq('clear a subclass', m.list_clear(chatty), (0, None))
eq('clear a subclass left', list(chatty), [])

# ── joining and concatenating bytes ────────────────────────────────────

eq('join', m.bytes_join(b'-', [b'a', b'b']), (b'a-b', None))
eq('join nothing', m.bytes_join(b'-', []), (b'', None))
eq('join a str separator', m.bytes_join('-', [b'a']),
   ('TypeError', 'sep: expected bytes, got str'))
eq('join a str item', m.bytes_join(b'-', ['a']),
   ('TypeError',
    'sequence item 0: expected a bytes-like object, str found'))

# The reference handed over is replaced by the concatenation, and a NULL
# right side is the caller asking for it to be dropped.
eq('concat', m.bytes_concat(b'ab', b'cd'), (b'abcd', None))
eq('concat nothing', m.bytes_concat(b'ab', None), (None, None))
eq('concat a str', m.bytes_concat(b'ab', 'cd'),
   (None, ('TypeError', "can't concat str to bytes")))
eq('concat a bytearray', m.bytes_concat(b'ab', bytearray(b'cd')),
   (b'abcd', None))
eq('concat and del', m.bytes_concat_and_del(b'ab', b'cd'), (b'abcd', None))

print('cpyext-writer-ok')
"#;

#[test]
fn the_str_an_extension_builds() {
    let fixtures = Fixtures::new("cpyext-writer");
    fixtures.compile("cpyext_writer");
    fixtures.expect_ok(SCRIPT, &[], "cpyext-writer-ok");
}
