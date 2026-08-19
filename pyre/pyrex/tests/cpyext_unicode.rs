//! End-to-end check for the canonical `str` representation an extension reads
//! and writes: the kind/data pair, the read and write macros, and the
//! allocate-then-fill shape `PyUnicode_New` exists for.

#![cfg(all(
    feature = "cpyext",
    not(feature = "sandbox"),
    any(target_os = "macos", target_os = "linux")
))]

mod cpyext_fixture;

use cpyext_fixture::Fixtures;

const SCRIPT: &str = r#"
import cpyext_unicode as u

assert (u.ONE_BYTE, u.TWO_BYTE, u.FOUR_BYTE) == (1, 2, 4)

# ── the width a string reports is the narrowest that holds it ──────────
# (kind, is_ascii, max_char_value, length)
assert u.shape('') == (1, 1, 0x7f, 0), u.shape('')
assert u.shape('abc') == (1, 1, 0x7f, 3), u.shape('abc')
assert u.shape('naïve') == (1, 0, 0xff, 5), u.shape('naïve')
assert u.shape('こんにちは') == (2, 0, 0xffff, 5), u.shape('こんにちは')
assert u.shape('\U0001f363') == (4, 0, 0x10ffff, 1), u.shape('\U0001f363')
# A surrogate pair is two code points in the source and one string of two.
assert u.shape('a\U0001f363') == (4, 0, 0x10ffff, 2), u.shape('a\U0001f363')

# ── the typed data casts agree with PyUnicode_READ_CHAR ────────────────
assert u.first_point('abc') == ord('a')
assert u.first_point('ïve') == ord('ï')
assert u.first_point('こ') == ord('こ')
assert u.first_point('\U0001f363') == 0x1f363

# ── allocate, fill, hand back: the result is an ordinary str ───────────
escaped = u.escape('a<b')
assert escaped == 'a&lt;b', escaped
assert type(escaped) is str
assert u.escape('<<') == '&lt;&lt;'
# The output keeps the width the input needed, whatever the escape adds.
assert u.escape('こ<ん') == 'こ&lt;ん', u.escape('こ<ん')
assert u.shape(u.escape('こ<ん')) == (2, 0, 0xffff, 6)
assert u.escape('\U0001f363<') == '\U0001f363&lt;'
assert u.shape(u.escape('\U0001f363<')) == (4, 0, 0x10ffff, 5)
# Nothing to escape: the argument itself comes back.
plain = 'no markup here'
assert u.escape(plain) is plain
# A long input takes the same path.
big = 'x' * 5000 + '<' + 'y' * 5000
assert u.escape(big) == 'x' * 5000 + '&lt;' + 'y' * 5000
# The written string is a str in every way, not just by value.
assert u.escape('a<b').upper() == 'A&LT;B'
assert len(u.escape('a<b')) == 6
assert {u.escape('a<b'): 1}['a&lt;b'] == 1

# ── the entry-point spellings of the same read and write ───────────────
assert u.reverse('abc') == 'cba'
assert u.reverse('こんにちは') == 'はちにんこ'
assert u.reverse('a\U0001f363b') == 'b\U0001f363a'
assert u.reverse('') == ''

# ── a new string handed to another entry point, not returned ───────────
# Its value is built where it first crosses back into the interpreter, so
# the pair below is an ordinary dict entry.
assert u.pairs() == {'kk': 'あああ'}, u.pairs()
assert u.join('ab') == 'ab\U0001f363\U0001f363', u.join('ab')
assert u.shape(u.join('ab')) == (4, 0, 0x10ffff, 4)

# ── an empty allocation, and the argument checks ───────────────────────
assert u.empty() == ''
assert u.out_of_range() is True
assert u.rejects() is True

try:
    u.escape(b'bytes')
except TypeError:
    pass
else:
    raise AssertionError('escape() accepted a non-str')

# ── a write is measured against the width the string was made with ─────
assert u.new_and_write(3, 0x7f, 0x41) == 'Axx'
assert u.new_and_write(3, 0xff, 0xff) == '\u00ffxx'
assert u.new_and_write(3, 0xffff, 0xffff) == '\uffffxx'
assert u.new_and_write(3, 0x10ffff, 0x1f363) == '\U0001f363xx'
# A surrogate is a code point a str holds, so it is written like any other.
# The width asked for is the one that holds it: a string wider than its
# contents need is not the string the same text spells.
assert u.new_and_write(3, 0xffff, 0xd800) == '\ud800xx'
for maxchar, value in [
    (0x7f, 0xff), (0x7f, 0x1f363), (0xff, 0x100),
    (0xffff, 0x10000), (0x10ffff, 0x110000),
]:
    try:
        u.new_and_write(3, maxchar, value)
    except ValueError as error:
        assert str(error) == 'character out of range', (maxchar, value, error)
    else:
        raise AssertionError(f'wrote {value:#x} into a string of {maxchar:#x}')

# ── neither accessor takes anything but a string ───────────────────────
refused = ('TypeError', 'bad argument type for built-in operation')
assert u.not_a_string(b'abc') == (refused, refused), u.not_a_string(b'abc')
assert u.not_a_string(3) == (refused, refused)

# ── a count of nothing names no buffer, but the kind is still checked ───
rows, refused_kind = u.from_nothing()
assert rows == ['', '', '', ''], rows
assert refused_kind == ('SystemError', 'invalid kind'), refused_kind

print('cpyext-unicode-ok')
"#;

#[test]
fn reads_and_writes_the_canonical_representation() {
    let fixtures = Fixtures::new("cpyext-unicode");
    fixtures.compile("cpyext_unicode");
    fixtures.expect_ok(SCRIPT, &[], "cpyext-unicode-ok");
}
