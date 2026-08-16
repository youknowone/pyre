//! End-to-end check for the method calling conventions, the argument parsers,
//! the object constructors and the exception indicator, driven from a
//! multi-phase (PEP 489) extension.

#![cfg(all(
    feature = "cpyext",
    not(feature = "sandbox"),
    any(target_os = "macos", target_os = "linux")
))]

mod cpyext_fixture;

use cpyext_fixture::Fixtures;

const SCRIPT: &str = r#"
import sys
import cpyext_methods as m

# ── the exec slot ran, and its module-state block is per-module ────────
assert m.__name__ == 'cpyext_methods'
assert m.__doc__ == 'pyre cpyext method module'
assert m.__file__.endswith('.so'), m.__file__
assert m.ANSWER == 42
assert m.GREETING == 'hi'
assert m.OWNED == 11
assert m.SHARED == 12
assert m.VIA_DICT == 'through-dict'

assert m.bump() == 1
assert m.bump() == 2
assert m.bump() == 3

# ── the carrier ────────────────────────────────────────────────────────
assert repr(m.bump) == '<built-in function bump>'
assert m.bump.__name__ == 'bump'
assert m.bump.__qualname__ == 'bump'
assert m.bump.__doc__ == 'bump the module counter'
assert m.wrap.__doc__ is None
assert m.bump.__self__ is m
assert m.bump.__module__ == 'cpyext_methods'
assert callable(m.bump)

def rejects(call, *args, **kwargs):
    try:
        call(*args, **kwargs)
    except TypeError:
        return
    raise AssertionError('%r accepted %r %r' % (call, args, kwargs))

# ── METH_NOARGS / METH_O ───────────────────────────────────────────────
rejects(m.bump, 1)
rejects(m.bump, key=1)
assert m.wrap(7) == (7, 'seen')
rejects(m.wrap)
rejects(m.wrap, 1, 2)

# ── METH_VARARGS and PyArg_ParseTuple ──────────────────────────────────
assert m.add(1, 2) == 3
assert m.add(5) == 15
rejects(m.add)
rejects(m.add, 1, 2, 3)

# ── METH_VARARGS | METH_KEYWORDS and PyArg_ParseTupleAndKeywords ───────
assert m.greet('pyre') == 'hello pyre!'
assert m.greet('pyre', '?') == 'hello pyre?'
assert m.greet('pyre', punct='.') == 'hello pyre.'
assert m.greet(name='pyre') == 'hello pyre!'
assert m.greet(name='pyre', punct='?') == 'hello pyre?'
rejects(m.greet, 'pyre', nope=1)
rejects(m.greet)

# ── METH_FASTCALL, with and without keywords ───────────────────────────
assert m.total() == 0
assert m.total(1, 2, 3) == 6
assert m.layout() == ([], [])
assert m.layout(1, 2) == ([1, 2], [])
assert m.layout(1, a=2, b=3) == ([1], [('a', 2), ('b', 3)])

# ── PyArg_UnpackTuple and the object protocol ──────────────────────────
assert m.apply(abs, -4) == 4
assert m.apply(str, 12) == '12'
rejects(m.apply, abs)

present, text, shown, size, truth = m.inspect([1, 2], 'append')
assert (present, text, shown, size, truth) == (1, '[1, 2]', '[1, 2]', 2, 1)
present, text, shown, size, truth = m.inspect(7, 'append')
assert (present, text, shown, size, truth) == (0, '7', '7', -1, 1)

# ── Py_BuildValue ──────────────────────────────────────────────────────
assert m.build() == {
    'int': 3,
    'long': 4,
    'float': 1.5,
    'str': 'text',
    'bytes': b'raw',
    'list': [1, 2, 3],
    'tuple': ('pair', 9),
}

# ── str / bytes round trips through the mirror's byte cache ────────────
assert m.roundtrip('sévère', b'a\x00b') == ('sévère', b'a\x00b', 6, 3)
assert m.roundtrip('', b'') == ('', b'', 0, 0)
rejects(m.roundtrip, b'bytes', b'bytes')

# ── int / float ────────────────────────────────────────────────────────
assert m.numbers(42) == (42, 42.0, -42, 1, 0, 0.25)
assert m.numbers(True) == (1, 1.0, -42, 1, 0, 0.25)

# ── dict / tuple / list ────────────────────────────────────────────────
assert m.dict_ops() == (7, 1, 1, 0, 1)
assert m.sequences() == ((1, 'two'), [5, b'ab\x00c'], 2, 2)

# ── the singletons are the interpreter's own objects ───────────────────
none, true, false, ellipsis, notimplemented = m.singletons()
assert none is None
assert true is True
assert false is False
assert ellipsis is Ellipsis
assert notimplemented is NotImplemented
assert m.predicates(None) == (1, 0, 0, 0, 0, 0, 0)
assert m.predicates(True) == (0, 1, 0, 0, 0, 0, 0)
assert m.predicates('x') == (0, 0, 1, 0, 0, 0, 0)
assert m.predicates(b'x') == (0, 0, 0, 1, 0, 0, 0)
assert m.predicates(()) == (0, 0, 0, 0, 1, 0, 0)
assert m.predicates([]) == (0, 0, 0, 0, 0, 1, 0)
assert m.predicates({}) == (0, 0, 0, 0, 0, 0, 1)

# ── the exception indicator ────────────────────────────────────────────
def raises(kind, cls, message):
    try:
        m.fail(kind)
    except cls as error:
        assert str(error) == message, (kind, str(error))
        return
    raise AssertionError('fail(%r) did not raise' % (kind,))

raises('value', ValueError, 'a value complaint')
raises('format', TypeError, 'formatted message and 7')
raises('object', KeyError, "'payload'")
raises('none', StopIteration, '')
raises('memory', MemoryError, 'out of memory')
raises('argument', TypeError, 'bad argument type for built-in operation')
raises('other', RuntimeError, 'unknown failure kind')

assert m.caught() == (1, 1, 0, 1, 'swallowed')
# PyErr_Restore: (cleared by NULL class, bare instance matches, is an instance,
# fetched pair round-trips, NULL class clears even with a value)
assert m.restore() == (1, 1, 1, 1, 1), m.restore()

# patchlevel.h, as an extension expands it: a banner string, the parts, and the
# packed hex.  Each is compared against what the runtime reports, so a header
# that drifts from sys is a failure rather than a silent disagreement.
banner, major, minor, micro, level, serial, version, hexversion = m.version_macros()
assert (major, minor, micro) == sys.version_info[:3], (major, minor, micro)
assert version == '%d.%d.%d' % sys.version_info[:3], version
assert (level, serial) == (0xF, 0), (level, serial)
assert sys.version_info.releaselevel == 'final'
assert hexversion == sys.hexversion, (hex(hexversion), hex(sys.hexversion))
assert banner.startswith('python ' + version + ' / pyre '), banner
assert banner.endswith('.'.join(str(p) for p in sys.pyre_version_info[:3])), banner
# The swallowed exception must not leak into the caller.
assert m.bump() == 4

# ── a re-import rebuilds the module from its definition ────────────────
m.runtime_only = 1
del sys.modules['cpyext_methods']
again = __import__('cpyext_methods')
assert again is not m
assert not hasattr(again, 'runtime_only')
assert again.ANSWER == 42
assert again.bump() == 1

# ── the call entry points ──────────────────────────────────────────────
# `record` answers with exactly what it was given, so each spelling's result
# states the argument vector that reached it.
def record(*args, **kwargs):
    return (args, kwargs)

noargs, onearg, vec, vec_kw, objargs, fmt, meth_no, meth_fmt = \
    m.call_surface(record, 7, 'v')

assert noargs == ((), {}), noargs
assert onearg == ((7,), {}), onearg
assert vec == ((7, 7), {}), vec
# One positional and one named: the name comes from the `kwnames` tuple and
# names the entry *after* the positional run.
assert vec_kw == ((7,), {'kw': 'v'}), vec_kw
assert objargs == ((7, 7, 7), {}), objargs
assert fmt == ((7, 7), {}), fmt
# The method spellings, against 'abc'.
assert meth_no == 'ABC', meth_no
assert meth_fmt == 1, meth_fmt

# ── the set protocol and the raw allocators ────────────────────────────
(empty_size, after_add, after_readd, has_key,
 first, second, after_pop, after_clear,
 is_set, is_frozen, is_any, typed) = m.set_ops([1, 2, 3], 9)

assert empty_size == 0, empty_size
# 9 is not in {1,2,3}, so adding it grows the set and re-adding does not.
assert (after_add, after_readd) == (4, 4), (after_add, after_readd)
assert has_key == 1
# `discard` finds it once, then reports absence rather than raising.
assert (first, second) == (1, 0), (first, second)
assert after_pop == 2, after_pop
assert after_clear == 0, after_clear
assert (is_set, is_frozen, is_any) == (1, 1, 1), (is_set, is_frozen, is_any)
# PyMem_Calloc zeroed, PyMem_Realloc kept the bytes, PyMem_New sized right.
assert typed == 1234, typed

# ── the rest of the dict protocol ──────────────────────────────────────
d = {'a': 1, 'b': 2}
(copy, keys, values, items, walked,
 ref_hit, ref_miss, str_hit, miss_clean, with_error_clean, got,
 after_merge, after_update, kept, replaced) = m.dict_more(d, 'b', {'b': 20, 'c': 3})

assert copy == d and copy is not d, copy
assert sorted(keys) == ['a', 'b'], keys
assert sorted(values) == [1, 2], values
assert sorted(items) == [('a', 1), ('b', 2)], items
# PyDict_Next reported every pair exactly once.
assert sorted(walked) == [('a', 1), ('b', 2)], walked
# GetItemRef: 1 when present, 0 when absent, and an absent key sets no error.
assert (ref_hit, ref_miss, str_hit) == (1, 0, 1), (ref_hit, ref_miss, str_hit)
assert miss_clean == 1
assert with_error_clean == 1
assert got == 2, got
# Merge without override kept 'b' at 2 and added 'c'; Update then replaced 'b'.
assert (after_merge, after_update) == (3, 3), (after_merge, after_update)
assert (kept, replaced) == (2, 20), (kept, replaced)

# ── the rest of the list protocol, PyTuple_Pack and PyTuple_GetSlice ───
(head, after_reverse, after_sort,
 as_tuple, sliced, mutated, packed, tuple_slice) = m.list_ops([3, 1, 2])

# 99 went in at the front and PyList_GetItemRef read it back.
assert head == 99, head
# [99,3,1,2] reversed is [2,1,3,99]; sorting it puts 1 first.
assert after_reverse == 2, after_reverse
assert after_sort == 1, after_sort
assert as_tuple == (1, 2, 3, 99), as_tuple
assert sliced == [2, 3], sliced
# [1,2,3,99] with [0:2] replaced by [-1,-2] is [-1,-2,3,99]; deleting [0:1]
# leaves [-2,3,99].
assert mutated == [-2, 3, 99], mutated
assert packed == (None, True, False), packed
assert tuple_slice == (True, False), tuple_slice

# ── the slice protocol ─────────────────────────────────────────────────
(u_start, u_stop, u_step,
 a_start, a_stop, a_len,
 g_start, g_stop, g_step,
 e_start, e_stop, e_step, e_len,
 made, made_is_slice, none_is_slice) = m.slice_ops(slice(1, 10, 2), 5)

# Unpack reads the bounds without consulting the length.
assert (u_start, u_stop, u_step) == (1, 10, 2), (u_start, u_stop, u_step)
# Adjusting against 5 clips the stop and leaves 2 items: 1 and 3.
assert (a_start, a_stop, a_len) == (1, 5, 2), (a_start, a_stop, a_len)
assert (g_start, g_stop, g_step) == (1, 5, 2), (g_start, g_stop, g_step)
assert (e_start, e_stop, e_step, e_len) == (1, 5, 2, 2)
assert made == slice(1, None, 1), made
assert (made_is_slice, none_is_slice) == (1, 0)

# An unbounded negative slice unpacks to the open-ended sentinels.
u_start, u_stop, u_step, a_start, a_stop, a_len = \
    m.slice_ops(slice(None, None, -1), 4)[:6]
assert u_start == sys.maxsize, u_start
assert u_stop == -sys.maxsize - 1, u_stop
assert u_step == -1
assert (a_start, a_stop, a_len) == (3, -1, 4), (a_start, a_stop, a_len)

# ── the rest of the sequence protocol, PyIter_NextItem, ToBase, ipow ───
(count, contained, fast_items, fast_size, fast_is_self,
 drained, clean, owned, based, powered) = m.seq_more([1, 2, 2, 3], 2)

assert (count, contained) == (2, 1), (count, contained)
# A list is already fast, so PySequence_Fast hands back the list itself.
assert fast_is_self == 1
assert (fast_items, fast_size) == ([1, 2, 2, 3], 4)
assert drained == [1, 2, 2, 3], drained
assert clean == 1
# [1,2,2,3] with [0:1] set to [7] is [7,2,2,3]; deleting [1:2] leaves [7,2,3].
assert owned == [7, 2, 3], owned
assert based == '0x2', based
assert powered == 81, powered

# A range is neither a list nor a tuple, so PySequence_Fast builds one.
(count, contained, fast_items, fast_size, fast_is_self,
 drained, clean, owned, based, powered) = m.seq_more(range(4), 2)

assert (count, contained) == (1, 1), (count, contained)
assert fast_is_self == 0
assert (fast_items, fast_size) == ([0, 1, 2, 3], 4)
assert drained == [0, 1, 2, 3], drained
assert owned == [7, 2, 3], owned
assert based == '0x2'

# PyNumber_ToBase rejects a base it has no marker for.
try:
    m.to_base(5, 7)
except SystemError:
    pass
else:
    raise AssertionError('to_base accepted base 7')
assert m.to_base(255, 2) == '0b11111111'
assert m.to_base(255, 8) == '0o377'
assert m.to_base(255, 10) == '255'
assert m.to_base(255, 16) == '0xff'

# PySequence_GetItem under sustained allocation pressure: 20000 calls, each
# boxing its own index, over a list built inside the same C call.
assert m.gc_window(20000) == 70000, m.gc_window(20000)

# ── the module constructors and accessors ──────────────────────────────
fresh, fresh_name, named_name, same_file = m.module_ops()

import types as _types
assert isinstance(fresh, _types.ModuleType), fresh
assert fresh.__name__ == 'cpyext_methods.fresh', fresh.__name__
assert fresh_name == 'cpyext_methods.fresh', fresh_name
assert fresh.__doc__ == 'a module built from C', fresh.__doc__
assert fresh.SEVEN == 7
# PyModule_AddFunctions binds the table to the module it was added to.
assert fresh.added() == 'cpyext_methods.fresh', fresh.added()
assert named_name == 'cpyext_methods.named', named_name
# Both spellings of __file__ name the same path.
assert same_file == 1

try:
    m.module_no_file()
except SystemError:
    pass
else:
    raise AssertionError('a module with no __file__ reported one')

# PyModule_FromDefAndSpec runs the same definition against a spec of its own,
# and PyModule_ExecDef then runs its exec slot.
class _Spec:
    name = 'cpyext_methods.again'

again = m.module_from_def(_Spec())
assert isinstance(again, _types.ModuleType), again
assert again.ANSWER == 42
assert again.GREETING == 'hi'
assert again.VIA_DICT == 'through-dict'
# A separate module has its own state block, so its counter starts over.
assert again.bump() == 1
assert m.bump() == m.bump() - 1

# ── the rest of the object protocol ────────────────────────────────────
class _Holder:
    def __init__(self):
        self.tag = 'original'

holder = _Holder()
present, missing, has, has_string, entries, deleted = m.object_attrs(holder, 'tag')
assert present == 1, present
# The absent lookup reports 0 and leaves no exception behind.
assert missing == 0, missing
assert has == 1 and has_string == 0, (has, has_string)
# PyObject_GenericGetDict is the instance's own __dict__.
assert entries == 1, entries
# PyObject_DelAttr removed it, and PyObject_HasAttr agreed.
assert deleted == 1, deleted
assert not hasattr(holder, 'tag')

mapping = {'a': 1, 'b': 2}
(less, same, differs, hashed, ascii_form, formatted,
 name_count, left_over, subclass) = m.object_values(3, 5, mapping, 'a')
assert less is True, less
assert same == 1 and differs == 1, (same, differs)
assert hashed == 1
assert ascii_form == '3', ascii_form
assert formatted == '3', formatted
assert name_count == len(dir(3)), (name_count, len(dir(3)))
# PyObject_DelItem removed one of the two keys.
assert left_over == 1 and mapping == {'b': 2}, (left_over, mapping)
assert subclass == 1

# PyObject_Bytes: an exact bytes is itself, a buffer is copied, and an
# iterable of ints is read element by element.
raw = b'raw'
assert m.object_bytes(raw) is raw
assert m.object_bytes(bytearray(b'buf')) == b'buf'
assert m.object_bytes([1, 2, 3]) == b'\x01\x02\x03'
assert m.bytes_from(bytearray(b'buf')) == b'buf'

class _Bytes:
    def __bytes__(self):
        return b'from-dunder'

# Only PyObject_Bytes consults __bytes__; PyBytes_FromObject does not, and an
# object with neither a buffer nor an __iter__ has nothing left to convert.
assert m.object_bytes(_Bytes()) == b'from-dunder'
try:
    m.bytes_from(_Bytes())
except TypeError:
    pass
else:
    raise AssertionError('PyBytes_FromObject consulted __bytes__')

for converter in (m.object_bytes, m.bytes_from):
    try:
        converter('text')
    except TypeError:
        pass
    else:
        raise AssertionError('a str was converted to bytes')

# PyBytes_FromStringAndSize(NULL, size): the buffer is written through
# PyBytes_AS_STRING and the result is an ordinary bytes.
assert m.bytes_fill(3) == b'abc'
assert type(m.bytes_fill(3)) is bytes
assert m.bytes_fill(0) == b''
assert m.bytes_fill(30) == bytes(ord('a') + i % 26 for i in range(30))
# The written bytes are a bytes in every way, not just by value.
assert len(m.bytes_fill(5)) == 5
assert m.bytes_fill(5).upper() == b'ABCDE'
assert {m.bytes_fill(2): 1}[b'ab'] == 1
# A buffer handed to another entry point instead of returned. Its value is
# built where it first crosses back into the interpreter.
assert m.bytes_pairs() == {b'kk': b'vv\x00'}, m.bytes_pairs()
assert m.bytes_empty() == b''

# Py_NewRef / Py_XNewRef.
marker = ['kept']
assert m.new_ref(marker) == (True, marker)
assert m.new_ref(marker)[1] is marker

# The object allocator: contents survive a realloc, a calloc is zeroed, and a
# wrapping product is refused.
assert m.object_blocks() == (1, 1, 1), m.object_blocks()

# ── the type mirror behind Py_TYPE ─────────────────────────────────────
# A built-in type is not a heap type; a class written in Python is.
assert m.type_mirror(1) == ('int', 0), m.type_mirror(1)
assert m.type_mirror('x') == ('str', 0), m.type_mirror('x')

class Named:
    pass

assert m.type_mirror(Named()) == ('Named', 1), m.type_mirror(Named())
# A mirror is minted once and read back the same however many times it is
# asked for, and an instance keeps its own type's mirror readable.
kept = Named()
assert m.type_mirror(kept) == m.type_mirror(Named())
# Classes minted and dropped in a loop: each one's mirror is released with the
# class rather than pinning it, so the name read here is this class's own.
for index in range(64):
    made = type('T%d' % index, (), {})
    assert m.type_mirror(made()) == ('T%d' % index, 1)
assert m.type_mirror(kept) == ('Named', 1)
# The mirror carries the ordinary link share, so having been read from C does
# not make the class outlive the last reference to it.  Two collections: the
# instance's mirror holds a reference to its heap type's mirror, and it is the
# first collection's drain that gives that one back.
import gc, weakref
gone = type('Gone', (), {})
assert m.type_mirror(gone()) == ('Gone', 1)
watch = weakref.ref(gone)
del gone
gc.collect()
gc.collect()
assert watch() is None, watch()

# ── the int conversions ────────────────────────────────────────────────
(from_small, from_wide, narrow, wide, overflow, too_big, needed,
 restored, unsigned_restored, digit_bits) = m.int_convert(1 << 200)
assert from_small == -7, from_small
# An unsigned value above the signed range is a positive int, not an overflow.
assert from_wide == (1 << 64) - 1, from_wide
assert narrow == -7 and wide == (1 << 64) - 1, (narrow, wide)
# -7 fits a C long, so no overflow was reported; 1 << 200 does not.
assert overflow == 0, overflow
assert too_big == 1, too_big
# -7 needs one byte, and the eight written spell it in both readings.
assert needed == 1, needed
assert restored == -7, restored
assert unsigned_restored == (1 << 64) - 7, unsigned_restored
assert digit_bits == sys.int_info.bits_per_digit, (digit_bits, sys.int_info)

# ── the private byte-array conversions ─────────────────────────────────
big, little, negative, roundtrip, kept_low, raised = m.byte_arrays()
assert big == 0x01020304, big
assert little == 0x04030201, little
# The same bytes read as two's complement: 0x01020304 has a clear sign bit.
assert negative == 0x01020304, negative
assert roundtrip == 1, roundtrip
# A destination too small keeps the low bytes and reports -1 without raising.
assert kept_low == 1, kept_low
# A negative value asked for as unsigned leaves the destination alone.
assert raised == 1, raised

# ── the unchecked accessors ────────────────────────────────────────────
points, raw = m.fast_accessors(b'abc')
# PyUnicode_GET_LENGTH counts code points, not the UTF-8 bytes behind them.
assert points == 5, points
assert raw == b'abc', raw

# ── the buffer-filling argument formats ────────────────────────────────
text, data, maybe_len, maybe_null, readonly = m.buffer_formats(
    'naïve', b'\x00\xff', None)
# 's*' encodes a str to UTF-8, so the view is longer than the string.
assert text == 'naïve'.encode() and len(text) == 6, text
assert data == b'\x00\xff', data
# 'z*' takes None as an empty view over no object.
assert (maybe_len, maybe_null) == (0, 1), (maybe_len, maybe_null)
assert readonly == 1, readonly
# 's*' also accepts a bytes-like object; 'y*' refuses a str.
assert m.buffer_formats(b'raw', b'\x01', None)[0] == b'raw'
try:
    m.buffer_formats('t', 'not bytes', None)
except TypeError:
    pass
else:
    raise AssertionError("'y*' accepted a str")
# 'w*' asks for a writable view, which an interpreter object never exports.
assert m.writable_buffer(bytearray(b'ab')) == 'read-only'

print('cpyext-methods-ok')
"#;

#[test]
fn calls_c_functions_through_every_supported_convention() {
    let fixtures = Fixtures::new("cpyext-methods");
    fixtures.compile("cpyext_methods");
    fixtures.expect_ok(SCRIPT, &[], "cpyext-methods-ok");
}
