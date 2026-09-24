# cpyext-fixture: cpyext_object_families
# cpyext-expect: cpyext-bytearray-ok

# `bytearray`, `complex`, `memoryview` and `weakref` through their concrete
# C API.
#
# Whole families an extension reaches for that the layer did not have, so an
# extension naming any of them did not compile.
#
# Every expectation was taken from CPython 3.14.6 running this same script
# against this same fixture, except where noted: two rows are where CPython
# checks its argument with an `assert` and so reads a release build's
# answer off whatever it was handed, which is not behaviour to match.

import cpyext_object_families as m

class BA(bytearray):
    pass

def eq(name, got, want):
    assert got == want, '%s: got %r, want %r' % (name, got, want)

eq('check(bytearray)', m.ba_check(bytearray(b'abc')), True)
eq('check(subclass)', m.ba_check(BA(b'abc')), True)
eq('check(bytes)', m.ba_check(b'abc'), False)
eq('check_exact(bytearray)', m.ba_check_exact(bytearray(b'abc')), True)
eq('check_exact(subclass)', m.ba_check_exact(BA(b'abc')), False)

eq('size', m.ba_size(bytearray(b'abc')), 3)
eq('size(subclass)', m.ba_size(BA(b'abc')), 3)
# CPython asserts its argument is a bytearray and otherwise reads `ob_size` off
# whatever it was given -- for `bytes` that is the length, by layout alone.
# There is nothing there to match, so this refuses instead.
eq('size(bytes) is refused', m.ba_size(b'abc'), -1)
eq('size(str) is refused', m.ba_size('abc'), -1)

eq('from_string_and_size', m.ba_from_string_and_size(b'hello', 5), bytearray(b'hello'))
eq('from_string_and_size, shorter', m.ba_from_string_and_size(b'hello', 2), bytearray(b'he'))
# A NULL source asks for a buffer of that size; what is in it is not defined.
eq('from NULL is that long', len(m.ba_from_null(4)), 4)
eq('from NULL of nothing', m.ba_from_null(0), bytearray())
# A length below zero is the caller's mistake, not a buffer of no bytes.
try:
    m.ba_from_null(-1)
except SystemError as error:
    eq('from negative', str(error),
       'Negative size passed to PyByteArray_FromStringAndSize')
else:
    raise AssertionError('PyByteArray_FromStringAndSize accepted a negative size')

eq('from_object(bytes)', m.ba_from_object(b'xy'), bytearray(b'xy'))
eq('from_object(list)', m.ba_from_object([1, 2, 3]), bytearray(b'\x01\x02\x03'))
eq('from_object(str) is refused', m.ba_from_object('nope'), 'fromobject-failed')

eq('concat', m.ba_concat(bytearray(b'ab'), bytearray(b'cd')), bytearray(b'abcd'))

# The payload, and the terminator one past the length that lets it be read as a
# C string.
eq('as_string', m.ba_as_string(bytearray(b'abc')), (b'abc', True))
eq('as_string of empty', m.ba_as_string(bytearray()), (b'', True))
eq('as_string(subclass)', m.ba_as_string(BA(b'qq')), (b'qq', True))

# A write through that pointer reaches the object Python holds.
target = bytearray(b'abc')
m.ba_write_through(target)
eq('write through is visible', bytes(target), b'Zbc')

grow = bytearray(b'ab')
m.ba_resize(grow, 5)
eq('resize grows', len(grow), 5)
eq('resize keeps the prefix', bytes(grow[:2]), b'ab')
shrink = bytearray(b'abcdef')
m.ba_resize(shrink, 2)
eq('resize shrinks', bytes(shrink), b'ab')
empty = bytearray(b'xy')
m.ba_resize(empty, 0)
eq('resize to nothing', bytes(empty), b'')

print('cpyext-bytearray-ok')
