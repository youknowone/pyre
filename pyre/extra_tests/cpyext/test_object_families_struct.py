# cpyext-fixture: cpyext_object_families
# cpyext-expect: cpyext-struct-ok

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

import struct

import cpyext_object_families as m


def eq(name, got, want):
    assert got == want, '%s: got %r, want %r' % (name, got, want)


# The pair is (byte size, number of values), which is `calcsize` beside the
# length of what `unpack` answers.
for fmt, size, count in (('B', 1, 1),
                         ('3B', 3, 3),
                         ('i4d', 40, 5),
                         ('4s', 4, 1),
                         ('3x', 3, 0),
                         ('0i', 0, 0),
                         ('', 0, 0)):
    eq(fmt, m.struct_counts(struct.Struct(fmt)), (size, count))
    eq(fmt + ' calcsize', struct.calcsize(fmt), size)
    eq(fmt + ' values', len(struct.unpack(fmt, bytes(size))), count)

# `Struct.__new__` without `__init__` leaves the pair every operation but the
# `size` getter rejects.
eq('uninitialised', m.struct_counts(struct.Struct.__new__(struct.Struct)), (-1, -1))

print('cpyext-struct-ok')
