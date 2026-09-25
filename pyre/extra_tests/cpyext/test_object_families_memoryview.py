# cpyext-fixture: cpyext_object_families
# cpyext-expect: cpyext-memoryview-ok

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


def eq(name, got, want):
    assert got == want, '%s: got %r, want %r' % (name, got, want)


# An exporter already laid out the way the caller asked for is handed back as
# it is, so what comes out reads the same bytes.
view = m.mv_contiguous(b'abcdef', 'r', 'C')
eq('a read view over bytes', bytes(view), b'abcdef')
eq('and it is read-only', view.readonly, True)
eq('any order accepts a one-dimensional layout',
   bytes(m.mv_contiguous(b'abcdef', 'r', 'A')), b'abcdef')
eq('so does Fortran order',
   bytes(m.mv_contiguous(b'abcdef', 'r', 'F')), b'abcdef')

data = bytearray(b'abcdef')
written = m.mv_contiguous(data, 'w', 'C')
eq('a write view is not read-only', written.readonly, False)
written[0] = ord('z')
eq('and it writes through to the exporter', data, bytearray(b'zbcdef'))

# What the buffer type asks for is checked against the exporter, not silently
# copied around.
eq('a write view over bytes', m.mv_contiguous(b'abcdef', 'w', 'C'),
   ('BufferError', 'underlying buffer is not writable'))

# A strided view is contiguous in no order, and only the read side has a copy
# to fall back on: it gets the selected elements laid out in the asked-for
# order, over storage of its own rather than the exporter's.
data = bytearray(b'abcdef')
strided = memoryview(data)[::2]
copied = m.mv_contiguous(strided, 'r', 'C')
eq('a read view over a strided one', bytes(copied), b'ace')
eq('and the copy is read-only', copied.readonly, True)
data[0] = ord('z')
eq('and it does not write through to the exporter', bytes(copied), b'ace')
eq('a write view over a strided one', m.mv_contiguous(strided, 'w', 'C'),
   ('BufferError',
    'writable contiguous buffer requested for a non-contiguous object.'))

# Both arguments are checked before the exporter is touched.
eq('a buffer type that is neither', m.mv_contiguous(b'abcdef', '?', 'C'),
   ('ValueError', 'buffertype must be PyBUF_READ or PyBUF_WRITE'))
eq('an order that is none of the three', m.mv_contiguous(b'abcdef', 'r', 'X'),
   ('ValueError', "order must be in ('C', 'F', 'A')"))

# An object that exports no buffer at all fails where `memoryview` would.
kind, _ = m.mv_contiguous(object(), 'r', 'C')
eq('an object with no buffer to export', kind, 'TypeError')

# An item naming two members is one the view does not take apart: the geometry
# is readable, and every operation that would read or write a single element
# refuses.  `adjust_fmt` is reached before the dimension and before the
# writability check, so the format is what each of them reports.
# Three formats one word wide that differ only in signedness.  A reader that
# takes the item's width and ignores its code answers a word with the top bit
# set as -1 for all three.
eq('a signed word', m.mv_word_format('n').tolist(), [-1, 1])
eq('an unsigned word', m.mv_word_format('N').tolist(), [18446744073709551615, 1])
eq('a pointer word', m.mv_word_format('P').tolist(), [18446744073709551615, 1])

compound = m.mv_compound_format()
eq('compound geometry',
   (compound.format, compound.itemsize, compound.nbytes, compound.ndim),
   ('II', 8, 16, 1))

wanted = ('NotImplementedError', 'memoryview: unsupported format II')
for what, call in (
    ('getitem', lambda: compound[0]),
    ('getitem tuple', lambda: compound[0,]),
    ('setitem', lambda: compound.__setitem__(0, 8)),
    ('slice assignment', lambda: compound.__setitem__(slice(0, 1), b'')),
    ('tolist', compound.tolist),
    ('iter', lambda: list(compound)),
):
    try:
        call()
    except Exception as exc:
        eq(what, (type(exc).__name__, str(exc)), wanted)
    else:
        raise AssertionError('%s did not refuse' % what)

# The geometry stays readable, and so does the raw memory.
eq('compound bytes', compound.tobytes(), bytes(compound))

print('cpyext-memoryview-ok')
