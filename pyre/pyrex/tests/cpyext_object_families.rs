//! `bytearray`, `complex`, `memoryview` and `weakref` through their concrete
//! C API.
//!
//! Whole families an extension reaches for that the layer did not have, so an
//! extension naming any of them did not compile.
//!
//! Every expectation was taken from CPython 3.14.6 running this same script
//! against this same fixture, except where noted: two rows are where CPython
//! checks its argument with an `assert` and so reads a release build's
//! answer off whatever it was handed, which is not behaviour to match.

#![cfg(all(
    feature = "cpyext",
    not(feature = "sandbox"),
    any(target_os = "macos", target_os = "linux")
))]

mod cpyext_fixture;

use cpyext_fixture::Fixtures;

const BYTEARRAY_SCRIPT: &str = r#"
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
"#;

const COMPLEX_SCRIPT: &str = r#"
import cpyext_object_families as m

class CX(complex):
    pass

class HasComplex:
    def __complex__(self):
        return 3 + 4j

def eq(name, got, want):
    assert got == want, '%s: got %r, want %r' % (name, got, want)

eq('check(complex)', m.cx_check(1 + 2j), True)
eq('check(subclass)', m.cx_check(CX(1, 2)), True)
eq('check(float)', m.cx_check(2.5), False)
eq('check_exact(complex)', m.cx_check_exact(1 + 2j), True)
eq('check_exact(subclass)', m.cx_check_exact(CX(1, 2)), False)

eq('parts(complex)', m.cx_parts(1 + 2j), (1.0, 2.0))
eq('parts(subclass)', m.cx_parts(CX(1, 2)), (1.0, 2.0))
# A real number has a real part to report and no imaginary one.
eq('parts(float)', m.cx_parts(2.5), (2.5, 0.0))
eq('parts(int)', m.cx_parts(7), (7.0, 0.0))

eq('from_doubles', m.cx_from_doubles(1.5, -2.5), 1.5 - 2.5j)
# Out through the by-value struct and back, which is its own convention.
eq('round trip', m.cx_round_trip(3 - 4j), 3 - 4j)

eq('as_ccomplex(complex)', m.cx_as_ccomplex(1 + 2j), (1.0, 2.0))
eq('as_ccomplex(float)', m.cx_as_ccomplex(2.5), (2.5, 0.0))
eq('as_ccomplex(int)', m.cx_as_ccomplex(7), (7.0, 0.0))
eq('as_ccomplex(__complex__)', m.cx_as_ccomplex(HasComplex()), (3.0, 4.0))
# `complex('1+2j')` parses, but converting a *number* does not read a string.
eq('as_ccomplex(str) is refused', m.cx_as_ccomplex('1+2j'), 'ascomplex-failed')
eq('as_ccomplex(list) is refused', m.cx_as_ccomplex([]), 'ascomplex-failed')

# The pair sits in the block, so an extension that casts to `PyComplexObject`
# reads the same numbers the accessors answer -- and the class has to say the
# block is that large before the cast is allowed at all.
size, declared = m.cx_basicsize()
eq('complex says its block holds the pair', size >= declared, True)
eq('the block of a complex', m.cx_block(1 + 2j), (1.0, 2.0))
eq('the block of a negative complex', m.cx_block(-1.5 - 2.5j), (-1.5, -2.5))
# A subclass is sized as its base, so its block carries the pair too.
eq('the block of a subclass', m.cx_block(CX(1, 2)), (1.0, 2.0))
eq('the block of a float', m.cx_block(2.5), 'not-a-complex')

print('cpyext-complex-ok')
"#;

const WEAKREF_SCRIPT: &str = r#"
import gc

import cpyext_object_families as m

class Holder:
    pass

def collect():
    for _ in range(3):
        gc.collect()

def eq(name, got, want):
    assert got == want, '%s: got %r, want %r' % (name, got, want)

holder = Holder()
reference = m.wr_new_ref(holder)
eq('new_ref type', type(reference).__name__, 'ReferenceType')
eq('new_ref derefs', m.wr_get_object(reference) is holder, True)
eq('check(ref)', m.wr_check(reference), True)
eq('check_ref(ref)', m.wr_check_ref(reference), True)
eq('check_proxy(ref)', m.wr_check_proxy(reference), False)
eq('get_ref while alive', (lambda p: (p[0], p[1] is holder))(m.wr_get_ref(reference)), (1, True))
eq('is_dead while alive', m.wr_is_dead(reference), False)

proxy = m.wr_new_proxy(holder)
eq('new_proxy type', type(proxy).__name__, 'ProxyType')
eq('check(proxy)', m.wr_check(proxy), True)
eq('check_ref(proxy)', m.wr_check_ref(proxy), False)
eq('check_proxy(proxy)', m.wr_check_proxy(proxy), True)

fired = []
with_callback = m.wr_new_ref_with_callback(holder, lambda ref: fired.append('called'))
eq('a ref with a callback is a ref', m.wr_check_ref(with_callback), True)

eq('check(non-weakref)', m.wr_check([]), False)
eq('get_ref(non-weakref)', m.wr_get_ref([]), 'getref-failed')
eq('is_dead(non-weakref)', m.wr_is_dead([]), 'isdead-failed')

# Reading through a weak reference must not be what keeps the referent alive:
# `GetObject` answered above, and the referent still has to go.
del holder
collect()
eq('is_dead once the referent is gone', m.wr_is_dead(reference), True)
eq('get_ref once the referent is gone', m.wr_get_ref(reference), (0, None))
eq('get_object once the referent is gone', m.wr_get_object(reference), None)
eq('the callback ran', fired, ['called'])

# A C deallocator that breaks its own object's weak references, which is what
# every extension holding a weakref list does.  The assertion after it is the
# point: an entry point reached from a deallocator must leave nothing pending,
# or the next call inherits it and fails with a SystemError of its own.
before = m.wr_cleared_count()
victim = m.Cleared()
watch = m.wr_new_ref(victim)
del victim
for _ in range(4):
    if m.wr_cleared_count() > before:
        break
    gc.collect()
eq('the deallocator ran', m.wr_cleared_count(), before + 1)
eq('the weak reference is dead', m.wr_get_object(watch), None)
eq('and the call after it is clean', m.wr_check([]), False)

print('cpyext-weakref-ok')
"#;

#[test]
fn the_bytearray_entry_points() {
    let fixtures = Fixtures::new("cpyext-bytearray");
    fixtures.compile("cpyext_object_families");
    fixtures.expect_ok(BYTEARRAY_SCRIPT, &[], "cpyext-bytearray-ok");
}

#[test]
fn the_complex_entry_points() {
    let fixtures = Fixtures::new("cpyext-complex");
    fixtures.compile("cpyext_object_families");
    fixtures.expect_ok(COMPLEX_SCRIPT, &[], "cpyext-complex-ok");
}

const MEMORYVIEW_SCRIPT: &str = r#"
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

# A strided view is contiguous in no order, and only the read side has a
# copy to fall back on -- which is not built yet.
strided = memoryview(bytearray(b'abcdef'))[::2]
eq('a read view over a strided one', m.mv_contiguous(strided, 'r', 'C'),
   ('NotImplementedError',
    'creating contiguous readonly buffer from non-contiguous not implemented yet'))
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
"#;

#[test]
fn a_contiguous_view_over_an_exporter() {
    let fixtures = Fixtures::new("cpyext-memoryview");
    fixtures.compile("cpyext_object_families");
    fixtures.expect_ok(MEMORYVIEW_SCRIPT, &[], "cpyext-memoryview-ok");
}

const STRUCT_SCRIPT: &str = r#"
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
"#;

#[test]
fn the_counts_a_struct_carries_as_fields() {
    let fixtures = Fixtures::new("cpyext-struct");
    fixtures.compile("cpyext_object_families");
    fixtures.expect_ok(STRUCT_SCRIPT, &[], "cpyext-struct-ok");
}

#[test]
fn reading_through_a_weak_reference_does_not_keep_the_referent_alive() {
    let fixtures = Fixtures::new("cpyext-weakref");
    fixtures.compile("cpyext_object_families");
    fixtures.expect_ok(WEAKREF_SCRIPT, &[], "cpyext-weakref-ok");
}
