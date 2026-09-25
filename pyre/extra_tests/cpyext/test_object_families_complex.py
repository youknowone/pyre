# cpyext-fixture: cpyext_object_families
# cpyext-expect: cpyext-complex-ok

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
