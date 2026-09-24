# cpyext-fixture: cpyext_object_families
# cpyext-expect: cpyext-weakref-ok

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
eq('the bound C method works', victim.ping(), None)
del victim
gc.collect()
eq('the deallocator ran', m.wr_cleared_count(), before + 1)
eq('the weak reference is dead', m.wr_get_object(watch), None)
eq('and the call after it is clean', m.wr_check([]), False)

print('cpyext-weakref-ok')
