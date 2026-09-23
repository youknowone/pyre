# CPython-suite gap: `test_pickle` reaches this only by accident.  Its
# `load_tests` runs `doctest.DocTestSuite(pickle)`, which hashes
# `struct.pack.__module__` long after `_struct` may have been dropped from
# `sys.modules`; whether that crashes depends on what reused the freed memory,
# so the suite goes red on one runner layout and stays green on another.
#
# parity-tests reason: a builtin function stays reachable after the module that
# defined it is collected, and it owns its `__module__` string and its module
# object.  Those fields must keep their referents alive for as long as the
# function is reachable from any holder, not only while the defining module's
# dict is.  A plain container holding the function is enough to observe it.
import gc
import sys

import _struct

kept = [_struct.pack, _struct.unpack, _struct.calcsize]
for name in ('struct', '_struct'):
    sys.modules.pop(name, None)
del _struct
for _ in range(5):
    gc.collect()

# Reuse whatever the collections freed.
junk = ['%s.%s.%d' % ('__main__', 'CUnpicklerTests', i) for i in range(200000)]
del junk

for f in kept:
    assert f.__module__ == '_struct', (f.__name__, f.__module__)
    assert f.__self__.__name__ == '_struct', (f.__name__, f.__self__)
assert len({f.__module__: None for f in kept}) == 1
print('OK')
