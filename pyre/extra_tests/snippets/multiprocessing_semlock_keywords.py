# pyre-check: gate=1
# `interp_semaphore.py` declares SemLock's parameters through `@unwrap_spec`,
# so the constructor binds them by name as well as by position.
import os
import sys

if sys.platform == 'win32':
    # The windows SemLock takes a different parameter set.
    raise SystemExit(0)

import _multiprocessing

RECURSIVE_MUTEX = 0
base = f'/pyre-semlock-{os.getpid()}'

by_keyword = _multiprocessing.SemLock(
    RECURSIVE_MUTEX, 1, 1, name=f'{base}-kw', unlink=True
)
assert by_keyword.kind == RECURSIVE_MUTEX
assert by_keyword.maxvalue == 1

# Only `subtype` is positional-only, so the first three bind by name as well.
by_all_keywords = _multiprocessing.SemLock(
    kind=RECURSIVE_MUTEX, value=1, maxvalue=1, name=f'{base}-all-kw', unlink=True
)
assert by_all_keywords.kind == RECURSIVE_MUTEX
assert by_all_keywords.maxvalue == 1

by_position = _multiprocessing.SemLock(
    RECURSIVE_MUTEX, 1, 1, f'{base}-pos', True
)
assert by_position.kind == RECURSIVE_MUTEX

# The name is not positional-only either way round.
mixed = _multiprocessing.SemLock(
    RECURSIVE_MUTEX, 1, 1, unlink=True, name=f'{base}-mixed'
)
assert mixed.maxvalue == 1

# `kind` outside the two known values is still rejected.
try:
    _multiprocessing.SemLock(99, 1, 1, name=f'{base}-bad', unlink=True)
except ValueError:
    pass
else:
    raise AssertionError('unrecognized kind must raise ValueError')
