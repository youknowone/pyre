# pyre-check: gate=1
# typeobject.py:520-523 W_TypeObject.check_user_subclass refuses
# `set.__new__(int)` (and similar cross-layout calls) before the
# base allocator runs. pyre's `check_user_subclass` enforces the
# same layout-typedef identity guard.
err = ''
try:
    set.__new__(int)
except TypeError as e:
    err = str(e)
frozen_err = ''
try:
    frozenset.__new__(int, [1, 2])
except TypeError as e:
    frozen_err = str(e)

assert err
assert frozen_err
