# pyre-check: gate=1
# setobject.py:160 W_SetObject.descr_init parses against
# `init_signature = Signature(['some_iterable'])`, so anything
# beyond `(self, iterable)` is a TypeError; setobject.py:631
# W_FrozensetObject.descr_new2 has the gateway-level fixed maxargs
# for `(space, w_frozensettype, w_iterable=None)`.
init_err = ''
try:
    set([1], 2)
except TypeError as e:
    init_err = str(e)
init_direct_err = ''
try:
    s = set()
    set.__init__(s, [1], 2)
except TypeError as e:
    init_direct_err = str(e)
frozen_err = ''
try:
    frozenset([1], 2)
except TypeError as e:
    frozen_err = str(e)
frozen_new_err = ''
try:
    frozenset.__new__(frozenset, [1], 2)
except TypeError as e:
    frozen_new_err = str(e)

assert init_err
assert init_direct_err
assert frozen_err
assert frozen_new_err
