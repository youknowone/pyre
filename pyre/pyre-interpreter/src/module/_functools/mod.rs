//! _functools module — CPython accelerator imported by
//! `lib-python/3/functools.py`.
//!
//! `cmp_to_key` follows the stdlib fallback structurally: each invocation
//! creates a lexical `K`, capturing `mycmp` instead of exposing it on K.

use pyre_object::*;

crate::py_module! {
    "_functools",
    inline_app: {
        r#"
def cmp_to_key(mycmp):
    class K(object):
        __slots__ = ['obj']
        def __init__(self, obj):
            self.obj = obj
        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0
        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0
        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0
        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0
        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0
        __hash__ = None
    return K

# `_functools.cmp_to_key` is an interp-level builtin in CPython.  Unlike an
# app-level function, it therefore does not acquire an instance when a caller
# stores it on a class (the CPython functools tests do exactly that).  A
# callable staticmethod preserves the app-level implementation while giving
# the exported object the same non-binding descriptor behavior.
cmp_to_key = staticmethod(cmp_to_key)


_initial_missing = object()


def reduce(function, sequence, initial=_initial_missing):
    # _functoolsmodule.c functools_reduce — reduce(function, iterable[, initial]).
    try:
        it = iter(sequence)
    except TypeError:
        raise TypeError("reduce() arg 2 must support iteration") from None
    if initial is not _initial_missing:
        accum = initial
    else:
        try:
            accum = next(it)
        except StopIteration:
            raise TypeError(
                "reduce() of empty iterable with no initial value") from None
    for element in it:
        accum = function(accum, element)
    return accum


# Same descriptor-neutral accelerator surface as cmp_to_key above.
reduce = staticmethod(reduce)
"# => ["cmp_to_key", "reduce"],
    },
}
