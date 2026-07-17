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


def reduce(*args):
    # _functoolsmodule.c functools_reduce — reduce(function, iterable[, initial]).
    if len(args) < 2:
        raise TypeError(
            "reduce() takes at least 2 positional arguments (%d given)" % len(args))
    if len(args) > 3:
        raise TypeError(
            "reduce() takes at most 3 arguments (%d given)" % len(args))
    function = args[0]
    try:
        it = iter(args[1])
    except TypeError:
        raise TypeError("reduce() arg 2 must support iteration") from None
    if len(args) == 3:
        accum = args[2]
    else:
        try:
            accum = next(it)
        except StopIteration:
            raise TypeError(
                "reduce() of empty iterable with no initial value") from None
    for element in it:
        accum = function(accum, element)
    return accum
"# => ["cmp_to_key", "reduce"],
    },
}
