//! _functools module — CPython accelerator imported by
//! `lib-python/3/functools.py`.
//!
//! `cmp_to_key` follows the stdlib fallback structurally: each invocation
//! creates a lexical `K`, capturing `mycmp` instead of exposing it on K.
//! `partial` follows PyPy's `lib_pypy/_functools.py`: its public state is
//! exposed through read-only properties backed by private slots. Placeholder
//! argument merging follows the Python 3.14 `functools.py` implementation.

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


# PyPy: lib_pypy/_functools.py `partial`, extended with the Placeholder
# semantics introduced in Python 3.14. The private storage plus read-only
# properties is intentional: it is the upstream accelerator object shape.
from operator import itemgetter as _partial_itemgetter
from reprlib import recursive_repr as _partial_recursive_repr
from types import GenericAlias as _PartialGenericAlias
from types import MethodType as _PartialMethodType


class _PlaceholderType:
    __instance = None
    __slots__ = ()

    def __init_subclass__(cls, *args, **kwargs):
        raise TypeError(f"type '{cls.__name__}' is not an acceptable base type")

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = object.__new__(cls)
        return cls.__instance

    def __repr__(self):
        return "Placeholder"

    def __reduce__(self):
        return "Placeholder"


Placeholder = _PlaceholderType()


def _partial_prepare_merger(args):
    if not args:
        return 0, None
    nargs = len(args)
    order = []
    j = nargs
    for i, arg in enumerate(args):
        if arg is Placeholder:
            order.append(j)
            j += 1
        else:
            order.append(i)
    phcount = j - nargs
    merger = _partial_itemgetter(*order) if phcount else None
    return phcount, merger


def _partial_new(cls, func, /, *args, **keywords):
    if not callable(func):
        raise TypeError("the first argument must be callable")
    if args and args[-1] is Placeholder:
        raise TypeError("trailing Placeholders are not allowed")
    for value in keywords.values():
        if value is Placeholder:
            raise TypeError("Placeholder cannot be passed as a keyword argument")

    if isinstance(func, partial):
        pto_phcount = func._phcount
        tot_args = func.args
        if args:
            tot_args += args
            if pto_phcount:
                nargs = len(args)
                if nargs < pto_phcount:
                    tot_args += (Placeholder,) * (pto_phcount - nargs)
                tot_args = func._merger(tot_args)
                if nargs > pto_phcount:
                    tot_args += args[pto_phcount:]
            phcount, merger = _partial_prepare_merger(tot_args)
        else:
            phcount, merger = pto_phcount, func._merger
        keywords = {**func.keywords, **keywords}
        func = func.func
    else:
        tot_args = args
        phcount, merger = _partial_prepare_merger(tot_args)

    self = object.__new__(cls)
    self._func = func
    self._args = tot_args
    self._keywords = keywords
    self._phcount = phcount
    self._merger = merger
    return self


def _partial_repr(self):
    cls = type(self)
    module = cls.__module__
    qualname = cls.__qualname__
    func, p_args, keywords = self.func, self.args, self.keywords
    args = [repr(func)]
    args.extend(map(repr, p_args))
    args.extend(f"{key}={value!r}" for key, value in keywords.items())
    return f"{module}.{qualname}({', '.join(args)})"


class partial(object):
    """New function with partial application of the given arguments
    and keywords.
    """

    __slots__ = ("_func", "_args", "_keywords", "_phcount", "_merger",
                 "__dict__", "__weakref__")

    __new__ = _partial_new
    __repr__ = _partial_recursive_repr()(_partial_repr)

    @property
    def func(self):
        return self._func

    @property
    def args(self):
        return self._args

    @property
    def keywords(self):
        return self._keywords

    def __delattr__(self, name):
        if name == "__dict__":
            raise TypeError("a partial object's dictionary may not be deleted")
        object.__delattr__(self, name)

    def __call__(self, /, *args, **keywords):
        phcount = self._phcount
        if phcount:
            try:
                pto_args = self._merger(self.args + args)
                args = args[phcount:]
            except IndexError:
                raise TypeError("missing positional arguments "
                                "in 'partial' call; expected "
                                f"at least {phcount}, got {len(args)}")
        else:
            pto_args = self.args
        keywords = {**self.keywords, **keywords}
        return self.func(*pto_args, *args, **keywords)

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return _PartialMethodType(self, obj)

    def __reduce__(self):
        return (type(self), (self.func,),
                (self.func, self.args, self.keywords or None,
                 self.__dict__ or None))

    def __setstate__(self, state):
        if not isinstance(state, tuple):
            raise TypeError("argument to __setstate__ must be a tuple")
        if len(state) != 4:
            raise TypeError(f"expected 4 items in state, got {len(state)}")
        func, args, keywords, namespace = state
        if (not callable(func) or not isinstance(args, tuple) or
                (keywords is not None and not isinstance(keywords, dict)) or
                (namespace is not None and not isinstance(namespace, dict))):
            raise TypeError("invalid partial state")
        if args and args[-1] is Placeholder:
            raise TypeError("trailing Placeholders are not allowed")

        phcount, merger = _partial_prepare_merger(args)
        args = tuple(args)
        if keywords is None:
            keywords = {}
        elif type(keywords) is not dict:
            keywords = dict(keywords)
        if namespace is None:
            namespace = {}

        self.__dict__ = namespace
        self._func = func
        self._args = args
        self._keywords = keywords
        self._phcount = phcount
        self._merger = merger

    __class_getitem__ = classmethod(_PartialGenericAlias)


# CPython exposes these accelerator types from the public `functools` module.
partial.__module__ = "functools"
_PlaceholderType.__module__ = "functools"
"# => ["cmp_to_key", "reduce", "partial", "Placeholder", "_PlaceholderType"],
    },
}
