class identity_dict(object):
    """A mapping keyed by object identity rather than equality.

    Stores entries in an internal dict keyed on ``id(key)`` so that
    unhashable objects (lists, dicts, sets) work as keys.  The value
    side is expected to keep the key object alive for the dict's
    lifetime, so ``id(key)`` stays valid.
    """

    def __init__(self):
        self._d = {}

    def __getitem__(self, key):
        return self._d[id(key)]

    def __setitem__(self, key, value):
        self._d[id(key)] = value

    def __delitem__(self, key):
        del self._d[id(key)]

    def __contains__(self, key):
        return id(key) in self._d

    def get(self, key, default=None):
        return self._d.get(id(key), default)

    def __len__(self):
        return len(self._d)

    def clear(self):
        self._d.clear()


def reversed_dict(d):
    """Enumerate the keys of a dict in reversed order.

    A ``__pypy__`` primitive so ``collections.OrderedDict`` can implement
    ``__reversed__`` even though CPython dicts are unordered.  Going
    through the unbound ``dict`` slot bypasses any ``__reversed__`` a
    subclass installs, matching the interp-level
    ``W_DictMultiObject.descr_reversed``.
    """
    if not isinstance(d, dict):
        raise TypeError("reversed_dict() argument must be a dict")
    return dict.__reversed__(d)


def move_to_end(d, key, last=True):
    """Move an existing key to the end (or the front if last is False).

    Raises KeyError if the key does not exist.  A ``__pypy__`` primitive
    backing ``collections.OrderedDict.move_to_end``.
    """
    if not isinstance(d, dict):
        raise TypeError("move_to_end() argument must be a dict")
    value = dict.__getitem__(d, key)  # raises KeyError when key is missing
    if last:
        dict.__delitem__(d, key)
        dict.__setitem__(d, key, value)
    else:
        dict.__delitem__(d, key)
        pairs = [(k, dict.__getitem__(d, k)) for k in list(dict.__iter__(d))]
        for k, _ in pairs:
            dict.__delitem__(d, k)
        dict.__setitem__(d, key, value)
        for k, v in pairs:
            dict.__setitem__(d, k, v)


# Persistent identity dict of objects currently being repr()'d.  PyPy keeps
# this per-thread on the ExecutionContext; pyre keeps a single module-global
# instance, which serves the same self-recursion guard under pyre's execution
# model.  It stays empty between reprs (each __repr__ removes itself in a
# finally block).
_objects_in_repr = identity_dict()


def objects_in_repr():
    """The identity dict of objects currently being repr()'d.

    ``__repr__`` methods (e.g. ``collections.OrderedDict``) use it to
    emit ``...`` instead of recursing on a self-referential container.
    """
    return _objects_in_repr
