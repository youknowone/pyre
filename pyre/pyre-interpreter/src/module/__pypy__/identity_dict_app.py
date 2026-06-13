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

    def __contains__(self, key):
        return id(key) in self._d

    def get(self, key, default=None):
        return self._d.get(id(key), default)

    def __len__(self):
        return len(self._d)

    def clear(self):
        self._d.clear()
