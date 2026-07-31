class identity_dict(object):
    """A mapping keyed by object identity rather than equality.

    Stores the key alongside each value so unhashable objects (lists,
    dicts, sets) work as keys and the key's identity remains valid for
    the lifetime of the entry.  This mirrors PyPy's interp-level
    ``W_IdentityDict.dict``.
    """

    def __init__(self):
        self._d = {}

    def __getitem__(self, key):
        return self._d[id(key)][1]

    def __setitem__(self, key, value):
        self._d[id(key)] = (key, value)

    def __delitem__(self, key):
        del self._d[id(key)]

    def __contains__(self, key):
        return id(key) in self._d

    def get(self, key, default=None):
        entry = self._d.get(id(key))
        return default if entry is None else entry[1]

    def __len__(self):
        return len(self._d)

    def clear(self):
        self._d.clear()

    def keys(self):
        return [key for key, _ in self._d.values()]

    def values(self):
        return [value for _, value in self._d.values()]

    def __iter__(self):
        raise TypeError("'identity_dict' object does not support iteration; "
                        "iterate over x.keys()")
