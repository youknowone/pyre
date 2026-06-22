"""App-level ``defaultdict`` for the ``_collections`` module.

``defaultdict`` lives at app-level because, like PyPy, pyre cannot express an
interp-level type that subclasses the app-level ``dict``
(``app_defaultdict.py`` in PyPy exists for the same reason).  The class
mirrors ``_collectionsmodule.c``'s ``defaultdict``:

* ``__init__`` takes the ``default_factory`` (``None`` or a callable) as the
  first positional argument and forwards the rest to ``dict.__init__``.
* ``__missing__`` stores and returns ``default_factory()`` for an absent key,
  or raises ``KeyError(key)`` when there is no factory.
* ``__reduce__`` returns ``(type, args, None, None, iter(items))`` where
  ``args`` is ``()`` for a ``None`` factory and ``(factory,)`` otherwise.
* ``copy``/``__copy__``, ``__or__``/``__ror__`` and ``__repr__`` preserve the
  exact type and the factory.

Two deliberate departures from ``_collectionsmodule.c``, both forced by pyre
storing a dict subclass's items in an instance attribute (which precludes
``__slots__`` on the subclass, the device PyPy and CPython use to keep the
factory off the instance dict):

* the factory is held in the ordinary instance dict rather than a slot, so a
  ``defaultdict`` instance carries a ``__dict__``;
* ``__missing__`` is written here rather than at interp-level (PyPy keeps it
  interp-level only for thread atomicity; the behaviour is identical).
"""


class defaultdict(dict):
    __module__ = 'collections'

    def __init__(self, *args, **kwds):
        if args:
            default_factory = args[0]
            args = args[1:]
            if default_factory is not None and not callable(default_factory):
                raise TypeError("first argument must be callable or None")
        else:
            default_factory = None
        self.default_factory = default_factory
        dict.__init__(self, *args, **kwds)

    def __missing__(self, key):
        factory = self.default_factory
        if factory is None:
            raise KeyError(key)
        self[key] = value = factory()
        return value

    def __reduce__(self):
        if self.default_factory is None:
            args = ()
        else:
            args = (self.default_factory,)
        return type(self), args, None, None, iter(self.items())

    def copy(self):
        return type(self)(self.default_factory, self)

    __copy__ = copy

    def __or__(self, other):
        if not isinstance(other, dict):
            return NotImplemented
        new = type(self)(self.default_factory)
        new.update(self)
        new.update(other)
        return new

    def __ror__(self, other):
        if not isinstance(other, dict):
            return NotImplemented
        new = type(self)(self.default_factory)
        new.update(other)
        new.update(self)
        return new

    def __repr__(self, recurse=set()):
        # ``defdict_repr``: "<typename>(<factory repr>, <dict repr>)".  The
        # factory repr is recursion-guarded so a factory that reprs back to the
        # dict renders as "..." instead of recursing forever; the dict part
        # rides ``dict.__repr__`` which renders a self-referential dict as
        # "{...}".  (Not thread-safe, but good enough.)
        dictrepr = dict.__repr__(self)
        if id(self) in recurse:
            factoryrepr = "..."
        else:
            try:
                recurse.add(id(self))
                factoryrepr = repr(self.default_factory)
            finally:
                recurse.remove(id(self))
        return "%s(%s, %s)" % (type(self).__name__, factoryrepr, dictrepr)
