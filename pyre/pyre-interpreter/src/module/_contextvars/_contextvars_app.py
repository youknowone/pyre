from __pypy__ import get_contextvar_context, set_contextvar_context
from _immutables_map import Map
try:
    from __pypy__ import hidden_applevel
except ImportError:
    hidden_applevel = lambda f: f


class Unsubclassable(type):
    def __new__(cls, name, bases, dct):
        for base in bases:
            if isinstance(base, Unsubclassable):
                raise TypeError(f"type '{base.__name__}' is not an acceptable base type")
        return type.__new__(cls, name, bases, dict(dct))


def get_context():
    context = get_contextvar_context()
    if context is None:
        context = Context()
        set_contextvar_context(context)
    return context


class Context(metaclass=Unsubclassable):

    #_data: Map
    #_is_entered: bool

    def __init__(self, *args, **kwargs):
        # CPython 3.14 context_new() rejects both positional and keyword
        # arguments with this dedicated error, while still accepting **{}.
        if args or kwargs:
            raise TypeError("Context() does not accept any arguments")
        self._data = Map()
        self._is_entered = False

    @hidden_applevel
    def run(self, callable, *args, **kwargs):
        if self._is_entered:
            raise RuntimeError(
                f'cannot enter context: {self} is already entered')

        # don't use get_context() here, to avoid creating a Context object
        _prev_context = get_contextvar_context()
        try:
            self._is_entered = True
            set_contextvar_context(self)
            return callable(*args, **kwargs)
        finally:
            set_contextvar_context(_prev_context)
            self._is_entered = False

    def copy(self):
        new = Context()
        new._data = self._data
        return new

    # Implement abstract Mapping.__getitem__
    def __getitem__(self, var):
        if not isinstance(var, ContextVar):
            raise TypeError("ContextVar key was expected")
        return self._data[var]

    # Implement abstract Mapping.__contains__
    def __contains__(self, var):
        if not isinstance(var, ContextVar):
            raise TypeError("ContextVar key was expected")
        return var in self._data

    # Implement abstract Mapping.__len__
    def __len__(self):
        return len(self._data)

    # Implement abstract Mapping.__iter__
    def __iter__(self):
        return iter(self._data)

    def get(self, key, default=None):
        if not isinstance(key, ContextVar):
            raise TypeError("ContextVar key was expected")
        try:
            return self._data[key]
        except KeyError:
            return default

    def keys(self):
        from collections.abc import KeysView
        return KeysView(self)

    def values(self):
        from collections.abc import ValuesView
        return ValuesView(self)

    def items(self):
        from collections.abc import ItemsView
        return ItemsView(self)

    def __eq__(self, other):
        if not isinstance(other, Context):
            return NotImplemented
        return dict(self.items()) == dict(other.items())


def copy_context():
    return get_context().copy()


Context.__module__ = '_contextvars'
