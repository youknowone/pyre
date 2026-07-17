# app_operator.py — app-level helpers for the operator module.
# attrgetter / itemgetter / methodcaller are callable factory classes that
# the interp-level operator module cannot express as plain functions.
# Mirrors pypy/module/operator/app_operator.py.

__name__ = 'operator'


def countOf(a, b):
    'countOf(a, b) -- Return the number of times b occurs in a.'
    count = 0
    for x in a:
        if x is b or x == b:
            count += 1
    return count


def concat(a, b):
    "Same as a + b, for a and b sequences."
    if not hasattr(a, '__getitem__'):
        msg = "'%s' object can't be concatenated" % type(a).__name__
        raise TypeError(msg)
    return a + b


def indexOf(a, b):
    "Return the first index of b in a."
    for i, j in enumerate(a):
        if j is b or j == b:
            return i
    else:
        raise ValueError('sequence.index(x): x not in sequence')


def inv(a):
    "Same as ~a."
    return ~a


def is_none(a):
    "Same as a is None."
    return a is None


def is_not_none(a):
    "Same as a is not None."
    return a is not None


def call(obj, /, *args, **kwargs):
    "Same as obj(*args, **kwargs)."
    return obj(*args, **kwargs)


class attrgetter(object):
    """
    Return a callable object that fetches the given attribute(s) from its operand.
    After f = attrgetter('name'), the call f(r) returns r.name.
    After g = attrgetter('name', 'date'), the call g(r) returns (r.name, r.date).
    After h = attrgetter('name.first', 'name.last'), the call h(r) returns
    (r.name.first, r.name.last).
    """

    def __init__(self, attr, *attrs):
        if (
            not isinstance(attr, str) or
            not all(isinstance(a, str) for a in attrs)
        ):
            raise TypeError("attribute name must be a string, not %r" %
                            type(attr).__name__)
        elif attrs:
            self._multi_attrs = [
                a.split(".") for a in [attr] + list(attrs)
            ]
            self._call = self._multi_attrgetter
        elif "." not in attr:
            self._simple_attr = attr
            self._call = self._simple_attrgetter
        else:
            self._single_attr = attr.split(".")
            self._call = self._single_attrgetter

    def __call__(self, obj):
        return self._call(obj)

    def _simple_attrgetter(self, obj):
        return getattr(obj, self._simple_attr)

    def _single_attrgetter(self, obj):
        for name in self._single_attr:
            obj = getattr(obj, name)
        return obj

    def _multi_attrgetter(self, obj):
        result = []
        for names in self._multi_attrs:
            o = obj
            for name in names:
                o = getattr(o, name)
            result.append(o)
        return tuple(result)

    def __reduce__(self):
        try:
            attrs = (self._simple_attr,)
        except AttributeError:
            try:
                attrs = ('.'.join(self._single_attr),)
            except AttributeError:
                attrs = tuple('.'.join(a) for a in self._multi_attrs)
        return (type(self), attrs)

    def __repr__(self):
        try:
            a = repr(self._simple_attr)
        except AttributeError:
            try:
                a = repr('.'.join(self._single_attr))
            except AttributeError:
                lst = self._multi_attrs
                a = ', '.join([repr('.'.join(a1)) for a1 in lst])
        return 'operator.attrgetter(%s)' % (a,)


class itemgetter(object):
    """
    Return a callable object that fetches the given item(s) from its operand.
    After f = itemgetter(2), the call f(r) returns r[2].
    After g = itemgetter(2, 5, 3), the call g(r) returns (r[2], r[5], r[3])
    """

    def __init__(self, item, *items):
        self._single = not bool(items)
        if self._single:
            self._idx = item
        else:
            self._idx = [item] + list(items)

    def __call__(self, obj):
        if self._single:
            return obj[self._idx]
        else:
            return tuple([obj[i] for i in self._idx])

    def __repr__(self):
        if self._single:
            a = repr(self._idx)
        else:
            a = ', '.join([repr(i) for i in self._idx])
        return 'operator.itemgetter(%s)' % (a,)


class methodcaller(object):
    """
    Return a callable object that calls the given method on its operand.
    After f = methodcaller('name'), the call f(r) returns r.name().
    After g = methodcaller('name', 'date', foo=1), the call g(r) returns
    r.name('date', foo=1).
    """

    def __init__(*args, **kwargs):
        if len(args) < 2:
            raise TypeError("methodcaller() called with not enough arguments")
        self, method_name = args[:2]
        if not isinstance(method_name, str):
            raise TypeError("method name must be a string")
        self._method_name = method_name
        self._args = args[2:]
        self._kwargs = kwargs

    def __call__(self, obj):
        return getattr(obj, self._method_name)(*self._args, **self._kwargs)

    def __repr__(self):
        args = [repr(self._method_name)]
        for a in self._args:
            args.append(repr(a))
        for key, value in self._kwargs.items():
            args.append('%s=%r' % (key, value))
        return 'operator.methodcaller(%s)' % (', '.join(args),)
