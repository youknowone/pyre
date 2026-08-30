"""Runtime objects for t-string template literals (PEP 750).

CPython implements Template and Interpolation in C and exposes them through
`string.templatelib`, whose `Template = type(t"{0}")` recovers the type from a
template literal.  Here they are app-level Python that the BUILD_TEMPLATE and
BUILD_INTERPOLATION opcodes construct through `_build_template` /
`_build_interpolation`.
"""

import itertools
from types import GenericAlias

# The BUILD_INTERPOLATION conversion oparg field: 0 means no conversion.
_CONVERSIONS = (None, 's', 'r', 'a')


class Interpolation:
    __match_args__ = ('value', 'expression', 'conversion', 'format_spec')

    # CPython 3.14 Objects/interpolationobject.c: interpolation_methods —
    # Py_GenericAlias with METH_CLASS.
    @classmethod
    def __class_getitem__(cls, item):
        return GenericAlias(cls, item)

    def __init__(self, value, expression='', conversion=None, format_spec=''):
        # `expression` / `format_spec` are typed `str` and `conversion` is
        # restricted to None / 's' / 'r' / 'a'; reject anything else rather
        # than store it unchecked.
        if not isinstance(expression, str):
            raise TypeError(
                "Interpolation() argument 'expression' must be str, "
                f'not {type(expression).__name__}')
        if conversion is not None and conversion not in ('s', 'r', 'a'):
            raise ValueError(
                "Interpolation() argument 'conversion' must be one of "
                "'s', 'r', or 'a'")
        if not isinstance(format_spec, str):
            raise TypeError(
                "Interpolation() argument 'format_spec' must be str, "
                f'not {type(format_spec).__name__}')
        self._value = value
        self._expression = expression
        self._conversion = conversion
        self._format_spec = format_spec

    @property
    def value(self):
        return self._value

    @property
    def expression(self):
        return self._expression

    @property
    def conversion(self):
        return self._conversion

    @property
    def format_spec(self):
        return self._format_spec

    def __repr__(self):
        return (f'Interpolation({self._value!r}, {self._expression!r}, '
                f'{self._conversion!r}, {self._format_spec!r})')

    def __reduce__(self):
        # CPython 3.14 `interpolation_reduce`: reconstruct through the exact
        # public type and the four constructor fields, in their storage order.
        return (type(self), (self._value, self._expression,
                             self._conversion, self._format_spec))


class Template:
    # CPython 3.14 Objects/templateobject.c: template_methods —
    # Py_GenericAlias with METH_CLASS.
    @classmethod
    def __class_getitem__(cls, item):
        return GenericAlias(cls, item)

    def __new__(cls, *args):
        # Public constructor: interleaved strings and Interpolations.  Adjacent
        # strings are merged and the result always begins and ends with a
        # string, so `len(strings) == len(interpolations) + 1`.
        strings = []
        interpolations = []
        current = ''
        for arg in args:
            if isinstance(arg, str):
                current += arg
            elif isinstance(arg, Interpolation):
                strings.append(current)
                current = ''
                interpolations.append(arg)
            else:
                raise TypeError('Template.__new__ *args need to be of type '
                                "'str' or 'Interpolation'")
        strings.append(current)
        return cls._make(tuple(strings), tuple(interpolations))

    @classmethod
    def _make(cls, strings, interpolations):
        self = object.__new__(cls)
        self._strings = strings
        self._interpolations = interpolations
        return self

    @property
    def strings(self):
        return self._strings

    @property
    def interpolations(self):
        return self._interpolations

    @property
    def values(self):
        return tuple(i.value for i in self._interpolations)

    def __iter__(self):
        for string, interpolation in itertools.zip_longest(
                self._strings, self._interpolations):
            if string:
                yield string
            if interpolation is not None:
                yield interpolation

    def __add__(self, other):
        # Only two Templates concatenate; a str operand is rejected so callers
        # must say whether it is static text (Template(...)) or dynamic data
        # (Interpolation(...)) rather than have it inferred.
        if isinstance(other, Template):
            return Template._make(
                _concat_boundary(self._strings, other._strings),
                self._interpolations + other._interpolations)
        raise TypeError(
            'can only concatenate string.templatelib.Template '
            f'(not "{type(other).__name__}") to string.templatelib.Template')

    def __repr__(self):
        return (f'Template(strings={self._strings!r}, '
                f'interpolations={self._interpolations!r})')

    def __reduce__(self):
        # CPython 3.14 `template_reduce` resolves the public app-level helper
        # instead of making the private runtime module part of the pickle.
        from string.templatelib import _template_unpickle
        return (_template_unpickle, (self._strings, self._interpolations))


def _concat_boundary(left, right):
    # Merge the touching boundary strings of two templates: the last static
    # string of `left` joins the first of `right`.
    return left[:-1] + (left[-1] + right[0],) + right[1:]


def _build_template(strings, interpolations):
    # BUILD_TEMPLATE: the compiler already split the literal into the static
    # string parts and the interpolations, with one more string than
    # interpolation.
    return Template._make(tuple(strings), tuple(interpolations))


def _build_interpolation(value, expression, conversion, format_spec):
    # BUILD_INTERPOLATION: `conversion` is the opcode's conversion oparg field.
    # CPython `_PyInterpolation_Build` is an internal unchecked constructor;
    # keep it separate from the validating public constructor above.
    interpolation = object.__new__(Interpolation)
    interpolation._value = value
    interpolation._expression = expression
    interpolation._conversion = _CONVERSIONS[conversion]
    interpolation._format_spec = format_spec
    return interpolation


# Present the types as living in their public home, `string.templatelib`, where
# `Template = type(t"{0}")` rebinds them.
for _cls in (Template, Interpolation):
    _cls.__module__ = 'string.templatelib'
del _cls
