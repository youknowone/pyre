"""A lone surrogate survives str()/repr() round trips through the object protocol.

A Python `str` may hold an unpaired surrogate -- `surrogateescape` puts one there
for every undecodable filesystem byte -- so any path that rebuilds a `str` from a
`__str__`/`__repr__` result has to carry it.  Assembling the result through a
lossy UTF-8 encode instead substitutes U+FFFD, which silently changes the value:
`f"{obj}"` stops equalling `str(obj)`, and two distinct strings compare equal.

`str()` and `%`-formatting were already lossless here; the entry points below are
the ones that rebuilt the value through a `String`.
"""

import weakref

S = "s\udcffz"
FFFD = "�"


class Str:
    def __str__(self):
        return S


class Seq:
    """A non-list/tuple sequence -- the group constructor saves its repr()."""

    def __repr__(self):
        return S

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if index == 0:
            return ValueError("v")
        raise IndexError(index)


def check_format():
    # `object.__format__` with an empty spec falls through to `str(self)`.
    assert f"{Str()}" == S, ascii(f"{Str()}")
    assert format(Str()) == S, ascii(format(Str()))
    assert format(Str(), "") == S, ascii(format(Str(), ""))
    # The two that were already lossless, kept as controls: if these ever break,
    # the cause is upstream of the entry points above.
    assert "%s" % Str() == S, ascii("%s" % Str())
    assert str(Str()) == S, ascii(str(Str()))


def check_weakref_proxy():
    obj = Str()
    proxy = weakref.proxy(obj)
    assert str(proxy) == S, ascii(str(proxy))


def check_exception_group_repr():
    group = ExceptionGroup("m", Seq())
    # The constructor records the sequence's repr() and `__repr__` replays it.
    assert repr(group) == "ExceptionGroup('m', %s)" % S, ascii(repr(group))
    assert FFFD not in repr(group), ascii(repr(group))


check_format()
check_weakref_proxy()
check_exception_group_repr()
print("OK")
