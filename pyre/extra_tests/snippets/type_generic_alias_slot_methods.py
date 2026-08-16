# pyre-check: gate=1

GA = type(list[int])
required = {
    '__repr__', '__hash__', '__call__',
    '__getattribute__', '__iter__', '__dir__',
    '__ne__', '__lt__', '__le__', '__gt__', '__ge__',
}
assert required <= GA.__dict__.keys()
assert GA.__repr__(list[int]) == 'list[int]'
assert GA.__hash__(list[int]) == hash(list[int])
assert GA.__getattribute__(list[int], '__origin__') is list
assert list(GA.__iter__(tuple[int, str])) == [(*tuple[int, str],)[0]]
assert '__origin__' in GA.__dir__(list[int])
assert '__bases__' not in GA.__dir__(list[int])

class C:
    def __init__(self, *, value):
        self.value = value

alias = GA(C, int)
instance = GA.__call__(alias, value=42)
assert instance.value == 42
assert instance.__orig_class__ is alias

# Instance lookup still follows PyPy GenericAlias.__getattribute__: names
# outside _ATTR_EXCEPTIONS delegate to the origin despite the type-dict rows.
assert list[int].__repr__ is list.__repr__
assert list[int].__hash__ is list.__hash__
assert GA.__ne__(list[int], list[str]) is True
assert GA.__ne__(list[int], list[int]) is False
assert GA.__ne__(list[int], list) is NotImplemented
assert GA.__lt__(list[int], list[str]) is NotImplemented
