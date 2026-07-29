# type.__name__ is the bare type name; a dotted tp_name keeps its module
# prefix only in repr.

assert int.__name__ == "int"
assert list.__name__ == "list"


class Foo:
    pass


assert Foo.__name__ == "Foo"

# PEP 604 unions: 3.14 unified `types.UnionType` with `typing.Union`, so the
# type reports the bare `Union` while its repr keeps the module prefix.
u = int | str
assert type(u).__name__ == "Union"
assert repr(type(u)) == "<class 'typing.Union'>"

print("type_name_bare ok")
