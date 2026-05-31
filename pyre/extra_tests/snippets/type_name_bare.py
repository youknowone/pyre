# type.__name__ is the bare type name; a dotted tp_name keeps its module
# prefix only in repr.

assert int.__name__ == "int"
assert list.__name__ == "list"


class Foo:
    pass


assert Foo.__name__ == "Foo"

# types.UnionType (PEP 604): __name__ strips the module prefix, repr keeps it.
u = int | str
assert type(u).__name__ == "UnionType"
assert repr(type(u)) == "<class 'types.UnionType'>"

print("type_name_bare ok")
