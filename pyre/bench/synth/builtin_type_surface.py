"""Version-neutral core surface for the completed set/singleton slice.

The exhaustive 3.14-only names live in ``extra_tests/parity_tests``.  Synthetic
baselines intentionally compare CPython 3.14 with the bundled PyPy 3.11 first,
so this file probes the common structural core instead of dumping raw type
dictionaries whose doc/version wrapper slots legitimately differ.
"""


def show(name, cls, required):
    print(name, "->", all(slot in cls.__dict__ for slot in required))


def main():
    for name, cls, required in [
        ("NoneType", type(None), ["__new__", "__repr__"]),
        ("NotImplementedType", type(NotImplemented), ["__new__", "__repr__"]),
        ("set", set, ["__new__", "__and__", "__or__", "__iter__", "add", "union"]),
        ("frozenset", frozenset, ["__new__", "__hash__", "__iter__", "union"]),
        (
            "set_iterator",
            type(iter(set())),
            ["__iter__", "__next__", "__length_hint__", "__reduce__"],
        ),
    ]:
        show(name, cls, required)


main()
