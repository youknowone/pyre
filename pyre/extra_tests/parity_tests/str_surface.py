"""PyPy W_UnicodeObject TypeDef parity, with Python 3.14's doc surface."""


assert {"__doc__", "__hash__", "__rmod__", "__rmul__"} <= set(str.__dict__)
assert str.__hash__("abc") == hash("abc")
assert str.__rmul__("ab", 3) == "ababab"
assert str.__rmod__("value=%s", 3) is NotImplemented
assert str.__rmod__("value=%s", "%s") == "value=%s"
assert str.__doc__.startswith("str(object='') -> str")


class Index:
    def __index__(self):
        return 2


assert str.__rmul__("xy", Index()) == "xyxy"
print("str surface: ok")
