"""dict.update must notice source mutation from destination key equality."""


class MutatingKey:
    def __hash__(self):
        return 0

    def __eq__(self, other_key):
        source.clear()
        return False


source = {1: 0, MutatingKey(): 0}
destination = {MutatingKey(): 0, 1: 1}

try:
    destination.update(source)
except RuntimeError as error:
    assert str(error) == "dict mutated during update", str(error)
else:
    raise AssertionError("dict.update accepted a source mutated by __eq__")

print("ok")
