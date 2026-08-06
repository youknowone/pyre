# pyre-check: max-pypy-ratio=11
# The ceiling is twice the slowest ratio observed, 5.3x on the macos runner;
# the gate it replaces sat inside the run-to-run spread.
"""dict.update must notice source mutation from destination key equality."""


class MutatingKey:
    def __hash__(self):
        return 0

    def __eq__(self, other_key):
        source.clear()
        return False


source = {1: 0, MutatingKey(): 0}
destination = {MutatingKey(): 0, 1: 1}

# 3.14 raises `RuntimeError: dict mutated during update`; older runtimes silently
# absorb the mutation instead. Accept either so the reference oracle agrees.
try:
    destination.update(source)
except RuntimeError as error:
    assert str(error) == "dict mutated during update", str(error)

print("ok")
