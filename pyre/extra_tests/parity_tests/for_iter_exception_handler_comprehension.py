# CPython-suite gap: comprehension tests cannot trace FOR_ITER exception handlers.
# parity-tests reason: this guards pyre's JIT comprehension handler control flow.

class Payload:
    def __init__(self, value):
        self.value = value


def make_payload(value):
    return Payload(value * 3 + 1)


def run(items):
    out = []
    seen = []
    for index in items:
        try:
            items[index + 100]
        except IndexError:
            out.extend([str(make_payload(value)) for value in range(1)])
        seen.append(len(range(index)))
        out.append(index)
    return len(out), len(seen)


items = [index % 5 for index in range(60)]
for _ in range(400):
    assert run(items) == (120, 60)

print("OK")
